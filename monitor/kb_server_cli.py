#!/usr/bin/env python
"""
ChromaDB Server Command Line Interface
Usage:
    python kb_server_cli.py start [--host HOST] [--port PORT] [--path PATH] [--force]
    python kb_server_cli.py stop [--force]
    python kb_server_cli.py restart [--host HOST] [--port PORT] [--path PATH]
    python kb_server_cli.py status
    python kb_server_cli.py delete [--collection NAME] [--metadata KEY:VALUE] [--force]
    python kb_server_cli.py list [--collection NAME] [--limit LIMIT]
    python kb_server_cli.py export [--collection NAME] [--output FILE] [--include-embeddings]
    python kb_server_cli.py import [--file FILE] [--collection NAME] [--overwrite]
    python kb_server_cli.py merge --collection NAME [--destination NAME] [--include-embeddings]
"""
import atexit
import argparse
import os
import sys
import json
import traceback
import fnmatch
from typing import Dict, Optional, List, Any, Callable, Tuple, Set
from datetime import datetime
from functools import wraps
import warnings

import chromadb
from chromadb.utils.embedding_functions import (
   SentenceTransformerEmbeddingFunction, OpenAIEmbeddingFunction)

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rome.kb_server import ChromaServerManager
from rome.logger import get_logger
from rome.config import LONG_SUMMARY_LEN, LONGEST_SUMMARY_LEN
from rome.kb_client import EMBEDDING_MODELS

logger = get_logger()
logger.configure({
    "level": "DEBUG",
    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    "console": True,
    "include_caller_info": "rome"
})

DEFAULT_CONFIG = {'host': 'localhost', 'port': 8000, 'persist_path': None}
SERVER_CONFIG = DEFAULT_CONFIG.copy()

def parse_collection_patterns(pattern_str: str) -> List[str]:
    """Parse comma-separated collection patterns"""
    return [p.strip() for p in pattern_str.split(',') if p.strip()]

def match_collections(patterns: List[str], available_collections: List) -> Set[str]:
    """
    Match collection patterns against available collections

    Args:
        patterns: List of collection names or wildcard patterns
        available_collections: List of collection objects with .name attribute

    Returns:
        Set of matched collection names
    """
    collection_names = [col.name for col in available_collections]
    matched = set()

    for pattern in patterns:
        if '*' in pattern or '?' in pattern or '[' in pattern:
            # Wildcard pattern matching
            matches = fnmatch.filter(collection_names, pattern)
            matched.update(matches)
        else:
            # Exact match
            if pattern in collection_names:
                matched.add(pattern)

    return matched


def resolve_target_collections(client, collection_arg: Optional[str]):
    """List collections on `client` and resolve a `--collection` arg against
    them. Centralizes the empty-DB / no-match guards that were duplicated
    across handle_list / handle_export / handle_delete / handle_merge.

    Returns (target_collections, all_collections):
      - both lists populated on success.
      - (None, []) if the database has no collections (caller decides
        whether that's 0 or 1; empty-DB message already printed).
      - (None, all_cols) if `collection_arg` didn't match anything (the
        '❌ No collections matched' message is already printed; caller
        should return 1).
    """
    all_collections = client.list_collections()
    if not all_collections:
        print("📄 No collections found in database")
        return None, []
    if collection_arg:
        patterns = parse_collection_patterns(collection_arg)
        matched = match_collections(patterns, all_collections)
        if not matched:
            print(f"❌ No collections matched pattern(s): {collection_arg}")
            return None, all_collections
        target = [c for c in all_collections if c.name in matched]
    else:
        target = list(all_collections)
    return target, all_collections

def compact_json_dump(data, file, indent=4):
    """Custom JSON dump that keeps embedding lists on single lines"""
    def format_item(obj, level=0):
        ind = ' ' * (indent * level)
        next_ind = ' ' * (indent * (level + 1))

        if isinstance(obj, dict):
            if not obj:
                return '{}'
            items = []
            for k, v in obj.items():
                # Special handling for 'embedding' key - keep list compact
                if k == 'embedding' and isinstance(v, list):
                    items.append(f'{next_ind}"{k}": {json.dumps(v)}')
                else:
                    items.append(f'{next_ind}"{k}": {format_item(v, level + 1)}')
            return '{\n' + ',\n'.join(items) + f'\n{ind}}}'
        elif isinstance(obj, list):
            if not obj:
                return '[]'
            # Check if this is a list of simple items or complex objects
            if all(not isinstance(item, (dict, list)) for item in obj):
                return json.dumps(obj)
            items = [format_item(item, level + 1) for item in obj]
            return '[\n' + ',\n'.join(f'{next_ind}{item}' for item in items) + f'\n{ind}]'
        else:
            return json.dumps(obj, ensure_ascii=False)

    file.write(format_item(data))

def error_handler(func: Callable) -> Callable:
    """Decorator for consistent error handling across commands"""
    @wraps(func)
    def wrapper(args):
        try:
            return func(args)
        except KeyboardInterrupt:
            print("\n⚠️  Interrupted by user")
            return 130
        except Exception as e:
            print(f"❌ Error in {func.__name__}: {e}")
            traceback.print_exc()
            return 1
    return wrapper

def requires_server(func: Callable) -> Callable:
    """Decorator to ensure server is running before executing command"""
    @wraps(func)
    def wrapper(args):
        manager = get_server_manager()
        if not manager.is_running():
            print("❌ ChromaDB server is not running")
            print("   Use 'python kb_server_cli.py start' to start the server first")
            return 1
        return func(args)
    return wrapper

def confirm_action(message: str) -> bool:
    """Ask for user confirmation"""
    try:
        response = input(f"{message} (Y/n): ").strip().lower()
        return response not in ['n', 'no']
    except KeyboardInterrupt:
        print("\nCancelled.")
        return False

def parse_metadata_filter(metadata_str: str) -> Tuple[str, Any]:
    """Parse metadata filter string in format 'key:value' with type conversion"""
    if ':' not in metadata_str:
        raise ValueError(f"Metadata filter must be in format 'key:value', got: {metadata_str}")

    parts = metadata_str.split(':', 1)
    if len(parts) != 2:
        raise ValueError(f"Invalid metadata filter format: {metadata_str}")

    key, value_str = parts[0].strip(), parts[1].strip()
    if not key:
        raise ValueError("Metadata key cannot be empty")

    value = _convert_value_type(value_str)
    return key, value

def _convert_value_type(value_str: str) -> Any:
    """Convert string value to ChromaDB-compatible type (str, int, float, bool only)"""
    if value_str in ('True', 'true'):
        return True
    elif value_str in ('False', 'false'):
        return False

    try:
        if '.' not in value_str and value_str.lstrip('-').isdigit():
            return int(value_str)
    except ValueError:
        pass

    try:
        if '.' in value_str:
            return float(value_str)
    except ValueError:
        pass

    return value_str

class ServerManager:
    """Unified server management interface"""

    def __init__(self, config: Dict = None):
        self._manager = None
        self._client = None

    def get_manager(self, detached: bool = False):
        """Get server manager instance"""
        if detached:
            return self._get_detached_manager()
        return ChromaServerManager.get_instance(SERVER_CONFIG)

    def _get_detached_manager(self):
        """Get detached server manager that doesn't auto-cleanup"""
        original_register = atexit.register
        atexit.register = lambda *args, **kwargs: None
        try:
            manager = ChromaServerManager.get_instance(SERVER_CONFIG)
            manager._shutdown_registered = True
            return manager
        finally:
            atexit.register = original_register

    def get_client(self):
        """Get ChromaDB client"""
        if not self._client:
            manager = self.get_manager()
            self._client = chromadb.HttpClient(host=manager.host, port=manager.port)
        return self._client

# Global server manager instance
server_mgr = ServerManager()
get_server_manager = lambda: server_mgr.get_manager()
get_client = lambda: server_mgr.get_client()

class CollectionManager:
    """Handles collection operations with batching and metadata filtering"""

    @staticmethod
    def delete_documents_by_metadata(collection, metadata_key: str, metadata_value: Any,
                                   batch_size: int = 100) -> Tuple[int, int]:
        """
        Delete documents matching metadata filter

        Returns:
            Tuple of (deleted_count, total_checked)
        """
        total_count = collection.count()
        deleted_count = 0
        total_checked = 0

        if total_count == 0:
            return 0, 0

        print(f"   Scanning {total_count} documents for metadata {metadata_key}:{metadata_value}")

        for offset in range(0, total_count, batch_size):
            result = collection.get(
                limit=batch_size,
                offset=offset,
                include=['metadatas']
            )

            batch_ids = result.get('ids', [])
            batch_metadatas = result.get('metadatas', [])

            matching_ids = []
            for i, metadata in enumerate(batch_metadatas):
                total_checked += 1
                if metadata and metadata.get(metadata_key) == metadata_value:
                    matching_ids.append(batch_ids[i])

            if matching_ids:
                collection.delete(ids=matching_ids)
                deleted_count += len(matching_ids)
                print(f"   Deleted {len(matching_ids)} documents from batch (offset {offset})")

        return deleted_count, total_checked

    @staticmethod
    def export_to_dict(collection, include_embeddings: bool = False, batch_size: int = 100) -> Dict[str, Any]:
        """Export collection to dictionary with batching"""
        total_count = collection.count()
        export_data = {'collection_name': collection.name, 'count': total_count, 'data': []}

        include_fields = ['documents', 'metadatas'] + (['embeddings'] if include_embeddings else [])

        for offset in range(0, total_count, batch_size):
            results = collection.get(limit=batch_size, offset=offset, include=include_fields)

            for i in range(len(results['ids'])):
                item = {
                    'id': results['ids'][i],
                    'document': results['documents'][i] if results['documents'] else None,
                    'metadata': results['metadatas'][i] if results['metadatas'] else None
                }
                if include_embeddings and results.get('embeddings') is not None:
                    embedding = results['embeddings'][i]
                    # Convert numpy array to list for JSON serialization
                    item['embedding'] = embedding.tolist() if hasattr(embedding, 'tolist') else embedding
                export_data['data'].append(item)

        return export_data

    @staticmethod
    def import_from_dict(client, data: Dict[str, Any], target_name: str = None,
                        embedding_model: str = None, overwrite: bool = False,
                        batch_size: int = 100) -> bool:
        """Import collection from dictionary with optional embedding model"""
        collection_name = target_name or data['collection_name']

        # Setup embedding function if model specified
        embedding_fn = None
        if embedding_model:
            expected_dim = EMBEDDING_MODELS.get(embedding_model)
            if not expected_dim:
                print(f"   Error: Invalid embedding model '{embedding_model}'")
                return False

            is_sentence_transformer = expected_dim == 384

            if is_sentence_transformer:
                embedding_fn = SentenceTransformerEmbeddingFunction(model_name=embedding_model)
            else:
                if not os.getenv('OPENAI_API_KEY'):
                    print(f"   Error: OPENAI_API_KEY required for {embedding_model}")
                    return False
                embedding_fn = OpenAIEmbeddingFunction(
                    model_name=embedding_model, api_key=os.getenv('OPENAI_API_KEY'))

        # Handle existing collection
        try:
            existing_collections = [col.name for col in client.list_collections()]
            if collection_name in existing_collections:
                if not overwrite and not confirm_action(f"   Collection '{collection_name}' already exists. Overwrite?"):
                    print(f"   Skipping import for '{collection_name}'")
                    return False
                print(f"   Deleting existing collection '{collection_name}'")
                client.delete_collection(collection_name)
        except Exception as e:
            print(f"   Warning: Could not check existing collections: {e}")

        # Create collection
        try:
            if embedding_fn:
                collection = client.create_collection(collection_name, embedding_function=embedding_fn)
                print(f"   Created collection '{collection_name}' with {embedding_model}")
            else:
                collection = client.create_collection(collection_name)
                print(f"   Created collection '{collection_name}'")

            total_items = len(data['data'])
            imported_count = 0

            for start_idx in range(0, total_items, batch_size):
                batch_data = data['data'][start_idx:start_idx + batch_size]
                batch_kwargs = CollectionManager._prepare_batch(batch_data)

                collection.add(**batch_kwargs)
                imported_count += len(batch_data)

                if total_items > batch_size:
                    print(f"   Imported {imported_count}/{total_items} documents...")

            print(f"   Successfully imported {imported_count} documents to '{collection_name}'")
            return True

        except Exception as e:
            print(f"   Error importing collection '{collection_name}': {e}")
            return False

    @staticmethod
    def _prepare_batch(batch_data: List[Dict]) -> Dict:
        """Prepare batch data for ChromaDB insertion"""
        batch_ids, batch_documents, batch_metadatas, batch_embeddings = [], [], [], []
        has_documents = has_metadatas = has_embeddings = False

        for item in batch_data:
            batch_ids.append(item['id'])

            if item.get('document') is not None:
                batch_documents.append(item['document'])
                has_documents = True
            else:
                batch_documents.append('')

            if item.get('metadata') is not None:
                batch_metadatas.append(item['metadata'])
                has_metadatas = True
            else:
                batch_metadatas.append({})

            if item.get('embedding') is not None:
                batch_embeddings.append(item['embedding'])
                has_embeddings = True

        kwargs = {'ids': batch_ids}
        if has_documents: kwargs['documents'] = batch_documents
        if has_metadatas: kwargs['metadatas'] = batch_metadatas
        if has_embeddings: kwargs['embeddings'] = batch_embeddings

        return kwargs

class OutputFormatter:
    """Handles all output formatting consistently"""

    @staticmethod
    def print_status_and_info(status: Dict, db_info: Dict):
        """Print combined status and database information"""
        if status['running']:
            print(f"✅ ChromaDB server is RUNNING")
            for key in ['URL', 'PID', 'Active clients', 'Host', 'Port', 'Startup timeout', 'Shutdown timeout']:
                value = status.get(key.lower().replace(' ', '_'), status.get(key.lower()))
                if value is not None:
                    unit = 's' if 'timeout' in key.lower() else ''
                    print(f"   {key}: {value}{unit}")
        else:
            print(f"❌ ChromaDB server is NOT RUNNING")
            print(f"   Would run at: {status['url']}")

        print(f"\n📊 Database Information")
        print(f"   Data path: {db_info['persist_path']}")
        print(f"   Running: {'Yes' if db_info['running'] else 'No'}")

        if 'error' in db_info:
            print(f"   Error: {db_info['error']}")
        elif 'collections' in db_info:
            print(f"   Total collections: {db_info['total_collections']}")
            if db_info['collections']:
                print(f"   Collections:")
                sorted_collections = sorted(db_info['collections'], key=lambda x: x['name'])
                for col in sorted_collections:
                    dimension_info = OutputFormatter._get_collection_dimensions(col.get('name'))
                    dim_str = f" | {dimension_info}d" if dimension_info else ""
                    print(f"     - {col['name']} ({col['count']} documents{dim_str})")
            else:
                print(f"   Collections: None")

    @staticmethod
    def _get_collection_dimensions(collection_name: str) -> str:
        """Get embedding dimensions for a collection"""
        try:
            client = get_client()
            collection = client.get_collection(collection_name)

            if collection.count() == 0:
                return "empty"

            result = collection.get(limit=1, include=["embeddings"])
            embeddings = result.get("embeddings")

            if embeddings is None:
                return "no embeddings"

            if hasattr(embeddings, 'tolist'):
                embeddings = embeddings.tolist()

            if len(embeddings) > 0 and embeddings[0] is not None:
                return str(len(embeddings[0]))
            else:
                return "unknown"

        except Exception as e:
            return "error"

    @staticmethod
    def print_documents(documents: Dict, collection_name: str = None, limit: int = None):
        """Print formatted document information"""
        title = f"Documents in collection '{collection_name}'" if collection_name else "All documents"
        print(f"📄 {title}:")

        total_docs = displayed_docs = 0

        for col_name, docs in documents.items():
            if collection_name and col_name != collection_name:
                continue

            doc_count = len(docs)
            total_docs += doc_count

            if not collection_name:
                dimension_info = OutputFormatter._get_collection_dimensions(col_name)
                dim_str = f" | {dimension_info}d" if dimension_info and dimension_info not in ["empty", "error"] else ""
                print(f"\n┌── Collection: {col_name} ({doc_count} documents{dim_str}) ──┐")

            for doc in docs:
                if limit and displayed_docs >= limit:
                    print(f"\n💡 Showing first {limit} of {total_docs} total documents")
                    return

                OutputFormatter._print_single_document(doc, displayed_docs + 1)
                displayed_docs += 1

        print(f"\n{'─' * 80}")
        if displayed_docs == 0:
            target = f"collection '{collection_name}'" if collection_name else "any collection"
            print(f"   No documents found in {target}")
        else:
            if collection_name:
                print(f"   Total: {displayed_docs} documents in '{collection_name}'")
            else:
                collections_shown = len([name for name in documents.keys() if not collection_name or name == collection_name])
                print(f"   Total: {displayed_docs} documents across {collections_shown} collections")

    @staticmethod
    def _print_single_document(doc: Dict, doc_num: int):
        """Print a single document with formatting"""
        doc_id = doc.get('id', 'Unknown')
        content = doc.get('document', '')
        metadata = doc.get('metadata', {})

        maxlen = LONGEST_SUMMARY_LEN
        if content and len(content) > maxlen:
            truncated = content[:maxlen]
            last_space = truncated.rfind(' ')
            if last_space > maxlen * 0.8:
                content_preview = truncated[:last_space] + "..."
            else:
                content_preview = truncated + "..."
        else:
            content_preview = content

        print(f"\n📋 Document #{doc_num}")
        print(f"   ID: {doc_id}")

        if content_preview:
            print(f"   Content:")
            for line in content_preview.split('\n'):
                print(f"     {line}")
        else:
            print(f"   Content: (empty)")

        if metadata:
            print(f"   Metadata:")
            for key, value in sorted(metadata.items()):
                if isinstance(value, (dict, list)):
                    value_str = json.dumps(value, indent=4)
                    print(f"     {key}: {value_str}")
                else:
                    print(f"     {key}: {value}")

def create_parser() -> argparse.ArgumentParser:
    """Create modular argument parser"""
    parser = argparse.ArgumentParser(
        description="ChromaDB Server Management CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Start server with defaults
    python kb_server_cli.py start

    # Start server on specific host/port with custom data path
    python kb_server_cli.py start --host 0.0.0.0 --port 8001 --path ./my_chroma_data

    # Stop server gracefully
    python kb_server_cli.py stop

    # Force stop server (kills process)
    python kb_server_cli.py stop --force

    # Restart server
    python kb_server_cli.py restart

    # Check server status and database info
    python kb_server_cli.py status

    # Delete all data (--force skips confirmation)
    python kb_server_cli.py delete --force

    # Delete specific collection
    python kb_server_cli.py delete --collection my_collection

    # Delete multiple collections with wildcards
    python kb_server_cli.py delete --collection "temp_*,test_*"

    # Delete specific collection with metadata filtering
    python kb_server_cli.py delete --collection my_collection --metadata key:value

    # List up to 10 documents in matching collections
    python kb_server_cli.py list --collection "data_*" --limit 10

    # Export specific collections with embeddings to JSON file
    python kb_server_cli.py export --collection "prod_*" --include-embeddings --output my_export.json

    # Import from JSON file and overwrite existing collection
    python kb_server_cli.py import --file my_export.json --overwrite

    # Merge multiple collections into a destination collection
    python kb_server_cli.py merge --collection "data_*,test_*" --destination merged_collection

    # Merge with embeddings preserved
    python kb_server_cli.py merge --collection "run1,run2,run3" --destination combined --include-embeddings
"""
    )

    subparsers = parser.add_subparsers(dest='command', help='Available commands')

    def add_server_args(parser):
        parser.add_argument('--host', default=DEFAULT_CONFIG['host'], help='Server host')
        parser.add_argument('--port', type=int, default=DEFAULT_CONFIG['port'], help='Server port')
        parser.add_argument('--path', default=DEFAULT_CONFIG['persist_path'], help='Data persistence path')

    def add_collection_arg(parser):
        parser.add_argument(
            '--collection',
            help='Collection name(s) - supports wildcards (* ?) and comma-separated list (e.g., "data_*,test_*")'
        )

    def add_force_arg(parser):
        parser.add_argument('--force', action='store_true', help='Force action without confirmation')

    def add_batch_arg(parser):
        parser.add_argument('--batch-size', type=int, default=100, help='Batch size for operations')

    def add_metadata_arg(parser):
        parser.add_argument('--metadata', help='Filter by metadata key:value (e.g., source:web)')

    # Every subcommand gets --host/--port/--path so a non-default server
    # (e.g. the web_server's chroma at :8091) can be targeted explicitly.
    # Without this, `stop` / `status` / `list` would silently default to
    # port 8000 — and `stop` would kill whichever server happens to live
    # there, which may not be the one the user meant.
    commands = [
        ('start', 'Start ChromaDB server', [add_server_args, add_force_arg]),
        ('stop', 'Stop ChromaDB server', [add_server_args, add_force_arg]),
        ('restart', 'Restart ChromaDB server', [add_server_args]),
        ('status', 'Show server status', [add_server_args]),
        ('delete', 'Delete database, collection, or filtered documents',
         [add_server_args, add_collection_arg, add_metadata_arg, add_force_arg]),
        ('list', 'List documents',
         [add_server_args, add_collection_arg,
          lambda p: p.add_argument('--limit', type=int, help='Limit results')]),
        ('export', 'Export to JSON',
         [add_server_args, add_collection_arg, add_batch_arg,
          lambda p: p.add_argument('--output', help='Output file'),
          lambda p: p.add_argument('--include-embeddings', action='store_true')]),
        ('import', 'Import from JSON',
         [add_server_args, add_collection_arg, add_batch_arg,
          lambda p: p.add_argument('--file', required=True, help='Input file'),
          lambda p: p.add_argument('--overwrite', action='store_true')]),
        ('merge', 'Merge collections into a destination collection',
         [add_server_args, add_collection_arg, add_batch_arg,
          lambda p: p.add_argument('--destination', required=True, help='Destination collection name'),
          lambda p: p.add_argument('--include-embeddings', action='store_true',
                                   help='Preserve embeddings from source collections'),
          add_force_arg])
    ]

    for cmd, help_text, arg_funcs in commands:
        cmd_parser = subparsers.add_parser(cmd, help=help_text)
        for arg_func in arg_funcs:
            arg_func(cmd_parser)

    return parser

@error_handler
def handle_start(args):
    """Handle start command"""
    manager = server_mgr.get_manager(detached=True)

    if manager.is_running() and not args.force:
        print("✅ ChromaDB server is already running")
        OutputFormatter.print_status_and_info(manager.get_status(), manager.get_database_info())
        return 0

    print("🚀 Starting ChromaDB server...")
    if args.force:
        print("   Force restart requested")

    success = manager.start(force_restart=args.force)

    if success:
        print("✅ ChromaDB server started successfully")
        OutputFormatter.print_status_and_info(manager.get_status(), manager.get_database_info())
        print("🔗 Server running in background. Use 'python kb_server_cli.py stop' to stop.")
        return 0
    else:
        print("❌ Failed to start ChromaDB server")
        print(f"   Try manually: chroma run --host {manager.host} --port {manager.port} --path {manager.persist_path}")
        return 1

@error_handler
def handle_stop(args):
    """Handle stop command"""
    manager = get_server_manager()

    if not manager.is_running():
        print("ℹ️  ChromaDB server is not running")
        return 0

    print("🛑 Stopping ChromaDB server...")
    if args.force:
        print("   Force stop requested")

    success = manager.stop(force=args.force)
    return 0 if success else 1

@error_handler
def handle_restart(args):
    """Handle restart command"""
    manager = get_server_manager()
    print("🔄 Restarting ChromaDB server...")

    success = manager.restart()
    if success:
        print("✅ ChromaDB server restarted successfully")
        OutputFormatter.print_status_and_info(manager.get_status(), manager.get_database_info())

    return 0 if success else 1

@error_handler
def handle_status(args):
    """Handle status command"""
    manager = get_server_manager()
    OutputFormatter.print_status_and_info(manager.get_status(), manager.get_database_info())
    return 0

@error_handler
def handle_delete(args):
    """Enhanced delete with multi-collection and wildcard support"""
    manager = get_server_manager()

    if args.metadata and not args.collection:
        print("❌ --metadata filter requires --collection to be specified")
        return 1

    metadata_key = metadata_value = None
    if args.metadata:
        try:
            metadata_key, metadata_value = parse_metadata_filter(args.metadata)
        except ValueError as e:
            print(f"❌ Invalid metadata filter: {e}")
            return 1

    if args.metadata:
        if not manager.is_running():
            print("❌ ChromaDB server is not running")
            return 1

        try:
            client = get_client()
            target_collections, _ = resolve_target_collections(client, args.collection)
            if target_collections is None:
                return 1
            matched = {c.name for c in target_collections}

            if len(matched) > 1:
                print(f"📋 Matched collections: {', '.join(sorted(matched))}")

            if not args.force:
                message = f"Delete documents in {len(matched)} collection(s) where {metadata_key}={metadata_value}?"
                if not confirm_action(message):
                    return 0

            total_deleted = total_checked = 0
            for collection_name in sorted(matched):
                collection = client.get_collection(collection_name)
                print(f"\n🗑️  Processing collection '{collection_name}'...")

                deleted, checked = CollectionManager.delete_documents_by_metadata(
                    collection, metadata_key, metadata_value
                )
                total_deleted += deleted
                total_checked += checked

                if deleted > 0:
                    print(f"   ✅ Deleted {deleted} documents")
                else:
                    print(f"   ℹ️  No matching documents found")

            print(f"\n✅ Summary: Deleted {total_deleted} documents across {len(matched)} collection(s) (checked {total_checked} total)")
            return 0

        except Exception as e:
            print(f"❌ Error deleting documents: {e}")
            return 1

    elif args.collection:
        if not manager.is_running():
            print("❌ ChromaDB server is not running")
            return 1

        try:
            client = get_client()
            target_collections, _ = resolve_target_collections(client, args.collection)
            if target_collections is None:
                return 1
            matched = {c.name for c in target_collections}

            print(f"📋 Matched collections: {', '.join(sorted(matched))}")

            if not args.force:
                message = f"Delete {len(matched)} collection(s)?"
                if not confirm_action(message):
                    return 0

            success_count = 0
            for collection_name in sorted(matched):
                print(f"🗑️  Deleting collection '{collection_name}'...")
                if manager.delete_collection(collection_name):
                    print(f"   ✅ Deleted successfully")
                    success_count += 1
                else:
                    print(f"   ❌ Failed to delete")

            print(f"\n✅ Deleted {success_count}/{len(matched)} collections")
            return 0 if success_count == len(matched) else 1

        except Exception as e:
            print(f"❌ Error: {e}")
            return 1
    else:
        if not args.force and not confirm_action("Delete ALL database data? This cannot be undone!"):
            return 0
        print("🗑️  Deleting entire database...")
        success = manager.clear_database(force=True)
        print(f"✅ Database deleted successfully" if success else f"❌ Failed to delete database")
        return 0 if success else 1

@error_handler
@requires_server
def handle_list(args):
    """Handle list with multi-collection and wildcard support"""
    target_collections, all_collections = resolve_target_collections(
        get_client(), args.collection)
    if target_collections is None:
        # Empty DB → success (already printed); no match → 1.
        return 0 if not all_collections else 1

    documents = {}
    for collection in target_collections:
        try:
            result = collection.get(include=['documents', 'metadatas'])
            documents[collection.name] = [
                {
                    'id': result['ids'][i],
                    'document': result['documents'][i] if result['documents'] else '',
                    'metadata': result['metadatas'][i] if result['metadatas'] else {}
                }
                for i in range(len(result['ids']))
            ]
        except Exception as e:
            print(f"⚠️  Error reading collection '{collection.name}': {e}")

    if not documents:
        print("📄 No documents found")
        return 0

    OutputFormatter.print_documents(documents, None, args.limit)
    return 0

@error_handler
@requires_server
def handle_export(args):
    """Handle export with multi-collection and wildcard support"""
    target_collections, all_collections = resolve_target_collections(
        get_client(), args.collection)
    if target_collections is None:
        return 0 if not all_collections else 1

    suffix = args.collection.replace(',', '_').replace('*', 'w') if args.collection else "all"

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file = args.output or f"chroma_export_{suffix}_{timestamp}.json"

    print(f"📤 Exporting to '{output_file}'...")
    if args.include_embeddings:
        print("   Including embeddings in export")

    if len(target_collections) > 1:
        print(f"   Exporting {len(target_collections)} collections: {', '.join(col.name for col in target_collections)}")

    export_data = {
        'export_type': 'filtered' if args.collection else 'all_collections',
        'collections': []
    }

    for collection in target_collections:
        print(f"   Exporting collection '{collection.name}'...")
        collection_data = CollectionManager.export_to_dict(
            collection, args.include_embeddings, args.batch_size
        )

        dimension = OutputFormatter._get_collection_dimensions(collection.name)
        embedding_model = None
        if dimension.isdigit():
            dim = int(dimension)
            embedding_model = next((model for model, d in EMBEDDING_MODELS.items() if d == dim), None)

        collection_data['embedding_model'] = embedding_model
        export_data['collections'].append(collection_data)

    with open(output_file, 'w', encoding='utf-8') as f:
        compact_json_dump(export_data, f, indent=4)

    total_collections = len(export_data['collections'])
    total_documents = sum(col['count'] for col in export_data['collections'])

    print(f"✅ Export completed successfully")
    print(f"   File: {output_file}")
    print(f"   Collections: {total_collections}")
    print(f"   Total documents: {total_documents}")
    print(f"   Embeddings included: {'Yes' if args.include_embeddings else 'No'}")

    return 0

@error_handler
@requires_server
def handle_import(args):
    """Handle import command with embedding model validation"""
    if not os.path.exists(args.file):
        print(f"❌ Input file '{args.file}' not found")
        return 1

    print(f"📥 Importing from '{args.file}'...")

    with open(args.file, 'r', encoding='utf-8') as f:
        import_data = json.load(f)

    if 'collections' not in import_data:
        print("❌ Invalid import file format: missing 'collections' key")
        return 1

    client = get_client()
    imported_collections = total_documents = 0

    for collection_data in import_data['collections']:
        if 'collection_name' not in collection_data:
            print("⚠️  Skipping collection with missing name")
            continue

        collection_name = args.collection or collection_data['collection_name']
        embedding_model = collection_data.get('embedding_model')

        if embedding_model:
            print(f"   Using embedding model: {embedding_model}")
            expected_dim = EMBEDDING_MODELS.get(embedding_model)
            if not expected_dim:
                print(f"   ⚠️  Unknown embedding model: {embedding_model}")

        print(f"   Importing collection '{collection_name}'...")

        if embedding_model and embedding_model in EMBEDDING_MODELS:
            success = CollectionManager.import_from_dict(
                client, collection_data, collection_name, embedding_model,
                args.overwrite, args.batch_size
            )
        else:
            success = CollectionManager.import_from_dict(
                client, collection_data, collection_name, None,
                args.overwrite, args.batch_size
            )

        if success:
            imported_collections += 1
            total_documents += collection_data.get('count', 0)

    if imported_collections > 0:
        print(f"✅ Import completed successfully")
        print(f"   Collections imported: {imported_collections}")
        print(f"   Total documents: {total_documents}")
        return 0
    else:
        print("❌ No collections were imported")
        return 1

@error_handler
@requires_server
def handle_merge(args):
    """Merge documents from source collections into a destination collection"""
    if not args.collection:
        print("❌ --collection is required to specify source collections")
        return 1

    client = get_client()
    target_collections, all_collections = resolve_target_collections(
        client, args.collection)
    if target_collections is None:
        # For merge, empty DB is also a failure case.
        return 1

    # Don't merge a collection into itself.
    dest_in_sources = any(c.name == args.destination for c in target_collections)
    target_collections = [c for c in target_collections if c.name != args.destination]
    if not target_collections:
        print("❌ No source collections remaining after excluding destination")
        return 1
    if dest_in_sources:
        print(f"   ℹ️  Excluding destination '{args.destination}' from sources")

    matched = {c.name for c in target_collections}
    source_names = sorted(matched)
    print(f"📋 Source collections: {', '.join(source_names)}")
    print(f"📋 Destination: {args.destination}")

    # Count total documents across sources
    total_docs = 0
    for col in all_collections:
        if col.name in matched:
            total_docs += col.count()
    print(f"   Total documents to merge: {total_docs}")

    if not args.force:
        if not confirm_action(f"Merge {len(source_names)} collection(s) ({total_docs} documents) into '{args.destination}'?"):
            return 0

    # Get or create destination collection
    existing_names = [col.name for col in all_collections]
    dest_exists = args.destination in existing_names

    include_fields = ['documents', 'metadatas']
    if args.include_embeddings:
        include_fields.append('embeddings')
        print("   Including embeddings in merge")

    if dest_exists:
        dest_collection = client.get_collection(args.destination)
        print(f"   Destination '{args.destination}' exists ({dest_collection.count()} documents)")
    else:
        dest_collection = client.create_collection(args.destination)
        print(f"   Created destination collection '{args.destination}'")

    merged_count = 0
    skipped_count = 0

    # Get existing IDs in destination to avoid duplicates
    dest_existing_ids = set()
    if dest_exists:
        dest_total = dest_collection.count()
        for offset in range(0, dest_total, args.batch_size):
            result = dest_collection.get(limit=args.batch_size, offset=offset)
            dest_existing_ids.update(result['ids'])

    for source_name in source_names:
        source_collection = client.get_collection(source_name)
        source_count = source_collection.count()
        print(f"\n🔀 Merging '{source_name}' ({source_count} documents)...")

        for offset in range(0, source_count, args.batch_size):
            results = source_collection.get(
                limit=args.batch_size, offset=offset, include=include_fields
            )

            # Filter out documents that already exist in destination
            batch_ids = []
            batch_documents = []
            batch_metadatas = []
            batch_embeddings = []

            for i in range(len(results['ids'])):
                doc_id = results['ids'][i]
                if doc_id in dest_existing_ids:
                    skipped_count += 1
                    continue

                batch_ids.append(doc_id)
                if results.get('documents'):
                    batch_documents.append(results['documents'][i])
                if results.get('metadatas'):
                    batch_metadatas.append(results['metadatas'][i])
                if args.include_embeddings and results.get('embeddings') is not None:
                    embedding = results['embeddings'][i]
                    batch_embeddings.append(
                        embedding.tolist() if hasattr(embedding, 'tolist') else embedding
                    )

            if not batch_ids:
                continue

            add_kwargs = {'ids': batch_ids}
            if batch_documents:
                add_kwargs['documents'] = batch_documents
            if batch_metadatas:
                add_kwargs['metadatas'] = batch_metadatas
            if batch_embeddings:
                add_kwargs['embeddings'] = batch_embeddings

            dest_collection.add(**add_kwargs)
            dest_existing_ids.update(batch_ids)
            merged_count += len(batch_ids)

            if source_count > args.batch_size:
                print(f"   Merged {min(offset + args.batch_size, source_count)}/{source_count} documents...")

        print(f"   ✅ Done with '{source_name}'")

    print(f"\n✅ Merge completed successfully")
    print(f"   Destination: {args.destination} ({dest_collection.count()} documents)")
    print(f"   Documents merged: {merged_count}")
    if skipped_count > 0:
        print(f"   Duplicates skipped: {skipped_count}")

    return 0

def main():
    """Main CLI entry point"""
    parser = create_parser()
    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return 1

    global SERVER_CONFIG
    for key in ['host', 'port', 'path']:
        if hasattr(args, key) and getattr(args, key) is not None:
            SERVER_CONFIG[key if key != 'path' else 'persist_path'] = getattr(args, key)

    commands = {
        'start': handle_start, 'stop': handle_stop, 'restart': handle_restart,
        'status': handle_status, 'delete': handle_delete, 'list': handle_list,
        'export': handle_export, 'import': handle_import, 'merge': handle_merge
    }

    handler = commands.get(args.command)
    if not handler:
        print(f"❌ Unknown command: {args.command}")
        parser.print_help()
        return 1

    return handler(args)

if __name__ == "__main__":
    sys.exit(main())
