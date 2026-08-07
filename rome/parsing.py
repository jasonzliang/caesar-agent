import ast
import hashlib
import json
import re
from typing import Dict, Optional, Any, Union, List
import zlib

from .config import DEFAULT_HASH_FUNC
from .logger import get_logger

# Simple global cache - you could also make this a class attribute
_ast_cache: Dict[str, ast.AST] = {}


def hash_string(content: str, hash_name: str = DEFAULT_HASH_FUNC) -> str:
    """
    Fast hash function with multiple algorithm support.

    Args:
        content: String content to hash
        hash_name: Hash algorithm name

    Returns:
        String hash value (hexadecimal)
    """
    try:
        import xxhash
    except:
        hash_name = "md5" # Fallback hash if xxhash not installed

    if hash_name in ["xxh64", "xxh128"]:
        hash_func = getattr(xxhash, hash_name)
        return hash_func(content.encode('utf-8')).hexdigest()
    elif hash_name in ["md5", "sha1", "sha256", "sha512", "blake2b", "blake2s"]:
        hash_func = getattr(hashlib, hash_name)
        return hash_func(content.encode('utf-8')).hexdigest()
    elif hash_name == "crc32":
        return f"{zlib.crc32(content.encode('utf-8')) & 0xffffffff:08x}"
    else:
        raise ValueError(f"Unsupported hash function: {hash_name}")


def parse_code_cached(code_content: str, hash_name: str = DEFAULT_HASH_FUNC) -> ast.AST:
    """
    Parse Python code with simple dictionary caching using configurable hash function.

    Args:
        code_content: Python source code as string
        hash_name: Hash algorithm to use for caching
                  Options: xxhash, crc32, md5, sha1, sha256, sha512, blake2b, blake2s

    Returns:
        Parsed AST tree
    """
    cache_key = hash_string(code_content, hash_name)

    # Check cache first
    if cache_key in _ast_cache:
        return _ast_cache[cache_key]

    # Parse and cache
    tree = ast.parse(code_content)
    _ast_cache[cache_key] = tree
    return tree


def clear_ast_cache():
    """Clear the AST cache if memory usage becomes a concern."""
    global _ast_cache
    _ast_cache.clear()


def get_cache_stats():
    """Get simple cache statistics."""
    return {
        'cache_size': len(_ast_cache),
        'num_entries': len(_ast_cache)
    }


def parse_python_response(text: str) -> Optional[str]:
    """
    Extracts Python content from markdown code blocks with fallback to inline code.

    Args:
        text (str): The markdown text containing code blocks

    Returns:
        Optional[str]: The first Python code block content, or None if not found

    Note:
        - Preserves internal whitespace
        - Handles escaped backticks within code
        - Skips invalid/incomplete code blocks
        - Falls back to inline code with Python keywords
    """
    if not text:
        return None

    # First try to extract from code blocks
    pattern = r"```[ \t]*(\w+)?[ \t]*\r?\n(.*?)\r?\n[ \t]*```"
    matches = re.finditer(pattern, text, re.DOTALL)

    for match in matches:
        try:
            language = match.group(1)
            # Remove only the trailing newline if it exists, preserve other whitespace
            content = match.group(2).rstrip('\r\n')

            # Skip empty code blocks
            if not content.strip():
                continue
            if language and language.lower() == "python":
                return content
            # If no language specified, assume it might be Python if it looks like it
            elif not language and _looks_like_python(content):
                return content

        except (IndexError, AttributeError):
            # Skip malformed matches
            continue

    # Fallback: try to find inline code with Python keywords
    inline_pattern = r'`([^`]+)`'
    inline_matches = re.findall(inline_pattern, text)

    python_keywords = ['def ', 'import ', 'from ', 'class ', 'if ', 'for ', 'while ', 'return ']
    for match in inline_matches:
        if any(keyword in match for keyword in python_keywords) or '=' in match or 'print(' in match:
            return match.strip()

    return None


def _looks_like_python(content: str) -> bool:
    """Helper function to determine if content looks like Python code."""
    python_indicators = ['def ', 'import ', 'from ', 'class ', 'if ', 'for ', 'while ', 'return ', '=', 'print(']
    return any(indicator in content for indicator in python_indicators)


def extract_function_names(code: str, include_methods: bool = False) -> List[str]:
    """
    Extract function names from Python code efficiently.

    Args:
        code: Python source code
        include_methods: Include class methods in results

    Returns:
        List of function names (methods prefixed with 'ClassName.')
    """
    if not code or not code.strip():
        return []

    try:
        tree = parse_code_cached(code)
    except SyntaxError:
        return []

    functions = []

    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            # Find parent class if this is a method
            parent_class = None
            if include_methods:
                for parent in ast.walk(tree):
                    if (isinstance(parent, ast.ClassDef) and
                        any(child is node for child in ast.walk(parent))):
                        parent_class = parent.name
                        break

            name = f"{parent_class}.{node.name}" if parent_class else node.name
            if name not in functions:  # Avoid duplicates
                functions.append(name)

    return functions


def extract_function_from_code(code_string: str, function_name: str,
                             include_methods: bool = False) -> Optional[str]:
    """
    Extract function source code efficiently with robust fallbacks.

    Args:
        code_string: Source code string
        function_name: Name of function to extract (or 'Class.method')
        include_methods: Search in class methods

    Returns:
        Function source code or None if not found
    """
    if not code_string or not function_name:
        return None

    try:
        tree = parse_code_cached(code_string)
    except SyntaxError:
        return None

    # Handle class.method syntax
    if '.' in function_name and include_methods:
        class_name, method_name = function_name.split('.', 1)
        target_names = {method_name}
        in_target_class = False
    else:
        target_names = {function_name}
        class_name = method_name = None
        in_target_class = True

    for node in ast.walk(tree):
        # Check if we're in the right class context
        if class_name and isinstance(node, ast.ClassDef):
            in_target_class = node.name == class_name
            continue

        # Look for target function
        if (isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and
            node.name in target_names and (in_target_class or not class_name)):

            # Try precise extraction first
            try:
                source = ast.get_source_segment(code_string, node)
                if source:
                    return source
            except (TypeError, AttributeError):
                pass

            # Fallback to unparsing
            try:
                return ast.unparse(node)
            except AttributeError:
                return None

    return None


def extract_comments_from_code(text: str, include_single_comments: bool = False,
                              background_only: bool = False, background_index: int = 0) -> str:
    """
    Extract comment blocks from Python source code, including docstrings.

    Args:
        text (str): Input text containing Python code and comments
        include_single_comments (bool): Whether to include single-line # comments

    Returns:
        str: A string containing the extracted comment blocks joined by newlines
    """
    # Regex pattern to match:
    # 1. Text inside triple quotes (docstrings)
    # 2. Single-line comments starting with #, including those after code
    if include_single_comments:
        comment_pattern = r'(""".*?"""|\'\'\'.*?\'\'\'|(?:^|\s*)#[^\n]*)'
    else:
        comment_pattern = r'(""".*?"""|\'\'\'.*?\'\'\')'

    # Re-use flags for multiline and dot matching
    flags = re.MULTILINE | re.DOTALL

    # Find all comment blocks
    comments = re.findall(comment_pattern, text, flags)

    # Clean up the comments
    processed_comments = []
    for comment in comments:
        # Remove triple quotes from docstrings
        cleaned_comment = re.sub(r'^(\'\'\'|""")|((\'\'\'|""")$)', '', comment).strip()

        # Ensure # comments are preserved
        if cleaned_comment.startswith('#') or not cleaned_comment:
            cleaned_comment = comment.strip()

        if cleaned_comment:
            processed_comments.append(cleaned_comment)

    return "\n".join(processed_comments)


def parse_json_response(response: str) -> Union[Dict[str, Any], List[Any]]:
    """
    Parse JSON response with optimized approach for both direct JSON and extraction.

    Handles both responses from JSON mode (response_format={"type": "json_object"})
    and extracts JSON from text responses when needed.

    Args:
        response (str): The response text, either direct JSON or containing JSON fragments

    Returns:
        Union[Dict[str, Any], List[Any]]: The parsed JSON object/array or None if parsing failed
    """
    if not response or not response.strip():
        return None

    # STEP 1: Try direct JSON parsing first (for response_format="json_object" responses)
    try:
        parsed_json = json.loads(response)
        # Return the parsed JSON as is, whether it's a dict or a list
        return parsed_json
    except json.JSONDecodeError:
        # Direct parsing failed, move to extraction methods
        pass

    # STEP 2: Extract JSON from code blocks
    json_block_patterns = [
        r'```\s*json\s*\n([\s\S]*?)\n```',  # ```json\n...\n```
        r'```\n([\s\S]*?)\n```'             # ```\n...\n```
    ]

    for pattern in json_block_patterns:
        matches = re.findall(pattern, response, re.DOTALL)
        for match in matches:
            try:
                cleaned_match = match.strip()
                parsed_json = json.loads(cleaned_match)
                # Return the parsed JSON as is, whether it's a dict or a list
                return parsed_json
            except json.JSONDecodeError:
                continue

    # STEP 3: Extract JSON from free text (balanced bracket approach)
    # Find objects with properly balanced braces
    start_indexes = []
    candidate_objects = []
    depth = 0

    for i, char in enumerate(response):
        if char == '{':
            if depth == 0:
                start_indexes.append(i)
            depth += 1
        elif char == '}':
            depth -= 1
            if depth == 0 and start_indexes:
                candidate_objects.append(response[start_indexes.pop():i+1])
        # Also check for JSON arrays
        elif char == '[':
            if depth == 0:
                start_indexes.append(i)
            depth += 1
        elif char == ']':
            depth -= 1
            if depth == 0 and start_indexes:
                candidate_objects.append(response[start_indexes.pop():i+1])

    # Try to parse each candidate JSON object or array
    for candidate in candidate_objects:
        try:
            parsed_json = json.loads(candidate)
            # Return the parsed JSON regardless of type
            return parsed_json
        except json.JSONDecodeError:
            continue

    # No valid JSON found
    return None


def extract_all_definitions(content: str) -> List[Dict[str, str]]:
    """
    Extract both function and class definitions from Python code.

    Args:
        content (str): Python source code as a string

    Returns:
        List[Dict[str, str]]: List of dictionaries containing definition info.
                             Each dict has keys: 'name', 'signature', 'docstring', 'line_number', 'type'
                             Functions also have 'type': 'function'
                             Classes also have 'type': 'class' and 'methods': list
    """
    if not content:
        return []

    definitions = []

    try:
        # Parse the code into an AST
        tree = parse_code_cached(content)

        # Walk through all nodes in the AST
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                # Extract function
                func_name = node.name
                func_signature = _get_function_signature(node)
                docstring = ast.get_docstring(node) or ""
                line_number = node.lineno

                definitions.append({
                    'name': func_name,
                    'signature': func_signature,
                    'docstring': docstring.strip(),
                    'line_number': line_number,
                    'type': 'function'
                })

            elif isinstance(node, ast.ClassDef):
                # Extract class
                class_name = node.name
                class_signature = _get_class_signature(node)
                docstring = ast.get_docstring(node) or ""
                line_number = node.lineno

                # Extract method names
                methods = []
                for child in node.body:
                    if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        methods.append(child.name)

                definitions.append({
                    'name': class_name,
                    'signature': class_signature,
                    'docstring': docstring.strip(),
                    'line_number': line_number,
                    'type': 'class',
                    'methods': methods
                })

    except SyntaxError:
        # If AST parsing fails due to syntax errors, return empty list
        # Most malformed code won't benefit from regex extraction anyway
        return []

    # Sort by line number
    definitions.sort(key=lambda x: x['line_number'])
    return definitions


def _get_function_signature(func_node: ast.FunctionDef) -> str:
    """
    Extract the complete function signature from an AST FunctionDef node.

    Args:
        func_node (ast.FunctionDef): The function definition node

    Returns:
        str: The function signature as a string
    """
    # Start with function name
    signature = f"def {func_node.name}("

    # Handle arguments
    args = []

    # Regular arguments
    for arg in func_node.args.args:
        arg_str = arg.arg
        # Add type annotation if present
        if arg.annotation:
            arg_str += f": {ast.unparse(arg.annotation)}" if hasattr(ast, 'unparse') else f": {_ast_to_string(arg.annotation)}"
        args.append(arg_str)

    # *args
    if func_node.args.vararg:
        vararg_str = f"*{func_node.args.vararg.arg}"
        if func_node.args.vararg.annotation:
            vararg_str += f": {ast.unparse(func_node.args.vararg.annotation)}" if hasattr(ast, 'unparse') else f": {_ast_to_string(func_node.args.vararg.annotation)}"
        args.append(vararg_str)

    # **kwargs
    if func_node.args.kwarg:
        kwarg_str = f"**{func_node.args.kwarg.arg}"
        if func_node.args.kwarg.annotation:
            kwarg_str += f": {ast.unparse(func_node.args.kwarg.annotation)}" if hasattr(ast, 'unparse') else f": {_ast_to_string(func_node.args.kwarg.annotation)}"
        args.append(kwarg_str)

    signature += ", ".join(args)
    signature += ")"

    # Add return type annotation if present
    if func_node.returns:
        signature += f" -> {ast.unparse(func_node.returns)}" if hasattr(ast, 'unparse') else f" -> {_ast_to_string(func_node.returns)}"

    return signature


def _ast_to_string(node) -> str:
    """
    Convert an AST node to string representation (fallback for older Python versions).

    Args:
        node: AST node

    Returns:
        str: String representation of the node
    """
    if isinstance(node, ast.Name):
        return node.id
    elif isinstance(node, ast.Constant):
        return repr(node.value)
    elif isinstance(node, ast.Attribute):
        return f"{_ast_to_string(node.value)}.{node.attr}"
    elif isinstance(node, ast.Subscript):
        return f"{_ast_to_string(node.value)}[{_ast_to_string(node.slice)}]"
    else:
        # For other types, return a generic representation
        return str(type(node).__name__)


def _get_class_signature(class_node: ast.ClassDef) -> str:
    """
    Extract the complete class signature from an AST ClassDef node.

    Args:
        class_node (ast.ClassDef): The class definition node

    Returns:
        str: The class signature as a string
    """
    # Start with class name
    signature = f"class {class_node.name}"

    # Handle inheritance
    if class_node.bases or class_node.keywords:
        args = []

        # Base classes
        for base in class_node.bases:
            if hasattr(ast, 'unparse'):
                args.append(ast.unparse(base))
            else:
                args.append(_ast_to_string(base))

        # Keyword arguments (like metaclass)
        for keyword in class_node.keywords:
            if hasattr(ast, 'unparse'):
                value = ast.unparse(keyword.value)
            else:
                value = _ast_to_string(keyword.value)
            args.append(f"{keyword.arg}={value}")

        if args:
            signature += f"({', '.join(args)})"

    signature += ":"
    return signature
