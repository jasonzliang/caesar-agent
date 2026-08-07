#!/usr/bin/env python
"""Universal Streamlit launcher with automatic dependency detection and incremental installation."""
import sys, os, subprocess, ast
from pathlib import Path

# Mapping of import names to pip package names (ONLY where they differ)
IMPORT_TO_PACKAGE = {
    'sklearn': 'scikit-learn', 'cv2': 'opencv-python', 'PIL': 'Pillow',
    'skimage': 'scikit-image', 'yaml': 'pyyaml', 'dateutil': 'python-dateutil',
    'dotenv': 'python-dotenv', 'tf': 'tensorflow',
    'sentence_transformers': 'sentence-transformers', 'bs4': 'beautifulsoup4',
    'MySQLdb': 'mysqlclient', 'psycopg2': 'psycopg2-binary',
    'memcache': 'python-memcached', 'google.cloud': 'google-cloud-storage',
    'Crypto': 'pycryptodome', 'Cryptodome': 'pycryptodomex',
    'magic': 'python-magic', 'slugify': 'python-slugify',
    'Levenshtein': 'python-Levenshtein', 'cStringIO': 'StringIO',
    'typing_extensions': 'typing-extensions', 'sqlite3': 'pysqlite3',
}

ALWAYS_INSTALL = ['streamlit', 'watchdog']

STDLIB = {
    'sys', 'os', 'subprocess', 'pathlib', 'json', 're', 'collections',
    'multiprocessing', 'warnings', 'ast', 'time', 'datetime', 'math',
    'random', 'itertools', 'functools', 'typing', 'copy', 'pickle',
    'io', 'logging', 'argparse', 'shutil', 'glob', 'tempfile', 'uuid',
    'hashlib', 'base64', 'struct', 'enum', 'dataclasses', 'abc', 'contextlib',
    'traceback', 'inspect', 'textwrap', 'string', 'weakref', 'threading',
    'queue', 'socket', 'urllib', 'http', 'ssl', 'email', 'zipfile', 'tarfile',
}

def extract_imports(script_path):
    """Extract imports and pip install comments from script."""
    imports, pip_packages = set(), set()
    try:
        with open(script_path, 'r', encoding='utf-8') as f:
            content = f.read()
        # Parse pip comments
        for line in content.split('\n'):
            if line.strip().startswith('#') and 'pip install' in line:
                packages = line.split('pip install', 1)[1].strip().split()
                pip_packages.update(p for p in packages if not p.startswith('-'))
        # Parse AST for imports
        tree = ast.parse(content)
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imports.update(alias.name.split('.')[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imports.add(node.module.split('.')[0])
    except Exception as e:
        print(f"Warning: Could not parse {script_path}: {e}")
    return imports, pip_packages

def get_package_names(imports):
    """Convert import names to pip package names."""
    return sorted({IMPORT_TO_PACKAGE.get(imp, imp) for imp in imports if imp not in STDLIB})

def get_installed_packages(venv_python):
    """Get currently installed packages in venv."""
    try:
        result = subprocess.run([str(venv_python), "-m", "pip", "list", "--format=freeze"],
                              capture_output=True, text=True, check=True)
        return {line.split('==')[0].lower().replace('_', '-')
                for line in result.stdout.strip().split('\n') if '==' in line}
    except Exception as e:
        print(f"Warning: Could not get installed packages: {e}")
        return set()

def setup_venv(script_path, venv_dir):
    """Create venv and install missing packages."""
    if not venv_dir.exists():
        print(f"Creating virtualenv at {venv_dir}...")
        subprocess.run([sys.executable, "-m", "venv", str(venv_dir)], check=True)

    venv_python = venv_dir / ("Scripts" if os.name == "nt" else "bin") / "python"

    print(f"Analyzing {script_path.name} for dependencies...")
    imports, pip_packages = extract_imports(script_path)
    all_packages = sorted(set(get_package_names(imports)) | pip_packages | set(ALWAYS_INSTALL))

    if not all_packages:
        print("No external packages detected.")
        return venv_python

    if pip_packages:
        print(f"  From pip comments: {', '.join(sorted(pip_packages))}")
    if imports - STDLIB:
        print(f"  From imports: {', '.join(sorted(get_package_names(imports)))}")

    print("Checking installed packages...")
    installed = get_installed_packages(venv_python)
    needed = [pkg for pkg in all_packages
              if pkg.lower().replace('_', '-') not in installed]

    if needed:
        print(f"Installing {len(needed)} new package(s): {', '.join(needed)}")
        subprocess.run([str(venv_python), "-m", "pip", "install", "--upgrade", "pip"],
                      check=True, capture_output=True)

        failed = []
        for pkg in needed:
            result = subprocess.run([str(venv_python), "-m", "pip", "install",
                                   "--only-binary", ":all:", pkg],
                                  capture_output=True, text=True)
            if result.returncode != 0:
                # Retry without --only-binary constraint
                result = subprocess.run([str(venv_python), "-m", "pip", "install", pkg],
                                      capture_output=True, text=True)
            if result.returncode != 0:
                print(f"  ✗ Failed to install {pkg}")
                failed.append(pkg)
            else:
                print(f"  ✓ Installed {pkg}")

        if failed:
            print(f"\n⚠ {len(failed)} package(s) failed to install: {', '.join(failed)}")
        print(f"✓ Successfully installed {len(needed) - len(failed)}/{len(needed)} package(s)")
    else:
        print("✓ All required packages already installed")

    return venv_python

def main():
    """Main entry point."""
    if len(sys.argv) < 2:
        print("Usage: streamlit_run.py <script.py> [streamlit args...]")
        print("\nFeatures:")
        print("  - Auto-detects dependencies from imports")
        print("  - Reads '# pip install ...' comments")
        print("  - Creates isolated venv per app")
        print("  - Only installs missing packages")
        sys.exit(1)

    script_path = Path(sys.argv[1]).resolve()
    if not script_path.exists():
        print(f"Error: Script not found: {script_path}")
        sys.exit(1)

    venv_root = script_path.parent / ".venv"
    venv_root.mkdir(exist_ok=True)
    venv_dir = venv_root / script_path.stem

    print(f"App: {script_path.name}")
    print(f"Virtual environment: .venv/{script_path.stem}\n")

    if sys.prefix == sys.base_prefix:
        setup_venv(script_path, venv_dir)
        print("\nLaunching Streamlit app...")
        print("-" * 50)
        streamlit_path = venv_dir / ("Scripts" if os.name == "nt" else "bin") / "streamlit"
        os.execv(str(streamlit_path),
                [str(streamlit_path), "run", str(script_path)] + sys.argv[2:])

if __name__ == "__main__":
    main()
