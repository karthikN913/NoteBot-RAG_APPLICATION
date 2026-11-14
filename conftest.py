import sys, pathlib
# Ensure repository root is on PYTHONPATH for test imports
repo_root = pathlib.Path(__file__).parent.resolve()
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

# Pre-initialize faiss C++ runtime cleanly on Windows before any I/O redirection
try:
    import faiss
except Exception:
    pass


