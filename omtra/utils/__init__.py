from pathlib import Path

def omtra_root() -> str:
    """Returns the root directory of the omtra package."""
    root_path = Path(__file__).parent.parent.parent
    return str(root_path.resolve())