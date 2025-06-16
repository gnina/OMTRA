import argparse
from pathlib import Path
import sys

def remove_broken_symlinks(directory: Path) -> None:
    """
    Recursively scan `directory` for symlinks and delete those whose targets do not exist.
    """
    for path in directory.rglob('*'):
        if path.is_symlink():
            try:
                # resolve(strict=True) will raise if the final target is missing
                _ = path.resolve(strict=True)
            except FileNotFoundError:
                print(f"Removing broken symlink: {path}")
                path.unlink()

def main():
    parser = argparse.ArgumentParser(
        description="Remove broken symbolic links in a directory tree"
    )
    parser.add_argument(
        "directory",
        type=Path,
        help="Path to the directory to scan for broken symlinks"
    )
    args = parser.parse_args()

    if not args.directory.is_dir():
        print(f"Error: {args.directory!r} is not a valid directory.", file=sys.stderr)
        sys.exit(1)

    remove_broken_symlinks(args.directory)

if __name__ == "__main__":
    main()