#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path
import numpy as np

def find_files(start_dir: Path, name: str = "*objectives_evaluations.npy"):
    # Recursively yield files that exactly match the target name
    for p in start_dir.rglob(name):
        if p.is_file():
            yield p

def load_shape(npy_path: Path):
    """
    Try to open the .npy without loading it fully into memory.
    Falls back to allow_pickle=True only if needed.
    Returns (shape, dtype, loaded_with_pickle) or raises an Exception.
    """
    try:
        arr = np.load(npy_path, mmap_mode="r", allow_pickle=False)
        # np.load may return an array-like (ndarray or memmap)
        shape = getattr(arr, "shape", None)
        dtype = getattr(arr, "dtype", None)
        if shape is None:
            raise TypeError("Loaded object has no 'shape' attribute.")
        return shape, dtype, False
    except ValueError as e:
        # Retry with pickle allowed if it's an object array/pickled content
        msg = str(e).lower()
        if "allow_pickle" in msg or "object arrays cannot be loaded" in msg:
            arr = np.load(npy_path, mmap_mode=None, allow_pickle=True)
            shape = getattr(arr, "shape", None)
            dtype = getattr(arr, "dtype", None)
            if shape is None:
                raise TypeError("Loaded (pickled) object has no 'shape' attribute.")
            return shape, dtype, True
        raise  # re-raise other ValueErrors

def main():
    parser = argparse.ArgumentParser(
        description="Traverse a directory, find all 'objectives_evaluations.npy' files, and print their shapes."
    )
    parser.add_argument("start_directory", type=Path, help="Path to start traversing from.")
    parser.add_argument(
        "--relative",
        action="store_true",
        help="Print paths relative to start_directory instead of absolute paths."
    )
    args = parser.parse_args()

    start_dir = args.start_directory.resolve()
    if not start_dir.exists():
        print(f"Error: start directory does not exist: {start_dir}", file=sys.stderr)
        sys.exit(1)
    if not start_dir.is_dir():
        print(f"Error: not a directory: {start_dir}", file=sys.stderr)
        sys.exit(1)

    found_any = False
    for path in find_files(start_dir):
        found_any = True
        display_path = path.relative_to(start_dir) if args.relative else path
        try:
            shape, dtype, used_pickle = load_shape(path)
            pickle_note = " (allow_pickle=True)" if used_pickle else ""
            print(f"{display_path}\tshape={tuple(shape)}, dtype={dtype}{pickle_note}")
        except Exception as e:
            print(f"{display_path}\tERROR: {e}", file=sys.stderr)

    if not found_any:
        print("No files named 'objectives_evaluations.npy' were found.", file=sys.stderr)

if __name__ == "__main__":
    main()
