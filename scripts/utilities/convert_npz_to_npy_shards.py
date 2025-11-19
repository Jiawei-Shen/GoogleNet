#!/usr/bin/env python3
import argparse
import os
from glob import glob

import numpy as np


def detect_xy_keys(npz):
    """
    Detect feature/label keys in an npz file.

    Supported:
      - ('data', 'labels')
      - ('x', 'y')
      - ('arr_0', 'arr_1')  # fallback if exactly 2 arrays
    """
    keys = list(npz.files)
    key_set = set(keys)

    if "data" in key_set and "labels" in key_set:
        return "data", "labels"
    if "x" in key_set and "y" in key_set:
        return "x", "y"
    if set(keys) == {"arr_0", "arr_1"} and len(keys) == 2:
        return "arr_0", "arr_1"

    raise RuntimeError(
        f"Unsupported key set {keys}; expected ('data','labels'), ('x','y') or ('arr_0','arr_1')."
    )


def convert_one_npz(npz_path, overwrite=False, remove_npz=False, dry_run=False):
    """
    Convert a single shard_XXXXX.npz into shard_XXXXX_x.npy and shard_XXXXX_y.npy.
    """
    dir_name = os.path.dirname(npz_path)
    base = os.path.splitext(os.path.basename(npz_path))[0]  # e.g. "shard_00000"

    out_x = os.path.join(dir_name, base + "_x.npy")
    out_y = os.path.join(dir_name, base + "_y.npy")

    if (os.path.exists(out_x) and os.path.exists(out_y)) and not overwrite:
        print(f"[SKIP] {npz_path} → {out_x}, {out_y} already exist (use --overwrite to force).")
        return

    print(f"[INFO] Converting {npz_path} → {out_x}, {out_y}")

    if dry_run:
        return

    with np.load(npz_path) as data:
        x_key, y_key = detect_xy_keys(data)
        x_arr = data[x_key]
        y_arr = data[y_key]

    if x_arr.ndim != 4 or x_arr.shape[1] != 6:
        raise ValueError(
            f"{npz_path}: expected x shape (N, 6, H, W), got {x_arr.shape}"
        )
    if y_arr.shape[0] != x_arr.shape[0]:
        raise ValueError(
            f"{npz_path}: len(x)={x_arr.shape[0]} != len(y)={y_arr.shape[0]}"
        )

    # Save as raw .npy (uncompressed) to allow fast memmap later
    np.save(out_x, x_arr)  # keep original dtype (int8 in your case)
    np.save(out_y, y_arr)

    if remove_npz:
        try:
            os.remove(npz_path)
            print(f"[INFO] Removed original NPZ: {npz_path}")
        except OSError as e:
            print(f"[WARN] Failed to remove {npz_path}: {e}")


def convert_split(root, split, pattern, overwrite=False, remove_npz=False, dry_run=False):
    """
    Convert all NPZ shards in one split directory, e.g. root/train or root/val.
    """
    split_dir = os.path.join(root, split)
    if not os.path.isdir(split_dir):
        print(f"[WARN] Split dir not found, skipping: {split_dir}")
        return

    npz_paths = sorted(glob(os.path.join(split_dir, pattern)))
    if not npz_paths:
        print(f"[WARN] No NPZ shards matching '{pattern}' in {split_dir}")
        return

    print(f"[INFO] Converting split '{split}' in {split_dir}: {len(npz_paths)} NPZ files found.")

    for i, p in enumerate(npz_paths, 1):
        print(f"[{split}] [{i}/{len(npz_paths)}] {os.path.basename(p)}")
        convert_one_npz(p, overwrite=overwrite, remove_npz=remove_npz, dry_run=dry_run)

    print(f"[DONE] Split '{split}' in {split_dir}")


def parse_args():
    ap = argparse.ArgumentParser(
        description="Convert NPZ shards (data+labels) into NPY shard_x.npy / shard_y.npy pairs."
    )
    ap.add_argument(
        "root",
        type=str,
        help="Root directory containing split subfolders like 'train/', 'val/' (e.g. .../ALL_chr_merged_REAL_sharded).",
    )
    ap.add_argument(
        "--splits",
        type=str,
        nargs="+",
        default=["train", "val"],
        help="Which subdirectories under root to process (default: train val).",
    )
    ap.add_argument(
        "--pattern",
        type=str,
        default="shard_*.npz",
        help="Glob pattern for shard NPZ files within each split (default: shard_*.npz).",
    )
    ap.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing *_x.npy and *_y.npy files.",
    )
    ap.add_argument(
        "--remove-npz",
        action="store_true",
        help="Delete NPZ file after successful conversion.",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be done, but do not write any files.",
    )
    return ap.parse_args()


def main():
    args = parse_args()
    root = os.path.abspath(os.path.expanduser(args.root))

    print(f"[INFO] Root: {root}")
    print(f"[INFO] Splits: {args.splits}")
    print(f"[INFO] Pattern: {args.pattern}")
    print(f"[INFO] Overwrite: {args.overwrite}")
    print(f"[INFO] Remove NPZ: {args.remove_npz}")
    print(f"[INFO] Dry-run: {args.dry_run}")

    for split in args.splits:
        convert_split(
            root=root,
            split=split,
            pattern=args.pattern,
            overwrite=args.overwrite,
            remove_npz=args.remove_npz,
            dry_run=args.dry_run,
        )


if __name__ == "__main__":
    main()
