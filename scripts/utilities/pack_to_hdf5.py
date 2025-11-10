#!/usr/bin/env python3
import os
import sys
import json
import math
import argparse
import numpy as np
import h5py
from typing import List, Tuple, Dict
from collections import defaultdict

# -------------------------------
# Helpers
# -------------------------------

def scan_split(root: str, split: str, resolve_symlinks: bool = True) -> Tuple[List[Tuple[str, int]], Dict[str, int]]:
    """
    Scan one dataset root for a split ('train' or 'val'), assuming structure:
      root/split/<class_name>/*.npy

    Returns:
      samples: list of (real_path, label_idx)
      class_to_idx: mapping of class_name -> index (sorted by name)
    """
    split_dir = os.path.join(os.path.expanduser(root), split)
    if not os.path.isdir(split_dir):
        raise FileNotFoundError(f"Missing split dir: {split_dir}")

    class_names = sorted([d.name for d in os.scandir(split_dir) if d.is_dir()])
    if not class_names:
        raise FileNotFoundError(f"No class subdirectories under {split_dir}")
    class_to_idx = {name: i for i, name in enumerate(class_names)}

    samples = []
    for cls in class_names:
        cls_dir = os.path.join(split_dir, cls)
        label = class_to_idx[cls]
        try:
            file_list = sorted(os.listdir(cls_dir))
        except PermissionError:
            # Skip unreadable dirs
            continue

        for fn in file_list:
            if not fn.lower().endswith(".npy"):
                continue
            fp = os.path.join(cls_dir, fn)
            rp = os.path.realpath(fp) if resolve_symlinks else fp
            if not os.path.exists(rp):
                raise FileNotFoundError(f"Broken symlink or missing file: {fp} -> {rp}")
            samples.append((rp, label))

    return samples, class_to_idx


def merge_class_maps(maps: List[Dict[str, int]]) -> Dict[str, int]:
    """
    Ensure all class maps are identical. If they are, return one of them.
    """
    if not maps:
        return {}
    base = maps[0]
    for m in maps[1:]:
        if m != base:
            raise ValueError(f"class_to_idx mismatch across roots. Example mismatch:\n{base}\nvs\n{m}")
    return base


def check_or_infer_shape(path: str) -> Tuple[int, int, int]:
    """
    Load one npy (mmap) and return (C,H,W). Validates channel-first with 6 channels.
    """
    arr = np.load(path, mmap_mode="r")
    if arr.ndim != 3:
        raise ValueError(f"{os.path.basename(path)} has {arr.ndim} dims; expected 3 (C,H,W)")
    if arr.shape[0] != 6:
        # Allow HWC if user accidentally saved like that
        if arr.shape[-1] == 6:
            raise ValueError(f"{path} is HWC with 6 channels; please transpose offline to CHW for packing.")
        raise ValueError(f"{path} first dim != 6; got {arr.shape}")
    C, H, W = arr.shape
    return C, H, W


def pad_to_shape(x: np.ndarray, target_hw: Tuple[int, int]) -> np.ndarray:
    """
    Zero-pad a CHW (6,H,W) array to (6, target_H, target_W).
    """
    C, H, W = x.shape
    tH, tW = target_hw
    if (H, W) == (tH, tW):
        return x
    if H > tH or W > tW:
        raise ValueError(f"Sample {(H,W)} larger than target {(tH,tW)}; cannot pad. Choose bigger --pad_to.")
    out = np.zeros((C, tH, tW), dtype=x.dtype)
    out[:, :H, :W] = x
    return out


def shard_ranges(n: int, shard_size: int) -> List[Tuple[int, int]]:
    if shard_size <= 0:
        return [(0, n)]
    shards = []
    start = 0
    while start < n:
        end = min(start + shard_size, n)
        shards.append((start, end))
        start = end
    return shards


def write_h5_shard(out_path: str,
                   paths: List[str],
                   labels: np.ndarray,
                   class_to_idx: Dict[str, int],
                   dtype: str,
                   pad_to: Tuple[int, int] = None,
                   compression: str = "lzf",
                   chunk_rows: int = 64):
    """
    Create a single HDF5 file shard writing images, labels, paths.
    """
    # Infer (C,H,W) from first sample
    C, H, W = check_or_infer_shape(paths[0])
    if pad_to:
        tH, tW = pad_to
        H, W = tH, tW

    n = len(paths)
    # Decide chunks
    chunk_rows = max(1, min(chunk_rows, n))
    chunks = (chunk_rows, C, H, W)

    # Use float32 (recommended) or float16
    if dtype not in ("float32", "float16"):
        raise ValueError("--dtype must be float32 or float16")
    np_dtype = np.float32 if dtype == "float32" else np.float16

    str_dt = h5py.string_dtype(encoding="utf-8")

    with h5py.File(out_path, "w") as h5:
        # datasets
        dset_x = h5.create_dataset(
            "images", shape=(n, C, H, W), dtype=np_dtype,
            chunks=chunks, compression=compression, shuffle=True
        )
        dset_y = h5.create_dataset("labels", shape=(n,), dtype=np.int32,
                                   chunks=(min(4096, n),), compression=compression, shuffle=True)
        dset_p = h5.create_dataset("paths", shape=(n,), dtype=str_dt)

        # attrs
        h5.attrs["class_to_idx"] = json.dumps(class_to_idx, ensure_ascii=False)
        h5.attrs["channels"] = C
        h5.attrs["height"] = H
        h5.attrs["width"] = W
        h5.attrs["dtype"] = dtype
        h5.attrs["compression"] = compression

        # write loop
        for i, p in enumerate(paths):
            arr = np.load(p, mmap_mode="r")
            # enforce CHW
            if arr.shape[0] != 6:
                raise ValueError(f"{p} not CHW-6; got {arr.shape}. Please fix offline.")
            if pad_to:
                arr = pad_to_shape(arr, (H, W))
            # cast
            if arr.dtype != np_dtype:
                arr = arr.astype(np_dtype, copy=False)
            dset_x[i, ...] = arr
            dset_y[i] = int(labels[i])
            dset_p[i] = p

            if (i + 1) % 1000 == 0 or (i + 1) == n:
                print(f"  wrote {i+1}/{n} samples -> {os.path.basename(out_path)}", flush=True)


def pack_split(out_dir: str,
               all_samples: List[Tuple[str, int]],
               class_to_idx: Dict[str, int],
               split: str,
               shard_size: int,
               dtype: str,
               pad_to: str,
               compression: str,
               chunk_rows: int):
    """
    Write many HDF5 shards for one split.
    """
    if not all_samples:
        print(f"[{split}] No samples; skipping.")
        return
    paths = [p for p, _ in all_samples]
    labels = np.array([y for _, y in all_samples], dtype=np.int32)

    # Optional padding target
    pad_hw = None
    if pad_to:
        try:
            tH, tW = map(int, pad_to.lower().split("x"))
            pad_hw = (tH, tW)
        except Exception:
            raise ValueError(f"--pad_to must look like '256x256' (got {pad_to})")

    # Make output dir/s
    os.makedirs(out_dir, exist_ok=True)

    # Shard and write
    total = len(paths)
    ranges = shard_ranges(total, shard_size)
    digits = int(math.ceil(math.log10(max(1, len(ranges)))))  # for zero-pad shard index

    for shard_idx, (s, e) in enumerate(ranges):
        shard_paths = paths[s:e]
        shard_labels = labels[s:e]
        shard_tag = f"{split}.shard_{str(shard_idx).zfill(digits)}.h5"
        out_path = os.path.join(out_dir, shard_tag)
        print(f"\n[{split}] Writing shard {shard_idx+1}/{len(ranges)} -> {out_path}")
        write_h5_shard(
            out_path, shard_paths, shard_labels, class_to_idx,
            dtype=dtype, pad_to=pad_hw, compression=compression, chunk_rows=chunk_rows
        )


# -------------------------------
# Main
# -------------------------------

def main():
    ap = argparse.ArgumentParser(description="Pack 6-channel .npy datasets (train/val) into HDF5 shards.")
    ap.add_argument("roots", nargs="+",
                    help="One or more dataset roots. Each must contain train/ and val/ with class folders.")
    ap.add_argument("-o", "--out_dir", required=True,
                    help="Output directory for HDF5 files (created if not exists).")
    ap.add_argument("--shard_size", type=int, default=20000,
                    help="Max samples per HDF5 file. Use a large value or 0 for single file per split.")
    ap.add_argument("--dtype", choices=["float32", "float16"], default="float32",
                    help="Data dtype stored in HDF5.")
    ap.add_argument("--pad_to", type=str, default=None,
                    help="Optional pad target for (H,W), e.g. '128x128'. If omitted, all samples must have same H,W.")
    ap.add_argument("--compression", choices=["lzf", "gzip", "None"], default="lzf",
                    help="HDF5 compression filter (lzf is fast; 'None' disables).")
    ap.add_argument("--chunk_rows", type=int, default=64,
                    help="Chunk rows for images dataset. Tune for I/O.")
    ap.add_argument("--no_resolve_symlinks", action="store_true",
                    help="Do not resolve symlinks (default is to resolve once).")
    args = ap.parse_args()

    if args.compression == "None":
        args.compression = None

    # Gather samples from all roots per split
    split_samples = {"train": [], "val": []}
    split_class_maps = {"train": [], "val": []}

    for root in args.roots:
        for split in ("train", "val"):
            samples, cls_map = scan_split(root, split, resolve_symlinks=not args.no_resolve_symlinks)
            split_samples[split].extend(samples)
            split_class_maps[split].append(cls_map)
            print(f"[{split}] {root}: +{len(samples)} samples, classes={list(cls_map.keys())}")

    # Validate consistent class maps (train/val may be identical; enforce each split)
    cls_map_train = merge_class_maps(split_class_maps["train"])
    cls_map_val = merge_class_maps(split_class_maps["val"])
    if cls_map_train != cls_map_val:
        raise ValueError(f"class_to_idx mismatch between train and val:\n{cls_map_train}\nvs\n{cls_map_val}")
    class_to_idx = cls_map_train

    # Sort samples (stable order)
    for split in ("train", "val"):
        split_samples[split].sort(key=lambda x: (x[1], x[0]))  # by label then path

    # If no pad_to, enforce uniform shape
    if args.pad_to is None:
        # Validate shapes are identical by checking a few randoms + first one of each label
        ref_c, ref_h, ref_w = check_or_infer_shape(split_samples["train"][0][0] if split_samples["train"] else split_samples["val"][0][0])
        for split in ("train", "val"):
            for i, (p, _) in enumerate(split_samples[split][:50]):  # sample a bit
                c, h, w = check_or_infer_shape(p)
                if (c, h, w) != (ref_c, ref_h, ref_w):
                    raise ValueError(f"Mixed shapes detected without --pad_to. {p} is {(c,h,w)} vs {(ref_c,ref_h,ref_w)}")

    # Write out
    out_dir = os.path.abspath(args.out_dir)
    pack_split(out_dir, split_samples["train"], class_to_idx, "train",
               shard_size=args.shard_size, dtype=args.dtype, pad_to=args.pad_to,
               compression=args.compression, chunk_rows=args.chunk_rows)
    pack_split(out_dir, split_samples["val"], class_to_idx, "val",
               shard_size=args.shard_size, dtype=args.dtype, pad_to=args.pad_to,
               compression=args.compression, chunk_rows=args.chunk_rows)

    print("\n✅ Done.")


if __name__ == "__main__":
    main()
