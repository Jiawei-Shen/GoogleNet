#!/usr/bin/env python3
import os
import sys
import json
import math
import argparse
import numpy as np
import h5py
from typing import List, Tuple, Dict

# -------------------------------
# Helpers
# -------------------------------

def log(msg):
    print(f"[LOG] {msg}", flush=True)

def scan_split(root: str, split: str, resolve_symlinks: bool = True) -> Tuple[List[Tuple[str, int]], Dict[str, int]]:
    log(f"Scanning split '{split}' in {root}")
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
            log(f"  [WARN] Skipping unreadable directory: {cls_dir}")
            continue

        for fn in file_list:
            if not fn.lower().endswith(".npy"):
                continue
            fp = os.path.join(cls_dir, fn)
            rp = os.path.realpath(fp) if resolve_symlinks else fp
            if not os.path.exists(rp):
                raise FileNotFoundError(f"Broken symlink or missing file: {fp} -> {rp}")
            samples.append((rp, label))

        log(f"  Collected {len(file_list)} files for class '{cls}'")

    log(f"Completed scan for {split}: total {len(samples)} samples")
    return samples, class_to_idx


def merge_class_maps(maps: List[Dict[str, int]]) -> Dict[str, int]:
    if not maps:
        return {}
    base = maps[0]
    for m in maps[1:]:
        if m != base:
            raise ValueError(f"class_to_idx mismatch:\n{base}\nvs\n{m}")
    return base


def check_or_infer_shape(path: str) -> Tuple[int, int, int]:
    log(f"Checking shape for sample: {path}")
    arr = np.load(path, mmap_mode="r")
    if arr.ndim != 3:
        raise ValueError(f"{os.path.basename(path)} has {arr.ndim} dims; expected 3")
    if arr.shape[0] != 6:
        if arr.shape[-1] == 6:
            raise ValueError(f"{path} is HWC with 6 channels; please transpose to CHW")
        raise ValueError(f"{path} first dim != 6; got {arr.shape}")
    return arr.shape


def write_h5_shard(out_path: str, paths: List[str], labels: np.ndarray,
                   class_to_idx: Dict[str, int], dtype: str,
                   pad_to: Tuple[int, int] = None,
                   compression: str = "lzf", chunk_rows: int = 64):
    log(f"Opening new shard: {out_path}")
    C, H, W = check_or_infer_shape(paths[0])
    if pad_to:
        H, W = pad_to

    n = len(paths)
    np_dtype = np.float32 if dtype == "float32" else np.float16
    chunks = (max(1, min(chunk_rows, n)), C, H, W)
    str_dt = h5py.string_dtype(encoding="utf-8")

    with h5py.File(out_path, "w") as h5:
        log(f"  Creating datasets with shape {(n, C, H, W)} and compression={compression}")
        dset_x = h5.create_dataset("images", shape=(n, C, H, W), dtype=np_dtype,
                                   chunks=chunks, compression=compression, shuffle=True)
        dset_y = h5.create_dataset("labels", shape=(n,), dtype=np.int32,
                                   chunks=(min(4096, n),), compression=compression, shuffle=True)
        dset_p = h5.create_dataset("paths", shape=(n,), dtype=str_dt)

        h5.attrs["class_to_idx"] = json.dumps(class_to_idx, ensure_ascii=False)
        h5.attrs.update(dict(channels=C, height=H, width=W,
                             dtype=dtype, compression=compression))

        for i, p in enumerate(paths):
            arr = np.load(p, mmap_mode="r")
            if arr.shape[0] != 6:
                raise ValueError(f"{p} not CHW-6; got {arr.shape}")
            if pad_to:
                tH, tW = pad_to
                out = np.zeros((C, tH, tW), dtype=arr.dtype)
                out[:, :arr.shape[1], :arr.shape[2]] = arr
                arr = out
            dset_x[i, ...] = arr.astype(np_dtype, copy=False)
            dset_y[i] = int(labels[i])
            dset_p[i] = p

            if (i + 1) % 1000 == 0 or (i + 1) == n:
                log(f"    Wrote {i+1}/{n} samples into {os.path.basename(out_path)}")

    log(f"Finished writing shard: {out_path}")


def pack_split(out_dir, all_samples, class_to_idx, split,
               shard_size, dtype, pad_to, compression, chunk_rows):
    if not all_samples:
        log(f"[{split}] No samples; skipping.")
        return
    log(f"[{split}] Preparing to pack {len(all_samples)} samples")
    paths = [p for p, _ in all_samples]
    labels = np.array([y for _, y in all_samples], dtype=np.int32)
    pad_hw = None
    if pad_to:
        tH, tW = map(int, pad_to.lower().split("x"))
        pad_hw = (tH, tW)

    os.makedirs(out_dir, exist_ok=True)
    ranges = [(i, min(i + shard_size, len(paths))) for i in range(0, len(paths), shard_size or len(paths))]
    log(f"[{split}] Total {len(ranges)} shards to write")

    for shard_idx, (s, e) in enumerate(ranges):
        log(f"[{split}] Starting shard {shard_idx+1}/{len(ranges)} [{s}:{e}]")
        shard_paths = paths[s:e]
        shard_labels = labels[s:e]
        shard_tag = f"{split}.shard_{str(shard_idx).zfill(3)}.h5"
        out_path = os.path.join(out_dir, shard_tag)
        write_h5_shard(out_path, shard_paths, shard_labels, class_to_idx,
                       dtype, pad_to=pad_hw, compression=compression, chunk_rows=chunk_rows)
    log(f"[{split}] Completed all shards ✅")


# -------------------------------
# Main
# -------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("roots", nargs="+")
    ap.add_argument("-o", "--out_dir", required=True)
    ap.add_argument("--shard_size", type=int, default=20000)
    ap.add_argument("--dtype", choices=["float32", "float16"], default="float32")
    ap.add_argument("--pad_to", type=str, default=None)
    ap.add_argument("--compression", choices=["lzf", "gzip", "None"], default="lzf")
    ap.add_argument("--chunk_rows", type=int, default=64)
    ap.add_argument("--no_resolve_symlinks", action="store_true")
    args = ap.parse_args()

    log("==== HDF5 PACK START ====")
    log(f"Roots: {args.roots}")
    log(f"Output dir: {args.out_dir}")
    log(f"Shard size: {args.shard_size}, dtype: {args.dtype}, compression: {args.compression}")

    split_samples = {"train": [], "val": []}
    split_class_maps = {"train": [], "val": []}

    for root in args.roots:
        for split in ("train", "val"):
            samples, cls_map = scan_split(root, split, resolve_symlinks=not args.no_resolve_symlinks)
            split_samples[split].extend(samples)
            split_class_maps[split].append(cls_map)
            log(f"[{split}] {root}: +{len(samples)} samples")

    class_to_idx = merge_class_maps(split_class_maps["train"])
    log(f"Class map: {class_to_idx}")

    for split in ("train", "val"):
        split_samples[split].sort(key=lambda x: (x[1], x[0]))
        log(f"[{split}] Sorted {len(split_samples[split])} samples")

    if args.pad_to is None and split_samples["train"]:
        ref_c, ref_h, ref_w = check_or_infer_shape(split_samples["train"][0][0])
        log(f"Reference shape: {(ref_c, ref_h, ref_w)}")

    out_dir = os.path.abspath(args.out_dir)
    log(f"Writing output to {out_dir}")

    pack_split(out_dir, split_samples["train"], class_to_idx, "train",
               shard_size=args.shard_size, dtype=args.dtype,
               pad_to=args.pad_to, compression=args.compression, chunk_rows=args.chunk_rows)
    pack_split(out_dir, split_samples["val"], class_to_idx, "val",
               shard_size=args.shard_size, dtype=args.dtype,
               pad_to=args.pad_to, compression=args.compression, chunk_rows=args.chunk_rows)

    log("==== HDF5 PACK COMPLETE ✅ ====")


if __name__ == "__main__":
    main()
