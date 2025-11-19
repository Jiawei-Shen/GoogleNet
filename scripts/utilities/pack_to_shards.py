#!/usr/bin/env python3
import os
import sys
import argparse
import numpy as np
import random
from multiprocessing import Pool, cpu_count
from functools import partial


def list_samples(split_dir):
    """Scans for .npy files and assigns labels (true=1, false=0)."""
    classes = []
    paths = []

    if not os.path.exists(split_dir):
        return [], np.array([])

    for cls in os.listdir(split_dir):
        p = os.path.join(split_dir, cls)
        if not os.path.isdir(p): continue

        lab = 1 if cls.lower() == "true" else 0

        for root, _, files in os.walk(p):
            for f in files:
                if f.endswith(".npy"):
                    paths.append(os.path.join(root, f))
                    classes.append(lab)

    return paths, np.asarray(classes, dtype=np.int64)


def infer_shape(first_path):
    a = np.load(first_path, mmap_mode='r')
    return a.shape, a.dtype


def write_single_shard(shard_info):
    """
    Worker function to write one shard.
    shard_info contains: (shard_id, outdir, split, file_paths, labels, shape, dtype)
    """
    shard_id, outdir, split, paths, labels, shape, dtype = shard_info
    n_samples = len(paths)

    X_path = os.path.join(outdir, f"X_{split}_{shard_id:03d}.npy")
    y_path = os.path.join(outdir, f"y_{split}_{shard_id:03d}.npy")

    # Create memmaps
    try:
        X_mmap = np.memmap(X_path, mode='w+', dtype=dtype, shape=(n_samples, *shape))
        y_mmap = np.memmap(y_path, mode='w+', dtype=np.int64, shape=(n_samples,))

        for i, p in enumerate(paths):
            arr = np.load(p)
            if arr.shape != shape:
                print(f"Error: Shape mismatch at {p}")
                return False
            X_mmap[i] = arr

        y_mmap[:] = labels

        # Flush and close
        X_mmap.flush()
        y_mmap.flush()
        del X_mmap
        del y_mmap
        return True

    except Exception as e:
        print(f"Failed writing shard {shard_id}: {e}")
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root", help="Dataset root")
    ap.add_argument("-o", "--outdir", default=None)
    ap.add_argument("--split", choices=["train", "val"], default=None)
    ap.add_argument("--shard_size", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--threads", type=int, default=4, help="Number of parallel workers")

    args = ap.parse_args()

    np.random.seed(args.seed)
    random.seed(args.seed)

    root = os.path.abspath(args.root)
    outdir = os.path.abspath(args.outdir or root)
    os.makedirs(outdir, exist_ok=True)

    splits = [args.split] if args.split else ["train", "val"]

    for split in splits:
        split_dir = os.path.join(root, split)
        print(f"--- Processing: {split} ---")

        # 1. Gather and Shuffle (Single Threaded for safety)
        print("Scanning files...")
        paths, labels = list_samples(split_dir)
        if len(paths) == 0:
            print(f"[skip] No .npy files in {split_dir}")
            continue

        print("Shuffling...")
        num_samples = len(paths)
        indices = np.arange(num_samples)
        np.random.shuffle(indices)
        paths = np.array(paths)[indices]
        labels = labels[indices]

        shape, dtype = infer_shape(paths[0])
        print(f"[{split}] N={num_samples}, Shape={shape}, Dtype={dtype}")

        # 2. Prepare Tasks
        tasks = []
        shard_id = 0
        for start in range(0, num_samples, args.shard_size):
            end = min(start + args.shard_size, num_samples)

            # Slice the data for this specific shard
            shard_paths = paths[start:end]
            shard_labels = labels[start:end]

            tasks.append((shard_id, outdir, split, shard_paths, shard_labels, shape, dtype))
            shard_id += 1

        # 3. Execute Parallel Writing
        print(f"Starting {args.threads} workers for {len(tasks)} shards...")

        # Use 'spawn' or 'fork' context if needed, but standard Pool usually works on Linux
        with Pool(processes=args.threads) as pool:
            results = pool.map(write_single_shard, tasks)

        if all(results):
            print(f"Successfully wrote all {len(results)} shards for {split}.")
        else:
            print(f"Warning: Some shards failed for {split}.")


if __name__ == "__main__":
    main()