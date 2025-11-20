#!/usr/bin/env python3
import os
import sys
import argparse
import numpy as np
import random
import datetime
from multiprocessing import Pool as ProcessPool
from multiprocessing.pool import ThreadPool


def log(msg):
    """Helper to print with timestamp"""
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {msg}", flush=True)


def list_samples(split_dir):
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
    Worker function.
    """
    shard_id, outdir, split, paths, labels, shape, dtype = shard_info
    n_samples = len(paths)
    X_path = os.path.join(outdir, f"X_{split}_{shard_id:03d}.npy")
    y_path = os.path.join(outdir, f"y_{split}_{shard_id:03d}.npy")

    try:
        # Create memmaps
        X_mmap = np.memmap(X_path, mode='w+', dtype=dtype, shape=(n_samples, *shape))
        y_mmap = np.memmap(y_path, mode='w+', dtype=np.int64, shape=(n_samples,))

        for i, p in enumerate(paths):
            arr = np.load(p)
            if arr.shape != shape:
                return (False, shard_id, f"Shape mismatch at {p}")
            X_mmap[i] = arr

        y_mmap[:] = labels

        X_mmap.flush();
        del X_mmap
        y_mmap.flush();
        del y_mmap
        return (True, shard_id, None)

    except Exception as e:
        return (False, shard_id, str(e))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root", help="Dataset root")
    ap.add_argument("-o", "--outdir", default=None)
    ap.add_argument("--split", choices=["train", "val"], default=None)
    ap.add_argument("--shard_size", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--worker_type", choices=["thread", "process"], default="thread",
                    help="Use 'thread' for shared memory (low RAM) or 'process' for max CPU speed.")

    args = ap.parse_args()

    np.random.seed(args.seed)
    random.seed(args.seed)

    root = os.path.abspath(args.root)
    outdir = os.path.abspath(args.outdir or root)
    os.makedirs(outdir, exist_ok=True)

    # Select the Pool Type
    if args.worker_type == "thread":
        PoolClass = ThreadPool
        log(f"Using MULTI-THREADING (Low RAM mode) with {args.threads} threads.")
    else:
        PoolClass = ProcessPool
        log(f"Using MULTI-PROCESSING (High Speed mode) with {args.threads} processes.")

    splits = [args.split] if args.split else ["train", "val"]

    for split in splits:
        split_dir = os.path.join(root, split)
        log(f"--- Processing Split: {split} ---")

        log(f"Scanning files in {split_dir}...")
        paths, labels = list_samples(split_dir)
        if len(paths) == 0:
            log(f"[skip] No .npy files found.")
            continue

        # Shuffle
        num_samples = len(paths)
        indices = np.arange(num_samples)
        np.random.shuffle(indices)
        paths = np.array(paths)[indices]
        labels = labels[indices]

        shape, dtype = infer_shape(paths[0])
        log(f"Found {num_samples} samples. Shape: {shape}")

        # Prepare Tasks
        tasks = []
        shard_id = 0
        for start in range(0, num_samples, args.shard_size):
            end = min(start + args.shard_size, num_samples)
            tasks.append((shard_id, outdir, split, paths[start:end], labels[start:end], shape, dtype))
            shard_id += 1

        total_shards = len(tasks)
        log(f"Launching {total_shards} tasks...")

        # Execute with the chosen Pool Class
        completed_count = 0
        errors = []
        with PoolClass(processes=args.threads) as pool:
            for success, sid, err_msg in pool.imap_unordered(write_single_shard, tasks):
                completed_count += 1
                if completed_count % 5 == 0 or completed_count == total_shards:
                    log(f"Progress: {completed_count}/{total_shards} shards written.")
                if not success:
                    log(f"ERROR: Shard {sid} failed: {err_msg}")
                    errors.append(sid)

        if not errors:
            log(f"SUCCESS: {split} complete.")
        else:
            log(f"WARNING: Failures in {split}: {errors}")

    log("Job Complete.")


if __name__ == "__main__":
    main()