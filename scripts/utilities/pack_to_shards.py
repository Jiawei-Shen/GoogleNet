#!/usr/bin/env python3
import os, sys, argparse, numpy as np

def list_samples(split_dir):
    # expects split_dir/{true,false}/*.npy (adjust if needed)
    classes = []
    paths = []
    for cls in sorted(os.listdir(split_dir)):
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

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root", help="dataset root containing train/ and val/")
    ap.add_argument("-o","--outdir", default=None)
    ap.add_argument("--split", choices=["train","val"], default=None,
                    help="If set, pack only this split; else both.")
    ap.add_argument("--shard_size", type=int, default=2_000_000,
                    help="Max samples per shard file")
    args = ap.parse_args()

    root = os.path.abspath(args.root)
    outdir = os.path.abspath(args.outdir or root)
    os.makedirs(outdir, exist_ok=True)

    splits = [args.split] if args.split else ["train","val"]
    for split in splits:
        split_dir = os.path.join(root, split)
        if not os.path.isdir(split_dir):
            print(f"[skip] {split_dir} not found")
            continue

        paths, labels = list_samples(split_dir)
        if len(paths) == 0:
            print(f"[skip] no samples in {split_dir}")
            continue

        paths = sorted(paths)  # deterministic order
        (C,H,W), dtype = infer_shape(paths[0])
        N = len(paths)
        print(f"[{split}] N={N} shape=(6,{H},{W}) dtype={dtype}")

        # Shard writing
        start = 0
        shard_id = 0
        while start < N:
            end = min(start + args.shard_size, N)
            n_this = end - start

            X_path = os.path.join(outdir, f"X_{split}_{shard_id:03d}.npy")
            y_path = os.path.join(outdir, f"y_{split}_{shard_id:03d}.npy")

            X = np.memmap(X_path, mode='w+', dtype=dtype, shape=(n_this, C, H, W))
            y = np.memmap(y_path, mode='w+', dtype=np.int64, shape=(n_this,))

            for i, p in enumerate(paths[start:end]):
                arr = np.load(p, mmap_mode='r')
                # ensure shape and dtype are consistent
                if arr.shape != (C,H,W):
                    raise ValueError(f"Shape mismatch at {p}: got {arr.shape}, expect {(C,H,W)}")
                X[i] = arr
            y[:] = labels[start:end]

            # flush to disk
            del X; del y
            print(f"  wrote {X_path} | {y_path} [{n_this} samples]")
            start = end
            shard_id += 1

if __name__ == "__main__":
    main()
