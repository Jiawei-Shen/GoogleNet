#!/usr/bin/env python3
import os
import glob
import time
import numpy as np
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

# ================== CONFIG ==================
SRC_ROOT = "/scratch/jshen/data/Pansoma/COLO829T_Pacbio/6ch_training_data_P25_SNV/ALL_chr_merged_REAL/train"
DST_ROOT = "/scratch/jshen/data/Pansoma/COLO829T_Pacbio/6ch_training_data_P25_SNV/ALL_chr_merged_REAL_sharded/train"
SHARD_SIZE = 4096
N_THREADS = 16          # 🔥 adjust: 4–16 depending on disk + CPU
# ============================================

os.makedirs(DST_ROOT, exist_ok=True)

def log(msg):
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)


def load_one(sample):
    """Helper for threaded loading: sample = (path, label)."""
    path, label = sample
    try:
        x = np.load(path)  # (6, 201, 100)
    except Exception as e:
        # bubble up which file failed
        raise RuntimeError(f"Error loading {path}: {e}")
    return x, label


def main():
    log("Starting sharding script.")
    log(f"SRC_ROOT = {SRC_ROOT}")
    log(f"DST_ROOT = {DST_ROOT}")
    log(f"SHARD_SIZE = {SHARD_SIZE}, N_THREADS = {N_THREADS}")

    # -------------------------------
    # Step 1: Scan dataset
    # -------------------------------
    log("Scanning dataset directories...")

    all_samples = []  # list of (path, label)
    for cls in ["false", "true"]:
        cls_dir = os.path.join(SRC_ROOT, cls)
        files = sorted(glob.glob(os.path.join(cls_dir, "*.npy")))
        label = 1 if cls == "true" else 0
        for p in files:
            all_samples.append((p, label))
        log(f"  Found {len(files):,} '{cls}' samples in {cls_dir}")

    total = len(all_samples)
    log(f"Total samples found: {total:,}")
    if total == 0:
        log("No samples found. Exiting.")
        return

    log("Beginning sharding...")

    # -------------------------------
    # Step 2: Create shards
    # -------------------------------
    shard_idx = 0

    for i in range(0, total, SHARD_SIZE):
        chunk = all_samples[i:i + SHARD_SIZE]
        log(f"\n--- Creating shard {shard_idx} "
            f"({i:,} → {i + len(chunk):,}) containing {len(chunk)} samples ---")

        xs = []
        ys = []

        # Threaded loading of this shard
        with ThreadPoolExecutor(max_workers=N_THREADS) as executor:
            futures = [executor.submit(load_one, sample) for sample in chunk]

            for f in tqdm(as_completed(futures),
                          total=len(futures),
                          desc=f"Loading shard {shard_idx}",
                          leave=False):
                try:
                    x, label = f.result()
                except Exception as e:
                    log(str(e))
                    raise
                xs.append(x)
                ys.append(label)

        # Stack
        log("Stacking arrays...")
        xs = np.stack(xs, axis=0)          # (N, 6, 201, 100)
        ys = np.array(ys, dtype=np.int64)  # (N,)

        shard_path = os.path.join(DST_ROOT, f"shard_{shard_idx:05d}.npz")
        log(f"Saving shard to {shard_path} ...")

        np.savez_compressed(
            shard_path,
            data=xs,
            labels=ys
        )

        log(f"Saved shard {shard_idx}: data shape={xs.shape}, labels shape={ys.shape}")
        shard_idx += 1

    log("\nSharding completed successfully!")
    log(f"Total shards created: {shard_idx}")


if __name__ == "__main__":
    main()
