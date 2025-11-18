import os
import glob
import numpy as np
from tqdm import tqdm

SRC_ROOT = "/scratch/jshen/data/Pansoma/COLO829T_Pacbio/6ch_training_data_P25_SNV/ALL_chr_merged_REAL/train"
DST_ROOT = "/scratch/jshen/data/Pansoma/COLO829T_Pacbio/6ch_training_data_P25_SNV/ALL_chr_merged_REAL_sharded/train"
os.makedirs(DST_ROOT, exist_ok=True)

# You may already have a manifest with paths+labels; otherwise scan dirs
all_samples = []  # list of (path, label)
for cls in ["false", "true"]:  # or whatever your classes are
    cls_dir = os.path.join(SRC_ROOT, cls)
    for p in glob.glob(os.path.join(cls_dir, "*.npy")):
        label = 1 if cls == "true" else 0
        all_samples.append((p, label))

SHARD_SIZE = 4096
shard_idx = 0

for i in tqdm(range(0, len(all_samples), SHARD_SIZE)):
    chunk = all_samples[i:i+SHARD_SIZE]
    xs = []
    ys = []
    for path, label in chunk:
        x = np.load(path)  # (6, 201, 100)
        xs.append(x)
        ys.append(label)
    xs = np.stack(xs, axis=0)          # (N, 6, 201, 100)
    ys = np.array(ys, dtype=np.int64)  # (N,)
    np.savez_compressed(
        os.path.join(DST_ROOT, f"shard_{shard_idx:05d}.npz"),
        data=xs,
        labels=ys
    )
    shard_idx += 1
