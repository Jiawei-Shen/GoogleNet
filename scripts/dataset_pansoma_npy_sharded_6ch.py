import os
import glob
from typing import List, Tuple, Dict, Union
from collections import OrderedDict

import numpy as np
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader, ConcatDataset

# ─────────────────────────────────────────────────────────────
#  Sharded NPY dataset with LRU memmap cache (fixes open-FD blowup)
# ─────────────────────────────────────────────────────────────


class NPYShardDataset(Dataset):
    """
    Sharded NPY dataset:

        root_dir/
            X_train_000_x.npy   (N0, 6, H, W)
            X_train_000_y.npy   (N0,)
            X_train_001_x.npy   (N1, 6, H, W)
            X_train_001_y.npy   (N1,)
            ...

    or generally: anything matching "*_x.npy" with a paired "*_y.npy".

    Key ideas (like your fast NPZ LRU):

      * Do NOT use memmap for training.
      * Load entire shard (x,y) into RAM when first needed.
      * Keep only a small number of shards cached (LRU) per worker.
      * Per-sample access is then just RAM indexing.
    """

    def __init__(
        self,
        root_dir: str,
        transform=None,
        return_paths: bool = False,
        max_cached_shards: int = 1,  # 1 shard ~= 470MB, so per-worker RAM tradeoff
    ):
        self.root_dir = os.path.abspath(os.path.expanduser(root_dir))
        self.transform = transform
        self.return_paths = return_paths
        self.max_cached_shards = max_cached_shards

        if not os.path.isdir(self.root_dir):
            raise FileNotFoundError(f"Root directory not found: {self.root_dir}")

        # Find shard pairs: *_x.npy with corresponding *_y.npy
        all_files = sorted(os.listdir(self.root_dir))
        x_paths: List[str] = []

        for f in all_files:
            if not f.endswith("_x.npy"):
                continue
            x_path = os.path.join(self.root_dir, f)
            y_path = x_path.replace("_x.npy", "_y.npy")
            if not os.path.isfile(y_path):
                print(f"[NPYShardDataset] WARNING: found {x_path} but missing {y_path}, skipping.")
                continue
            x_paths.append(x_path)

        if not x_paths:
            raise ValueError(f"No *_x.npy shard files found in {self.root_dir}")

        self.x_paths: List[str] = []
        self.y_paths: List[str] = []
        self.shard_sizes: List[int] = []
        self.cumulative_sizes: List[int] = []

        total = 0
        C = H = W = None
        dtype = None

        # Only open each shard once here just to validate shapes
        for x_path in x_paths:
            y_path = x_path.replace("_x.npy", "_y.npy")

            try:
                # --- load feature shard ---
                x_arr = np.load(x_path, allow_pickle=False)
                if x_arr.ndim != 4 or x_arr.shape[1] != 6:
                    raise ValueError(
                        f"{x_path}: expected shape (N, 6, H, W), got {x_arr.shape}"
                    )
                n, c, h, w = x_arr.shape
                if C is None:
                    C, H, W = c, h, w
                    dtype = x_arr.dtype
                else:
                    if (c, h, w) != (C, H, W):
                        raise ValueError(
                            f"{x_path}: inconsistent shape {x_arr.shape}, "
                            f"expected (*, {C}, {H}, {W})"
                        )

                # --- load label shard ---
                y_arr = np.load(y_path, allow_pickle=False)
                if y_arr.ndim != 1:
                    raise ValueError(f"{y_path}: expected 1D labels, got {y_arr.shape}")
                if y_arr.shape[0] != n:
                    raise ValueError(
                        f"{x_path}/{y_path}: feature/label length mismatch: {n} vs {y_arr.shape[0]}"
                    )

            except Exception as e:
                print(f"[NPYShardDataset] WARNING: skipping bad shard pair {x_path}, {y_path}: {e}")
                continue

            # Valid shard
            self.x_paths.append(x_path)
            self.y_paths.append(y_path)
            self.shard_sizes.append(n)
            total += n
            self.cumulative_sizes.append(total)

        if total == 0 or not self.x_paths:
            raise ValueError(
                f"NPYShardDataset from {self.root_dir} has zero usable samples "
                f"(all shards invalid or skipped)."
            )

        self.num_channels = C
        self.height = H
        self.width = W
        self.dtype = dtype

        # LRU cache: shard_idx -> (x_arr, y_arr)
        self._shard_cache: "OrderedDict[int, Tuple[np.ndarray, np.ndarray]]" = OrderedDict()

        print(
            f"Initialized NPYShardDataset from {self.root_dir}: "
            f"{len(self.x_paths)} shards, {total} samples total. "
            f"Shape=(6,{H},{W}) dtype={dtype}, max_cached_shards={self.max_cached_shards}"
        )

    def __len__(self) -> int:
        return self.cumulative_sizes[-1]

    def _locate(self, idx: int) -> Tuple[int, int]:
        if idx < 0 or idx >= len(self):
            raise IndexError(f"Index {idx} out of range 0..{len(self)-1}")

        # linear scan is fine; could be binary search if you want
        for shard_idx, cum in enumerate(self.cumulative_sizes):
            if idx < cum:
                prev_cum = 0 if shard_idx == 0 else self.cumulative_sizes[shard_idx - 1]
                local_idx = idx - prev_cum
                return shard_idx, local_idx

        raise RuntimeError(f"Failed to locate index {idx}")

    def _get_shard_arrays(self, shard_idx: int) -> Tuple[np.ndarray, np.ndarray]:
        # LRU hit
        if shard_idx in self._shard_cache:
            x_arr, y_arr = self._shard_cache.pop(shard_idx)
            self._shard_cache[shard_idx] = (x_arr, y_arr)  # move to end
            return x_arr, y_arr

        # Cache miss: load full shard into RAM (NO memmap)
        x_path = self.x_paths[shard_idx]
        y_path = self.y_paths[shard_idx]

        x_arr = np.load(x_path)   # shape (N, 6, H, W)
        y_arr = np.load(y_path)   # shape (N,)

        # Ensure contiguous & writable; matches old fast path
        x_arr = np.ascontiguousarray(x_arr)
        y_arr = np.ascontiguousarray(y_arr)

        self._shard_cache[shard_idx] = (x_arr, y_arr)
        if len(self._shard_cache) > self.max_cached_shards:
            self._shard_cache.popitem(last=False)  # evict least-recently-used

        return x_arr, y_arr

    def __getitem__(self, idx: int):
        shard_idx, local_idx = self._locate(idx)
        x_arr, y_arr = self._get_shard_arrays(shard_idx)

        try:
            x = x_arr[local_idx]   # (6, H, W)
            y = y_arr[local_idx]   # scalar
        except Exception as e:
            raise RuntimeError(
                f"Failed to access local_idx={local_idx} in shard_idx={shard_idx}: {e}"
            )

        if x.ndim != 3 or x.shape[0] != 6:
            raise ValueError(
                f"Sample from shard_idx={shard_idx} local_idx={local_idx} has shape {x.shape}, "
                f"expected (6, H, W)."
            )

        # Copy is cheap compared to 470MB shard load; ensures writable, kills the warning
        x_tensor = torch.from_numpy(x.copy()).float()
        y_tensor = torch.tensor(int(y), dtype=torch.long)

        if self.transform is not None:
            x_tensor = self.transform(x_tensor)

        if self.return_paths:
            sample_id = f"{os.path.basename(self.x_paths[shard_idx])}#{local_idx}"
            return x_tensor, y_tensor, sample_id
        else:
            return x_tensor, y_tensor



# ─────────────────────────────────────────────────────────────
#  Genotype map & get_data_loader (unchanged API)
# ─────────────────────────────────────────────────────────────

GENOTYPE_MAP: Dict[str, int] = {
    "false": 0,
    "true": 1,
}

def _to_list(x):
    if x is None:
        return []
    if isinstance(x, (list, tuple)):
        return list(x)
    return [x]

def get_data_loader(
    data_dir: Union[str, List[str], Tuple],
    dataset_type: str,
    batch_size: int = 32,
    num_workers: int = 16,
    shuffle: bool = False,
    return_paths: bool = False,
):
    """
    NPY-sharded version of your original get_data_loader, same interface.

    Layout per root:
        root/
          train/
            shard_train_000_x.npy
            shard_train_000_y.npy
            ...
          val/
            shard_val_000_x.npy
            shard_val_000_y.npy
            ...

    Supports:
      • data_dir = "/path/to/root"
      • data_dir = ["/root1", "/root2", ...]
      • data_dir = (train_roots, val_roots)  # split-mode
    """
    # Decide roots & subfolders
    if (
        isinstance(data_dir, (list, tuple))
        and len(data_dir) == 2
        and (isinstance(data_dir[0], (str, list, tuple)) and isinstance(data_dir[1], (str, list, tuple)))
    ):
        roots = _to_list(data_dir[0] if dataset_type == "train" else data_dir[1])
        subfolders = ["train", "val"]  # your original slightly-unusual behavior
    else:
        roots = _to_list(data_dir)
        subfolders = [dataset_type]

    dataset_dirs: List[str] = []
    for r in roots:
        r = os.path.abspath(os.path.expanduser(r))
        for sf in subfolders:
            dataset_dirs.append(os.path.join(r, sf))

    missing = [p for p in dataset_dirs if not os.path.exists(p)]
    if missing:
        raise FileNotFoundError(f"Dataset path(s) do not exist: {missing}")

    # Same 6-channel normalization as before
    transform = transforms.Compose([
        transforms.Normalize(
            mean=[
                18.417816162109375,
                12.649129867553711,
                -0.5452527403831482,
                24.723854064941406,
                4.690611362457275,
                0.2813551473402196,
            ],
            std=[
                25.028322219848633,
                14.809632301330566,
                0.6181337833404541,
                29.972835540771484,
                7.9231791496276855,
                0.7659083659074717,
            ],
        )
    ])

    datasets: List[Dataset] = []
    for d in dataset_dirs:
        ds = NPYShardDataset(
            root_dir=d,
            transform=transform,
            return_paths=return_paths,
            max_cached_shards=2,   # <- keep this small to avoid too many open files
        )
        datasets.append(ds)

    if len(datasets) == 1:
        dataset = datasets[0]
    else:
        dataset = ConcatDataset(datasets)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
    )

    return loader, GENOTYPE_MAP


# (Optional) simple main for quick smoke test
if __name__ == "__main__":
    data_root = "/path/to/your_6channel_npy_sharded_dataset"  # contains train/ and val/
    batch_size = 16
    num_workers = 0

    if data_root == "/path/to/your_6channel_npy_sharded_dataset":
        print("🛑 Please update 'data_root' to your NPY-sharded dataset root (with train/ and val/).")
    else:
        try:
            print(f"\n--- Loading Training Data (Batch Size: {batch_size}) ---")
            train_loader, class_map = get_data_loader(
                data_dir=data_root,
                dataset_type="train",
                batch_size=batch_size,
                num_workers=num_workers,
                shuffle=True,
            )
            print(f"✅ Loaded {len(train_loader.dataset)} training samples.")
            print(f"Genotype map: {class_map}")

            for i, (data, labels) in enumerate(train_loader):
                print(f"Batch {i + 1}: data.shape={data.shape}, labels[0:5]={labels[:5]}")
                if i >= 1:
                    break

        except Exception as e:
            print(f"Error loading training data: {e}")
