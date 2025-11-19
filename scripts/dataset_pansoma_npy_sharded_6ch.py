import os
import glob
from typing import List, Tuple, Dict, Union

import numpy as np
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader, ConcatDataset

# ---------------- NPY-sharded Dataset ----------------
class NPYShardDataset(Dataset):
    """
    Dataset for NPY shards.

    Each shard is represented by two files in `root_dir`:
      - shard_x.npy: shape (N, 6, H, W)  (features)
      - shard_y.npy: shape (N,)          (labels)

    Example:
        shard_00000_x.npy
        shard_00000_y.npy

    All x-files and y-files are opened with mmap_mode="r" for
    true random access without decompression.
    """

    def __init__(
        self,
        root_dir: str,
        transform=None,
        return_paths: bool = False,
    ):
        self.root_dir = os.path.abspath(os.path.expanduser(root_dir))
        self.transform = transform
        self.return_paths = return_paths

        if not os.path.isdir(self.root_dir):
            raise FileNotFoundError(f"Root directory not found: {self.root_dir}")

        # Find all *_x.npy shards
        x_paths = sorted(
            [
                os.path.join(self.root_dir, f)
                for f in os.listdir(self.root_dir)
                if f.endswith("_x.npy")
            ]
        )

        if not x_paths:
            raise ValueError(f"No *_x.npy shard files found in {self.root_dir}")

        self.shard_x_paths: List[str] = []
        self.shard_y_paths: List[str] = []
        self.shard_sizes: List[int] = []
        self.cumulative_sizes: List[int] = []

        # We keep memmap arrays in memory for fast indexing
        self._x_arrays: List[np.memmap] = []
        self._y_arrays: List[np.memmap] = []

        total = 0

        for x_path in x_paths:
            base = x_path[:-6]  # strip "_x.npy"
            y_path = base + "_y.npy"

            if not os.path.exists(y_path):
                raise FileNotFoundError(f"Missing label shard for {x_path}: expected {y_path}")

            # Open with memmap
            x_arr = np.load(x_path, mmap_mode="r")
            y_arr = np.load(y_path, mmap_mode="r")

            if x_arr.ndim != 4 or x_arr.shape[1] != 6:
                raise ValueError(
                    f"{x_path}: expected x shape (N, 6, H, W), got {x_arr.shape}"
                )

            n = int(x_arr.shape[0])
            if y_arr.shape[0] != n:
                raise ValueError(
                    f"{x_path}: feature and label length mismatch: {x_arr.shape[0]} vs {y_arr.shape[0]}"
                )

            self.shard_x_paths.append(x_path)
            self.shard_y_paths.append(y_path)
            self._x_arrays.append(x_arr)
            self._y_arrays.append(y_arr)

            self.shard_sizes.append(n)
            total += n
            self.cumulative_sizes.append(total)

        if total == 0:
            raise ValueError(f"NPYShardDataset from {self.root_dir} has zero samples.")

        print(
            f"Initialized NPYShardDataset from {self.root_dir}: "
            f"{len(self.shard_x_paths)} shards, {total} samples total."
        )

    def __len__(self) -> int:
        return self.cumulative_sizes[-1]

    def _locate(self, idx: int) -> Tuple[int, int]:
        if idx < 0 or idx >= len(self):
            raise IndexError(f"Index {idx} out of range 0..{len(self)-1}")

        # simple linear scan is fine given shard count is small (~hundreds)
        for shard_idx, cum in enumerate(self.cumulative_sizes):
            if idx < cum:
                prev_cum = 0 if shard_idx == 0 else self.cumulative_sizes[shard_idx - 1]
                local_idx = idx - prev_cum
                return shard_idx, local_idx

        raise RuntimeError(f"Failed to locate index {idx}")

    def __getitem__(self, idx: int):
        shard_idx, local_idx = self._locate(idx)

        x_arr = self._x_arrays[shard_idx]
        y_arr = self._y_arrays[shard_idx]

        try:
            x = x_arr[local_idx]    # (6, H, W)
            y = y_arr[local_idx]    # scalar
        except Exception as e:
            raise RuntimeError(
                f"Failed to access local_idx={local_idx} in shard_idx={shard_idx}: {e}"
            )

        if x.ndim != 3 or x.shape[0] != 6:
            raise ValueError(
                f"Sample from shard_idx={shard_idx} local_idx={local_idx} has shape {x.shape}, "
                f"expected (6, H, W)."
            )

        x_tensor = torch.from_numpy(x).float()  # per-sample cast to float32
        y_tensor = torch.tensor(int(y), dtype=torch.long)

        if self.transform is not None:
            x_tensor = self.transform(x_tensor)

        if self.return_paths:
            sample_id = f"{os.path.basename(self.shard_x_paths[shard_idx])}#{local_idx}"
            return x_tensor, y_tensor, sample_id
        else:
            return x_tensor, y_tensor


# We hardcode the genotype map to match your original ("false", "true") binary setup.
GENOTYPE_MAP: Dict[str, int] = {
    "false": 0,
    "true": 1,
}


def _to_list(x) -> List[str]:
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
    NPY-sharded version of your original get_data_loader.

    Behavior:
      • If data_dir is (train_roots, val_roots):
          - Pick roots by dataset_type ("train" -> train_roots, "val" -> val_roots)
          - For EACH root, include BOTH 'train' and 'val' subfolders
            (to preserve your previous behavior).
      • Else (str or list/tuple of str): back-compat mode:
          - Include only the requested `dataset_type` subfolder for each root.

    Layout expected:
        root/
          train/
            shard_00000_x.npy
            shard_00000_y.npy
            ...
          val/
            shard_00000_x.npy
            shard_00000_y.npy
            ...
    """
    # Decide roots & subfolders
    if (
        isinstance(data_dir, (list, tuple))
        and len(data_dir) == 2
        and (isinstance(data_dir[0], (str, list, tuple)) and isinstance(data_dir[1], (str, list, tuple)))
    ):
        # New split-mode: (train_roots, val_roots)
        roots = _to_list(data_dir[0] if dataset_type == "train" else data_dir[1])
        subfolders = ["train", "val"]  # include BOTH for each root (your old behavior)
    else:
        # Back-compat: single root or list of roots; only requested split
        roots = _to_list(data_dir)
        subfolders = [dataset_type]

    # Build actual dataset dirs
    dataset_dirs: List[str] = []
    for r in roots:
        r = os.path.abspath(os.path.expanduser(r))
        for sf in subfolders:
            dataset_dirs.append(os.path.join(r, sf))

    # Existence check
    missing = [p for p in dataset_dirs if not os.path.exists(p)]
    if missing:
        raise FileNotFoundError(f"Dataset path(s) do not exist: {missing}")

    # Normalization (same stats as your original)
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

    # Build datasets
    datasets: List[Dataset] = []
    for d in dataset_dirs:
        ds = NPYShardDataset(root_dir=d, transform=transform, return_paths=return_paths)
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

    # Return loader and genotype map
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
