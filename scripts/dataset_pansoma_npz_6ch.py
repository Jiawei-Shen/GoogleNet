import os
import glob
from typing import List, Tuple, Dict, Union

import numpy as np
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader, ConcatDataset

from collections import OrderedDict

class NPZShardDataset(Dataset):
    """
    Dataset for NPZ shards.

    Each .npz file is expected to contain two arrays:
      - features: shape (N, 6, H, W)
      - labels:   shape (N,)

    Supported key conventions:
      • ('x', 'y')
      • ('data', 'labels')   <-- your case
      • ('arr_0', 'arr_1')   (when exactly two arrays exist)

    To avoid insane slowness, we cache a small number of fully
    loaded shards in RAM per worker (LRU cache).
    """

    def __init__(
        self,
        root_dir: str,
        transform=None,
        return_paths: bool = False,
        max_cached_shards: int = 1,  # <= IMPORTANT: RAM vs speed tradeoff
    ):
        self.root_dir = os.path.abspath(os.path.expanduser(root_dir))
        self.transform = transform
        self.return_paths = return_paths
        self.max_cached_shards = max_cached_shards

        if not os.path.isdir(self.root_dir):
            raise FileNotFoundError(f"Root directory not found: {self.root_dir}")

        # Find all .npz files
        shard_paths = sorted(
            [
                os.path.join(self.root_dir, f)
                for f in os.listdir(self.root_dir)
                if f.lower().endswith(".npz")
            ]
        )

        if not shard_paths:
            raise ValueError(f"No .npz files found in {self.root_dir}")

        self.shard_paths: List[str] = []
        self.shard_sizes: List[int] = []
        self.cumulative_sizes: List[int] = []

        self.x_key: str = ""
        self.y_key: str = ""

        total = 0

        for p in shard_paths:
            try:
                with np.load(p) as data:
                    files = list(data.files)
                    key_set = set(files)

                    x_key = y_key = None
                    if "x" in key_set and "y" in key_set:
                        x_key, y_key = "x", "y"
                    elif "data" in key_set and "labels" in key_set:
                        x_key, y_key = "data", "labels"
                    elif set(files) == {"arr_0", "arr_1"} and len(files) == 2:
                        x_key, y_key = "arr_0", "arr_1"
                    else:
                        print(
                            f"[NPZShardDataset] WARNING: skipping shard {p} – "
                            f"could not find ('x','y'), ('data','labels') or ('arr_0','arr_1'). "
                            f"Keys present: {files}"
                        )
                        continue

                    x_arr = data[x_key]
                    y_arr = data[y_key]

                    n = int(x_arr.shape[0])
                    if y_arr.shape[0] != n:
                        raise ValueError(
                            f"{p}: feature and label length mismatch: "
                            f"{x_arr.shape[0]} vs {y_arr.shape[0]}"
                        )

                    if x_arr.ndim != 4 or x_arr.shape[1] != 6:
                        raise ValueError(
                            f"{p}: expected x shape (N, 6, H, W), got {x_arr.shape}"
                        )

            except Exception as e:
                print(f"[NPZShardDataset] WARNING: skipping bad shard {p}: {e}")
                continue

            if not self.shard_paths:
                self.x_key, self.y_key = x_key, y_key
            else:
                if x_key != self.x_key or y_key != self.y_key:
                    raise RuntimeError(
                        f"Shard {p} uses different keys ({x_key}, {y_key}) "
                        f"than previous shards ({self.x_key}, {self.y_key})."
                    )

            self.shard_paths.append(p)
            self.shard_sizes.append(n)
            total += n
            self.cumulative_sizes.append(total)

        if total == 0 or not self.shard_paths:
            raise ValueError(
                f"NPZShardDataset from {self.root_dir} has zero usable samples "
                f"(all shards invalid or skipped)."
            )

        # LRU cache: shard_idx -> (x_arr, y_arr)
        self._shard_cache: "OrderedDict[int, Tuple[np.ndarray, np.ndarray]]" = OrderedDict()

        print(
            f"Initialized NPZShardDataset from {self.root_dir}: "
            f"{len(self.shard_paths)} shards, {total} samples total. "
            f"Using keys: x='{self.x_key}', y='{self.y_key}'. "
            f"max_cached_shards={self.max_cached_shards}"
        )

    def __len__(self) -> int:
        return self.cumulative_sizes[-1]

    def _locate(self, idx: int) -> Tuple[int, int]:
        if idx < 0 or idx >= len(self):
            raise IndexError(f"Index {idx} out of range 0..{len(self)-1}")

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
            self._shard_cache[shard_idx] = x_arr, y_arr  # move to end (most recent)
            return x_arr, y_arr

        # Cache miss: load and maybe evict oldest
        path = self.shard_paths[shard_idx]
        with np.load(path) as data:
            x_arr = data[self.x_key]     # (N, 6, H, W), int8
            y_arr = data[self.y_key]     # (N,)

            # Make sure they're real ndarrays, not weird views
            x_arr = np.ascontiguousarray(x_arr)
            y_arr = np.ascontiguousarray(y_arr)

        self._shard_cache[shard_idx] = (x_arr, y_arr)
        # Evict least-recently-used
        if len(self._shard_cache) > self.max_cached_shards:
            self._shard_cache.popitem(last=False)

        return x_arr, y_arr

    def __getitem__(self, idx: int):
        shard_idx, local_idx = self._locate(idx)
        x_arr, y_arr = self._get_shard_arrays(shard_idx)

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

        x_tensor = torch.from_numpy(x).float()  # per-sample cast, cheap
        y_tensor = torch.tensor(int(y), dtype=torch.long)

        if self.transform is not None:
            x_tensor = self.transform(x_tensor)

        if self.return_paths:
            sample_id = f"{os.path.basename(self.shard_paths[shard_idx])}#{local_idx}"
            return x_tensor, y_tensor, sample_id
        else:
            return x_tensor, y_tensor


# We hardcode the genotype map to match your original ("false", "true") binary setup.
# Adjust if you have more classes or different ordering.
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
    NPZ-based version of your original get_data_loader, with the same interface.

    Behavior:
      • If data_dir is (train_roots, val_roots):
          - Pick roots by dataset_type ("train" -> train_roots, "val" -> val_roots)
          - For EACH root, include BOTH 'train' and 'val' subfolders.
            (keeps your previous, slightly unusual behavior intact)
      • Else (str or list/tuple of str): back-compat mode:
          - Include only the requested `dataset_type` subfolder for each root.

    Layout expected:
        root/
          train/
            shard_0001.npz
            shard_0002.npz
            ...
          val/
            shard_0001.npz
            shard_0002.npz
            ...

    Returns:
        (DataLoader, genotype_map)
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

    # Normalization (copied from your original script)
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
        ds = NPZShardDataset(root_dir=d, transform=transform, return_paths=return_paths)
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

    # We return the genotype map directly
    return loader, GENOTYPE_MAP


def get_inference_data_loader(
    data_dir,
    batch_size: int = 256,
    num_workers: int = 4,
    shuffle: bool = False,
    return_paths: bool = True,
    enforce_float32: bool = True,
    pin_memory: bool = True,
):
    """
    Inference loader for unlabeled NPZ shards.

    Expected each .npz contains:
        x: (N, 6, H, W)

    Returns:
        (loader, {})  # empty class map signals 'unlabeled' mode
    """

    def _to_list_local(x):
        if x is None:
            return []
        if isinstance(x, (list, tuple)):
            return list(x)
        return [x]

    # Resolve roots: str | list[str] | (train_roots, val/test_roots)
    if (
        isinstance(data_dir, (list, tuple))
        and len(data_dir) == 2
        and isinstance(data_dir[1], (str, list, tuple))
    ):
        roots = _to_list_local(data_dir[1])  # second half for inference
    else:
        roots = _to_list_local(data_dir)

    # Collect *.npz recursively
    files: List[str] = []
    for r in roots:
        r = os.path.abspath(os.path.expanduser(r))
        if not os.path.isdir(r):
            raise FileNotFoundError(f"Not a directory: {r}")
        found = glob.glob(os.path.join(r, "**", "*.npz"), recursive=True)
        files.extend(found)

    seen = set()
    files = [f for f in sorted(files) if not (f in seen or seen.add(f))]
    if not files:
        raise FileNotFoundError(f"No .npz files found under: {', '.join(map(str, roots))}")

    class _NPZUnlabeledDataset(Dataset):
        def __init__(self, paths: List[str], return_paths: bool):
            self.paths = paths
            self.return_paths = return_paths

        def __len__(self):
            return len(self.paths)

        def __getitem__(self, idx: int):
            path = self.paths[idx]
            data = np.load(path)
            if "x" not in data:
                raise KeyError(f"{path} does not contain key 'x'")
            x = data["x"]  # (N, 6, H, W) or maybe a single sample

            if x.ndim == 3 and x.shape[0] == 6:
                # Single sample
                arr = x
            elif x.ndim == 4 and x.shape[1] == 6:
                # Multiple samples; here we only support "one-sample-per-npz" for inference.
                # You can extend to pick all samples if needed.
                raise RuntimeError(
                    f"{path}: 'x' has shape {x.shape}; for inference loader we currently "
                    f"only support single-sample NPZ (6,H,W)."
                )
            else:
                raise RuntimeError(f"{path}: unexpected 'x' shape {x.shape}, expected (6,H,W).")

            t = torch.from_numpy(arr)
            if enforce_float32 and t.dtype != torch.float32:
                t = t.float()

            y = -1  # unlabeled
            if self.return_paths:
                return t, y, path
            return t, y

    ds = _NPZUnlabeledDataset(files, return_paths=return_paths)

    loader_kwargs = dict(
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=False,
    )
    if num_workers and num_workers > 0:
        loader_kwargs["prefetch_factor"] = 2

    loader = DataLoader(ds, **loader_kwargs)
    return loader, {}  # unlabeled mode


if __name__ == "__main__":
    # Simple smoke test: update this path if you want to test directly.
    data_root = "/path/to/your_6channel_npz_dataset"

    batch_size = 16
    num_workers = 0

    if data_root == "/path/to/your_6channel_npz_dataset":
        print("🛑 Please update 'data_root' to your NPZ dataset root (with train/ and val/).")
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
