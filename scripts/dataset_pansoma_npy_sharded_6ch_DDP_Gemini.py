import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader, ConcatDataset
import numpy as np
import os
import glob


class NpyDataset(Dataset):
    """
    Hybrid Dataset with RAM Preloading support.
    """

    def __init__(self, root_dir, transform=None, return_paths=False, preload_ram=False):
        # root_dir can be a directory OR a direct path to a .npy file
        self.root_path = os.path.abspath(os.path.expanduser(root_dir))
        self.transform = transform
        self.return_paths = return_paths
        self.preload_ram = preload_ram
        self.samples = []
        self.mode = None
        self.class_to_idx = {}
        self.data_cache = {}  # <--- NEW: Stores {filename: numpy_array} in RAM

        # ---------------------------------------------------------
        # 1. CASE: Specific File (Single Shard Mode)
        # ---------------------------------------------------------
        if os.path.isfile(self.root_path) and self.root_path.endswith("_data.npy"):
            self.mode = 'sharded'
            self._init_sharded([self.root_path])

        # ---------------------------------------------------------
        # 2. CASE: Directory
        # ---------------------------------------------------------
        elif os.path.isdir(self.root_path):
            shard_files = sorted(glob.glob(os.path.join(self.root_path, "*_data.npy")))
            if len(shard_files) > 0:
                self.mode = 'sharded'
                self._init_sharded(shard_files)
            else:
                self.mode = 'folder'
                self._init_folder()
        else:
            raise FileNotFoundError(f"Path not found: {self.root_path}")

        if len(self.samples) == 0:
            raise ValueError(f"No data found in {self.root_path}")

        if self.mode == 'sharded':
            unique_labels = sorted(list(set(s[2] for s in self.samples)))
            self.class_to_idx = {str(l): l for l in unique_labels}

            # --- PRELOAD INTO RAM IF REQUESTED ---
            if self.preload_ram:
                self._preload_all_data()

    def _init_sharded(self, shard_files):
        for data_path in shard_files:
            label_path = data_path.replace("_data.npy", "_labels.npy")
            if not os.path.exists(label_path):
                print(f"WARNING: Missing label file for {data_path}. Skipping.")
                continue

            labels = np.load(label_path)
            for local_idx, label in enumerate(labels):
                self.samples.append((data_path, local_idx, int(label)))

    def _init_folder(self):
        class_folders = sorted([d.name for d in os.scandir(self.root_path) if d.is_dir()])
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(class_folders)}

        for cls_name in class_folders:
            class_path = os.path.join(self.root_path, cls_name)
            label = self.class_to_idx[cls_name]
            for file_name in sorted(os.listdir(class_path)):
                if file_name.lower().endswith(".npy"):
                    file_path = os.path.join(class_path, file_name)
                    self.samples.append((file_path, -1, label))

    def _preload_all_data(self):
        """Loads unique files into RAM dictionary."""
        unique_files = sorted(list(set(s[0] for s in self.samples)))
        # Only print if we are loading a directory (to avoid spamming logs for 100 separate shard loaders)
        if len(unique_files) > 1:
            print(f"Preloading {len(unique_files)} files into RAM...")

        for fpath in unique_files:
            if fpath not in self.data_cache:
                try:
                    # Load FULL array into RAM. No mmap_mode.
                    self.data_cache[fpath] = np.load(fpath)
                except Exception as e:
                    raise RuntimeError(f"Failed to preload {fpath}: {e}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, local_idx, label = self.samples[idx]

        try:
            if self.mode == 'sharded':
                if self.preload_ram and path in self.data_cache:
                    # --- FAST PATH: Read from RAM Cache ---
                    image_np = self.data_cache[path][local_idx].copy()
                else:
                    # --- DISK PATH: Memory Map ---
                    mmap_arr = np.load(path, mmap_mode='r')
                    image_np = mmap_arr[local_idx].copy()
            else:
                image_np = np.load(path)
        except Exception as e:
            raise RuntimeError(f"Failed to load {path} at idx {local_idx}: {e}")

        if image_np.ndim != 3 or image_np.shape[0] != 6:
            raise ValueError(f"Shape mismatch: {image_np.shape}. Expected (6, H, W).")

        image_tensor = torch.from_numpy(image_np).float()

        if self.transform:
            image_tensor = self.transform(image_tensor)

        if self.return_paths:
            return image_tensor, label, path
        else:
            return image_tensor, label


def get_data_loader(data_dir, dataset_type, batch_size=32, num_workers=16, shuffle: bool = False,
                    return_paths: bool = False):
    """
    Standard Concatenated Loader (Used for Validation).
    NOTE: We generally DO NOT preload validation data to save RAM for training,
    unless you explicitly modify this to pass preload_ram=True.
    """

    def _to_list(x):
        if x is None: return []
        if isinstance(x, (list, tuple)): return list(x)
        return [x]

    if isinstance(data_dir, (list, tuple)) and len(data_dir) == 2 \
            and isinstance(data_dir[0], (list, tuple, str)) \
            and isinstance(data_dir[1], (list, tuple, str)):
        roots = _to_list(data_dir[0] if dataset_type == "train" else data_dir[1])
        subfolders = ["train", "val"]
    else:
        roots = _to_list(data_dir)
        subfolders = [dataset_type]

    final_dirs = []
    for r in roots:
        r = os.path.abspath(os.path.expanduser(r))
        if len(glob.glob(os.path.join(r, "*_data.npy"))) > 0:
            final_dirs.append(r)
        else:
            found_sub = False
            for sf in subfolders:
                target = os.path.join(r, sf)
                if os.path.exists(target):
                    final_dirs.append(target)
                    found_sub = True
            if not found_sub and os.path.exists(r):
                final_dirs.append(r)

    if not final_dirs:
        raise FileNotFoundError(f"No valid '{dataset_type}' folders found in {roots}")

    transform = transforms.Compose([
        transforms.Normalize(
            mean=[18.417816162109375, 12.649129867553711, -0.5452527403831482,
                  24.723854064941406, 4.690611362457275, 0.2813551473402196],
            std=[25.028322219848633, 14.809632301330566, 0.6181337833404541,
                 29.972835540771484, 7.9231791496276855, 0.7659083659074717]
        )
    ])

    datasets = []
    unified_map = None

    for d in final_dirs:
        try:
            # Validation usually doesn't need preloading, kept False to save RAM
            ds = NpyDataset(root_dir=d, transform=transform, return_paths=return_paths, preload_ram=False)
            if unified_map is None:
                unified_map = ds.class_to_idx
            datasets.append(ds)
        except (ValueError, FileNotFoundError):
            continue

    if not datasets:
        raise RuntimeError("Could not create any datasets.")

    final_dataset = ConcatDataset(datasets)
    loader = DataLoader(final_dataset, batch_size=batch_size, shuffle=shuffle,
                        num_workers=num_workers, pin_memory=True, persistent_workers=(num_workers > 0))

    return loader, unified_map