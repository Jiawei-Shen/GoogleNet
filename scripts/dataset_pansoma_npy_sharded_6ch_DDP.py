import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader, ConcatDataset
import numpy as np
import os
import glob


class NpyDataset(Dataset):
    """
    Hybrid Dataset that supports two modes:
    1. Sharded Mode (New): Reads paired *_data.npy and *_labels.npy files.
       - Efficiently loads labels into RAM.
       - Keeps data on disk (mmap) until needed.
    2. Folder Mode (Old): Reads individual .npy files inside class folders (0/, 1/).
    """

    def __init__(self, root_dir, transform=None, return_paths=False):
        self.root_dir = os.path.expanduser(root_dir)
        self.transform = transform
        self.return_paths = return_paths
        self.samples = []
        self.mode = None

        if not os.path.isdir(self.root_dir):
            raise FileNotFoundError(f"Root directory not found: {self.root_dir}")

        # ---------------------------------------------------------
        # DETECT MODE: Check for Sharded files first (*_data.npy)
        # ---------------------------------------------------------
        shard_files = sorted(glob.glob(os.path.join(self.root_dir, "*_data.npy")))

        if len(shard_files) > 0:
            self.mode = 'sharded'
            self._init_sharded(shard_files)
        else:
            self.mode = 'folder'
            self._init_folder()

        # Common validation
        if len(self.samples) == 0:
            raise ValueError(f"No data found in {self.root_dir} (Checked for shards and class folders).")

        # Derive classes from loaded data to ensure consistency
        # (For sharded data, we infer from the labels found)
        if self.mode == 'sharded':
            # Get unique labels from the loaded samples
            unique_labels = sorted(list(set(s[2] for s in self.samples)))
            self.class_to_idx = {str(l): l for l in unique_labels}

        print(f"[{self.mode.upper()}] Initialized {self.root_dir}: {len(self.samples)} samples.")

    def _init_sharded(self, shard_files):
        """
        Loads metadata for Sharded format. 
        We load all labels into memory immediately (fast) to build the index.
        """
        print(f"Found {len(shard_files)} shards. Indexing...")

        for data_path in shard_files:
            # Find corresponding label file
            label_path = data_path.replace("_data.npy", "_labels.npy")
            if not os.path.exists(label_path):
                raise FileNotFoundError(f"Missing label file for {data_path}")

            # Load labels into RAM (Integers are small, this is safe)
            # shape: (4096,)
            labels = np.load(label_path)

            # Store tuple: (path_to_shard_data, index_inside_shard, label_value)
            for local_idx, label in enumerate(labels):
                self.samples.append((data_path, local_idx, int(label)))

    def _init_folder(self):
        """
        Legacy mode: scans subfolders '0', '1', etc.
        """
        class_folders = sorted([d.name for d in os.scandir(self.root_dir) if d.is_dir()])
        if not class_folders:
            raise FileNotFoundError(f"No class subdirectories or shards found in {self.root_dir}")

        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(class_folders)}

        for cls_name in class_folders:
            class_path = os.path.join(self.root_dir, cls_name)
            label = self.class_to_idx[cls_name]

            # Recursively or standard listdir
            for file_name in sorted(os.listdir(class_path)):
                if file_name.lower().endswith(".npy"):
                    file_path = os.path.join(class_path, file_name)
                    # For folder mode, local_idx is -1 (unused)
                    self.samples.append((file_path, -1, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        # Retrieve metadata
        path, local_idx, label = self.samples[idx]

        try:
            if self.mode == 'sharded':
                # 1. Load specific shard in mmap mode (lazy load, no RAM spike)
                # Shape of mmap: (N, 6, H, W)
                # We only read ONE sample from disk here
                mmap_arr = np.load(path, mmap_mode='r')
                image_np = mmap_arr[local_idx].copy()  # Copy to detach from mmap file handle
            else:
                # Folder mode: load entire file
                image_np = np.load(path)

        except Exception as e:
            raise RuntimeError(f"Failed to load {path} at idx {local_idx}: {e}")

        # Shape Validation: Expected (6, H, W)
        # Note: Your logs say (4096, 6, 201, 100), so a single item is (6, 201, 100).
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
    Robust Loader that supports:
    1. Lists of paths (P25, P50, etc.)
    2. Split Tuple (train_roots, val_roots)
    3. Single String path
    """

    def _to_list(x):
        if x is None: return []
        if isinstance(x, (list, tuple)): return list(x)
        return [x]

    # 1. Resolve Roots based on input type
    if isinstance(data_dir, (list, tuple)) and len(data_dir) == 2 \
            and isinstance(data_dir[0], (list, tuple, str)) \
            and isinstance(data_dir[1], (list, tuple, str)):
        # Mode: (train_roots, val_roots)
        roots = _to_list(data_dir[0] if dataset_type == "train" else data_dir[1])
        # In split mode, we assume the roots point directly to data, 
        # OR we assume standard structure. Let's look for standard structure first.
        subfolders = ["train", "val"]
    else:
        # Mode: Single root or List of roots (e.g. P25, P50)
        roots = _to_list(data_dir)
        subfolders = [dataset_type]

    # 2. Build actual directory paths
    final_dirs = []
    for r in roots:
        r = os.path.abspath(os.path.expanduser(r))
        # Check if the root *itself* contains shards (User provided direct path to train/)
        if len(glob.glob(os.path.join(r, "*_data.npy"))) > 0:
            final_dirs.append(r)
        else:
            # Otherwise, look for subfolders (train/ or val/)
            found_sub = False
            for sf in subfolders:
                target = os.path.join(r, sf)
                if os.path.exists(target):
                    final_dirs.append(target)
                    found_sub = True

            if not found_sub and os.path.exists(r):
                # Fallback: maybe the folder itself is the dataset (legacy structure)
                final_dirs.append(r)

    if not final_dirs:
        raise FileNotFoundError(f"No valid '{dataset_type}' folders found in {roots}")

    # 3. Define Transform (6-channel normalization)
    transform = transforms.Compose([
        transforms.Normalize(
            mean=[18.417816162109375, 12.649129867553711, -0.5452527403831482,
                  24.723854064941406, 4.690611362457275, 0.2813551473402196],
            std=[25.028322219848633, 14.809632301330566, 0.6181337833404541,
                 29.972835540771484, 7.9231791496276855, 0.7659083659074717]
        )
    ])

    # 4. Create Datasets
    datasets = []
    unified_map = None

    print(f"Loading {dataset_type} data from {len(final_dirs)} directories...")

    for d in final_dirs:
        try:
            ds = NpyDataset(root_dir=d, transform=transform, return_paths=return_paths)
            if unified_map is None:
                unified_map = ds.class_to_idx
            datasets.append(ds)
        except (ValueError, FileNotFoundError) as e:
            print(f"Skipping empty/invalid dir {d}: {e}")
            continue

    if not datasets:
        raise RuntimeError("Could not create any datasets from provided paths.")

    # 5. Concatenate
    final_dataset = ConcatDataset(datasets)

    # 6. Loader
    loader = DataLoader(
        final_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=(num_workers > 0)
    )

    return loader, unified_map