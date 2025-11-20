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
        self.data_cache = {}

        # Case 1: Specific File (Single Shard Mode)
        if os.path.isfile(self.root_path) and self.root_path.endswith("_data.npy"):
            self.mode = 'sharded'
            self._init_sharded([self.root_path])

        # Case 2: Directory
        elif os.path.isdir(self.root_path):
            # Look for shards
            shard_files = sorted(glob.glob(os.path.join(self.root_path, "*_data.npy")))

            # DEBUG PRINT: See what the script actually finds
            # print(f"DEBUG: Checking {self.root_path} -> Found {len(shard_files)} shards.")

            if len(shard_files) > 0:
                self.mode = 'sharded'
                self._init_sharded(shard_files)
            else:
                # If no shards, assume it's a Class Folder structure (0/, 1/)
                self.mode = 'folder'
                self._init_folder()
        else:
            raise FileNotFoundError(f"Path not found: {self.root_path}")

        if len(self.samples) == 0:
            raise ValueError(f"No data found in {self.root_path} (Checked for shards and class folders).")

        # Infer classes
        unique_labels = sorted(list(set(s[2] for s in self.samples)))
        self.class_to_idx = {str(l): l for l in unique_labels}

        # RAM Preload (Only for single file mode to be safe)
        if self.preload_ram and self.mode == 'sharded':
            if len(set(s[0] for s in self.samples)) == 1:
                data_path = self.samples[0][0]
                try:
                    self.data_in_ram = np.load(data_path)
                except Exception as e:
                    raise RuntimeError(f"Failed to load {data_path} into RAM: {e}")

    def _init_sharded(self, shard_files):
        for data_path in shard_files:
            label_path = data_path.replace("_data.npy", "_labels.npy")
            if not os.path.exists(label_path):
                raise FileNotFoundError(f"Missing label file for {data_path}")

            labels = np.load(label_path)
            for local_idx, label in enumerate(labels):
                self.samples.append((data_path, local_idx, int(label)))

    def _init_folder(self):
        class_folders = sorted([d.name for d in os.scandir(self.root_path) if d.is_dir()])
        if not class_folders:
            raise FileNotFoundError(f"No shards (*_data.npy) AND no class subdirectories found in {self.root_path}")

        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(class_folders)}

        for cls_name in class_folders:
            class_path = os.path.join(self.root_path, cls_name)
            label = self.class_to_idx[cls_name]
            for file_name in sorted(os.listdir(class_path)):
                if file_name.lower().endswith(".npy"):
                    file_path = os.path.join(class_path, file_name)
                    self.samples.append((file_path, -1, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, local_idx, label = self.samples[idx]

        try:
            if self.mode == 'sharded':
                if getattr(self, 'data_in_ram', None) is not None:
                    image_np = self.data_in_ram[local_idx].copy()
                else:
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
        # If the root itself has shards, use it
        if len(glob.glob(os.path.join(r, "*_data.npy"))) > 0:
            final_dirs.append(r)
        else:
            # Otherwise look for subfolders
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

    print(f"Scanning {len(final_dirs)} directories for '{dataset_type}' data...")

    for d in final_dirs:
        try:
            # --- SKIPPING LOGIC ---
            # If a folder is empty or invalid, skip it instead of crashing
            ds = NpyDataset(root_dir=d, transform=transform, return_paths=return_paths, preload_ram=False)

            if unified_map is None:
                unified_map = ds.class_to_idx
            datasets.append(ds)
        except (ValueError, FileNotFoundError) as e:
            # This print confirms we are safely skipping empty folders
            print(f"   -> Skipping empty/invalid dir: {d}")
            continue

    if not datasets:
        raise RuntimeError(f"Could not create ANY datasets for '{dataset_type}'. Please check your paths.")

    print(f"   -> Successfully loaded {len(datasets)} valid datasets.")

    final_dataset = ConcatDataset(datasets)
    loader = DataLoader(final_dataset, batch_size=batch_size, shuffle=shuffle,
                        num_workers=num_workers, pin_memory=True, persistent_workers=(num_workers > 0))

    return loader, unified_map