import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader, ConcatDataset
import os
import numpy as np


class NpyDataset(Dataset):
    """
    --- REVISED ---
    Custom PyTorch Dataset to load 6-channel .npy files.
    Accepts .npy files with the shape (6, W, H).
    """

    def __init__(self, root_dir, transform=None, return_paths=False):
        self.root_dir = os.path.expanduser(root_dir)
        self.transform = transform
        self.return_paths = return_paths
        self.samples = []
        self.classes = []
        self.class_to_idx = {}

        if not os.path.isdir(self.root_dir):
            raise FileNotFoundError(f"Root directory not found: {self.root_dir}")

        class_folders = sorted([d.name for d in os.scandir(self.root_dir) if d.is_dir()])
        if not class_folders:
            raise FileNotFoundError(f"No class subdirectories found in {self.root_dir}")

        self.classes = class_folders
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(self.classes)}

        for cls_name in class_folders:
            class_path = os.path.join(self.root_dir, cls_name)
            for file_name in sorted(os.listdir(class_path)):
                if file_name.lower().endswith(".npy"):
                    file_path = os.path.join(class_path, file_name)
                    self.samples.append((file_path, self.class_to_idx[cls_name]))

        if len(self.samples) == 0:
            raise ValueError(f"No .npy files found in {self.root_dir}")

        print(
            f"Initialized NpyDataset from {self.root_dir}: Found {len(self.samples)} samples in {len(self.classes)} classes.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        file_path, label = self.samples[idx]
        try:
            image_np = np.load(file_path)
        except Exception as e:
            raise RuntimeError(f"Failed to load .npy file {file_path}: {e}")

        if not isinstance(image_np, np.ndarray):
            raise TypeError(f"File {file_path} did not load as a NumPy array (loaded type: {type(image_np)}).")

        # --- REVISED: Validate for 6 channels in (C, W, H) format ---
        # This check now assumes the input shape is (6, W, H) as requested.
        if image_np.ndim != 3 or image_np.shape[0] != 6:
            raise ValueError(
                f"Loaded .npy file {file_path} has an unsupported shape {image_np.shape}. "
                f"Expected shape is (6, W, H)."
            )

        image_tensor = torch.from_numpy(image_np.copy()).float()
        if self.transform:
            image_tensor = self.transform(image_tensor)

        if self.return_paths:
            return image_tensor, label, file_path
        else:
            return image_tensor, label


def get_data_loader(data_dir, dataset_type, batch_size=32, num_workers=16, shuffle: bool = False,
                    return_paths: bool = False):
    """
    Load dataset(s) using NpyDataset.

    Behavior:
      • If data_dir is (train_roots, val_roots):
          - Pick roots by dataset_type ("train" -> train_roots, "val" -> val_roots)
          - For EACH root, include BOTH 'train' and 'val' subfolders.
      • Else (str or list/tuple of str): back-compat mode:
          - Include only the requested `dataset_type` subfolder for each root.

    Returns: (DataLoader, class_to_idx)
    """
    import os
    from torch.utils.data import DataLoader, ConcatDataset
    from torchvision import transforms

    def _to_list(x):
        if x is None:
            return []
        if isinstance(x, (list, tuple)):
            return list(x)
        return [x]

    # Decide which roots and which subfolders to include
    if isinstance(data_dir, (list, tuple)) and len(data_dir) == 2 \
       and (isinstance(data_dir[0], (str, list, tuple)) and isinstance(data_dir[1], (str, list, tuple))):
        # New split-mode: (train_roots, val_roots)
        roots = _to_list(data_dir[0] if dataset_type == "train" else data_dir[1])
        subfolders = ["train", "val"]  # ← include BOTH for each root
    else:
        # Back-compat: single root or list of roots; only requested split
        roots = _to_list(data_dir)
        subfolders = [dataset_type]

    # Expand to concrete dataset directories
    dataset_dirs = []
    for r in roots:
        r = os.path.abspath(os.path.expanduser(r))
        for sf in subfolders:
            dataset_dirs.append(os.path.join(r, sf))

    # Existence check (strict, like original)
    missing = [p for p in dataset_dirs if not os.path.exists(p)]
    if missing:
        raise FileNotFoundError(f"Dataset path(s) do not exist: {missing}")

    # 6-channel normalization (unchanged placeholders)
    transform = transforms.Compose([
        transforms.Normalize(
            mean=[18.417816162109375, 12.649129867553711, -0.5452527403831482,
                  24.723854064941406, 4.690611362457275, 10.659969329833984],
            std=[25.028322219848633, 14.809632301330566, 0.6181337833404541,
                 29.972835540771484, 7.9231791496276855, 27.151996612548828]
        )
    ])

    # Build datasets and verify consistent class_to_idx
    datasets = []
    unified_class_to_idx = None
    for d in dataset_dirs:
        ds = NpyDataset(root_dir=d, transform=transform, return_paths=return_paths)
        if unified_class_to_idx is None:
            unified_class_to_idx = ds.class_to_idx
        else:
            if ds.class_to_idx != unified_class_to_idx:
                raise ValueError("class_to_idx mismatch across dataset roots/subfolders.")
        datasets.append(ds)

    dataset = datasets[0] if len(datasets) == 1 else ConcatDataset(datasets)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                        num_workers=num_workers, pin_memory=True)

    return loader, unified_class_to_idx


def get_testing_data_loader(data_dir, batch_size=32, num_workers=16, shuffle: bool = False,
                            return_paths: bool = False):
    """
    Build a DataLoader for TESTING where each root dir directly contains .npy files.
    No 'train'/'val'/'test' subfolders are assumed.

    Accepted inputs:
      • str: a single directory with .npy files
      • list/tuple[str]: multiple directories (concatenated)
      • (train_roots, val_roots): 2-tuple; the second element is used for testing

    Returns:
      (DataLoader, class_to_idx)
    """

    def _to_list(x):
        if x is None:
            return []
        if isinstance(x, (list, tuple)):
            return list(x)
        return [x]

    # Resolve testing roots
    if (
        isinstance(data_dir, (list, tuple)) and len(data_dir) == 2
        and isinstance(data_dir[0], (str, list, tuple))
        and isinstance(data_dir[1], (str, list, tuple))
    ):
        roots = _to_list(data_dir[1])  # use the "val/test" side for testing
    else:
        roots = _to_list(data_dir)

    dataset_dirs = [os.path.abspath(os.path.expanduser(r)) for r in roots]

    missing = [p for p in dataset_dirs if not os.path.isdir(p)]
    if missing:
        raise FileNotFoundError(f"Dataset directory(ies) do not exist: {missing}")

    # 6-channel normalization (same as your training)
    transform = transforms.Compose([
        transforms.Normalize(
            mean=[18.417816162109375, 12.649129867553711, -0.5452527403831482,
                  24.723854064941406, 4.690611362457275, 10.659969329833984],
            std=[25.028322219848633, 14.809632301330566, 0.6181337833404541,
                 29.972835540771484, 7.923179149780273, 27.151996612548828]
        )
    ])

    # Build datasets and ensure consistent mapping
    datasets = []
    unified_class_to_idx = None
    for d in dataset_dirs:
        ds = NpyDataset(root_dir=d, transform=transform, return_paths=return_paths)
        if unified_class_to_idx is None:
            unified_class_to_idx = ds.class_to_idx
        else:
            if ds.class_to_idx != unified_class_to_idx:
                raise ValueError(
                    f"class_to_idx mismatch across roots. "
                    f"First: {unified_class_to_idx} vs {ds.class_to_idx} in {d}"
                )
        datasets.append(ds)

    dataset = datasets[0] if len(datasets) == 1 else ConcatDataset(datasets)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=(num_workers > 0),
    )
    return loader, unified_class_to_idx


if __name__ == "__main__":
    # --- IMPORTANT ---
    # Update this path to the root directory containing your 'train', 'val', and 'test' folders.
    data_root = "/path/to/your_6channel_dataset"
    # --- IMPORTANT ---

    batch_size = 16
    num_workers = 0

    if data_root == "/path/to/your_6channel_dataset":
        print("🛑 Please update the 'data_root' variable in the script to your dataset's actual path.")
    else:
        try:
            print(f"\n--- Loading Training Data (Batch Size: {batch_size}) ---")
            train_loader, class_map = get_data_loader(
                data_dir=data_root,
                dataset_type="train",
                batch_size=batch_size,
                num_workers=num_workers,
                shuffle=True
            )
            print(f"✅ Loaded {len(train_loader.dataset)} training samples from {os.path.join(data_root, 'train')}.")
            print(f"Class-to-index mapping: {class_map}")

            print("\n--- Checking a few training batches ---")
            for i, (data, labels) in enumerate(train_loader):
                # Now data.shape will be [batch_size, 6, H, W]
                print(f"Batch {i + 1}: Data shape: {data.shape}, Labels: {labels[:5]}...")
                if i >= 2:
                    break
            if not train_loader:
                print("Train loader is empty.")

        except (FileNotFoundError, ValueError, RuntimeError) as e:
            print(f"Error loading training data: {e}")
        except Exception as e:
            print(f"An unexpected error occurred: {e}")