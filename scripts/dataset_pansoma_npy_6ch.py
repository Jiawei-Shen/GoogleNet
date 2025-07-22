import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
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


def get_data_loader(data_dir, dataset_type, batch_size=32, num_workers=8, shuffle: bool = False,
                    return_paths: bool = False):
    """
    Load dataset from the given path using NpyDataset.
    Returns a DataLoader and a class-to-index mapping.
    """
    dataset_path = os.path.join(data_dir, dataset_type)

    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset path {dataset_path} does not exist.")

    # --- REVISED: Updated normalization for 6 channels ---
    # 🚨 IMPORTANT: You must calculate and replace these placeholder values
    # with the actual mean and std of your 6-channel dataset.
    transform = transforms.Compose([
        transforms.Normalize(mean = [18.417816162109375, 12.649129867553711, -0.5452527403831482, 24.723854064941406, 4.690611362457275, 10.659969329833984],
                             std  = [25.028322219848633, 14.809632301330566, 0.6181337833404541, 29.972835540771484, 7.9231791496276855, 27.151996612548828])
    ])

    dataset = NpyDataset(root_dir=dataset_path, transform=transform, return_paths=return_paths)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=True)

    return loader, dataset.class_to_idx


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