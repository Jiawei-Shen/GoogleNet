import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import os
import numpy as np


class NpyDataset(Dataset):
    """
    Custom PyTorch Dataset to load 6-channel .npy files from subdirectories as labeled data.
    Works like torchvision.datasets.ImageFolder but for .npy files.
    """

    def __init__(self, root_dir, transform=None):
        """
        :param root_dir: Path to dataset folder (e.g., "/path/to/dataset/train")
        :param transform: PyTorch transforms to apply
        """
        self.root_dir = root_dir
        self.transform = transform
        self.samples = []
        self.class_to_idx = {}

        # Read class directories and assign labels
        class_folders = sorted(os.listdir(root_dir))
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(class_folders)}

        # Load .npy files and their labels
        for cls_name in class_folders:
            class_path = os.path.join(root_dir, cls_name)
            if not os.path.isdir(class_path):
                continue

            for file in os.listdir(class_path):
                if file.endswith(".npy"):
                    file_path = os.path.join(class_path, file)
                    self.samples.append((file_path, self.class_to_idx[cls_name]))

        if len(self.samples) == 0:
            raise ValueError(f"No .npy files found in {root_dir}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        file_path, label = self.samples[idx]

        # Load .npy file as a tensor
        image = np.load(file_path).astype(np.float32)  # (101, 201, 6)
        image = torch.tensor(image).permute(2, 0, 1)  # Convert to (C, H, W)

        # Ensure correct shape (6, H, W)
        # if image.ndim != 3:
        #     raise ValueError(f"Expected (6, H, W) shape but got {image.shape} in {file_path}")
        #
        # if image.shape[0] != 6:
        #     raise ValueError(f"Expected 6 channels, but got {image.shape[0]} in {file_path}")

        if self.transform:
            image = self.transform(image)

        return image, label


def get_data_loader(data_dir, dataset_type, batch_size=32):
    """
    Load dataset from the given path using NpyDataset.
    Returns a DataLoader and a class-to-index mapping.

    :param data_dir: Path to the dataset directory (should contain "train", "val", "test")
    :param dataset_type: One of "train", "val", or "test"
    :param batch_size: Batch size for the DataLoader
    :return: DataLoader and class-to-index mapping
    """
    dataset_path = os.path.join(data_dir, dataset_type)

    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset path {dataset_path} does not exist.")

    transform = transforms.Compose([
        transforms.Normalize(mean=[0.5] * 6, std=[0.5] * 6)  # Normalize all 6 channels independently
    ])

    dataset = NpyDataset(root_dir=dataset_path, transform=transform)
    shuffle = dataset_type == "train"  # Only shuffle training data

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=4)

    return loader, dataset.class_to_idx  # Return DataLoader and class mapping


if __name__ == "__main__":
    data_root = "/path/to/organized_pileups_dataset_6channels"  # Update this path

    # Load training data
    train_loader, class_map = get_data_loader(data_root, dataset_type="train", batch_size=16)
    print(f"Loaded {len(train_loader.dataset)} training samples.")

    # Load validation data
    val_loader, _ = get_data_loader(data_root, dataset_type="val", batch_size=16)
    print(f"Loaded {len(val_loader.dataset)} validation samples.")

    # Load test data
    test_loader, _ = get_data_loader(data_root, dataset_type="test", batch_size=16)
    print(f"Loaded {len(test_loader.dataset)} test samples.")

    print(f"Class-to-index mapping: {class_map}")
