import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import os
import json
import numpy as np

class PileupDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        """
        PyTorch Dataset for loading pileup images and corresponding genotypes.

        :param data_dir: Directory containing the .npy files and metadata.json
        :param transform: Image transformations
        """
        self.data_dir = data_dir
        self.transform = transform
        self.samples = []

        metadata_path = os.path.join(data_dir, "metadata.json")
        if not os.path.exists(metadata_path):
            raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

        with open(metadata_path, "r") as f:
            metadata = json.load(f)

        for entry in metadata:
            img_path = os.path.join(data_dir, entry["npy_path"])
            genotype = entry["genotype"]
            if os.path.exists(img_path):
                self.samples.append((img_path, genotype))

        self.genotype_classes = sorted(set(gt for _, gt in self.samples))
        self.genotype_to_idx = {gt: i for i, gt in enumerate(self.genotype_classes)}

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, genotype = self.samples[idx]
        image = np.load(img_path).astype(np.float32)
        image = torch.tensor(image).permute(2, 0, 1)  # Convert to (C, H, W)

        if self.transform:
            image = self.transform(image)

        label = self.genotype_to_idx[genotype]
        return image, label


def get_data_loader(data_dir, batch_size=32, train=True):
    transform = transforms.Compose([
        transforms.Normalize(mean=[0.5], std=[0.5])  # Normalize for custom images
    ])

    dataset = PileupDataset(data_dir, transform=transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=train, num_workers=4)

    return loader, dataset.genotype_to_idx


if __name__ == "__main__":
    data_dir = "output_pileups"  # Update this to your dataset path
    train_loader, genotype_map = get_data_loader(data_dir, batch_size=32, train=True)
    print(f"Loaded {len(train_loader.dataset)} training samples.")
    print(f"Genotype classes: {genotype_map}")
