import os
import shutil
import random
import json


def organize_dataset(input_dir, output_dir, train_ratio=0.7, val_ratio=0.15, include_png=False):
    """
    Organizes a dataset of .npy (and optionally .png) files into train/val/test folders.

    :param input_dir: Directory containing the dataset files.
    :param output_dir: Directory where the organized dataset will be saved.
    :param train_ratio: Proportion of data to use for training (default: 0.7).
    :param val_ratio: Proportion of data to use for validation (default: 0.15).
    :param include_png: Whether to include corresponding .png files (default: False).
    """
    # Ensure sum of ratios is not greater than 1
    test_ratio = 1.0 - (train_ratio + val_ratio)
    assert test_ratio >= 0, "Invalid split ratios! Ensure train + val <= 1.0"

    # Create directories for train, val, and test
    train_dir = os.path.join(output_dir, "train")
    val_dir = os.path.join(output_dir, "val")
    test_dir = os.path.join(output_dir, "test")
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    # Get all .npy files
    files = [f for f in os.listdir(input_dir) if f.endswith(".npy")]

    # Parse genotype labels from filenames and group them
    genotype_samples = {}
    for file in files:
        parts = file.split("_")
        genotype = parts[-1].replace(".npy", "")  # Extract GT info (e.g., GT0|1)

        if genotype not in genotype_samples:
            genotype_samples[genotype] = []

        genotype_samples[genotype].append(file)

    # Split data into train, val, and test
    metadata = {"train": {}, "val": {}, "test": {}}

    for genotype, samples in genotype_samples.items():
        random.shuffle(samples)
        total_samples = len(samples)
        train_idx = int(total_samples * train_ratio)
        val_idx = train_idx + int(total_samples * val_ratio)

        train_samples = samples[:train_idx]
        val_samples = samples[train_idx:val_idx]
        test_samples = samples[val_idx:]

        # Create directories for each genotype
        os.makedirs(os.path.join(train_dir, genotype), exist_ok=True)
        os.makedirs(os.path.join(val_dir, genotype), exist_ok=True)
        os.makedirs(os.path.join(test_dir, genotype), exist_ok=True)

        # Move .npy files
        for file in train_samples:
            shutil.copy(os.path.join(input_dir, file), os.path.join(train_dir, genotype, file))
            if include_png:
                png_file = file.replace(".npy", ".png")
                if os.path.exists(os.path.join(input_dir, png_file)):
                    shutil.copy(os.path.join(input_dir, png_file), os.path.join(train_dir, genotype, png_file))

        for file in val_samples:
            shutil.copy(os.path.join(input_dir, file), os.path.join(val_dir, genotype, file))
            if include_png:
                png_file = file.replace(".npy", ".png")
                if os.path.exists(os.path.join(input_dir, png_file)):
                    shutil.copy(os.path.join(input_dir, png_file), os.path.join(val_dir, genotype, png_file))

        for file in test_samples:
            shutil.copy(os.path.join(input_dir, file), os.path.join(test_dir, genotype, file))
            if include_png:
                png_file = file.replace(".npy", ".png")
                if os.path.exists(os.path.join(input_dir, png_file)):
                    shutil.copy(os.path.join(input_dir, png_file), os.path.join(test_dir, genotype, png_file))

        # Store metadata
        metadata["train"][genotype] = train_samples
        metadata["val"][genotype] = val_samples
        metadata["test"][genotype] = test_samples

    # Save metadata.json
    with open(os.path.join(output_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=4)

    print("Dataset organized successfully!")


if __name__ == "__main__":
    input_dir = "/home/jiawei/PycharmProjects/TensorBuild/output_pileups_6channels"  # Change this to your dataset directory
    output_dir = "/home/jiawei/Documents/Dockers/GoogleNet/data/organized_pileups_dataset_6channels"
    train_ratio = 0.8
    val_ratio = 0.1
    include_png = True  # Set to False if you don’t need .png files

    organize_dataset(input_dir, output_dir, train_ratio, val_ratio, include_png)
