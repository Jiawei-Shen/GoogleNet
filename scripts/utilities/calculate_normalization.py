import os
import torch
import numpy as np
from tqdm import tqdm
import argparse # --- ADDED: Import argparse

def calculate_mean_std(root_dir: str, num_channels: int):
    """
    Calculates the mean and standard deviation of a dataset of .npy files.

    Args:
        root_dir (str): The root directory of the training dataset.
        num_channels (int): The number of channels in the image files.
    """
    # Find all .npy file paths
    filepaths = []
    for dirpath, _, filenames in os.walk(root_dir):
        for f in filenames:
            if f.lower().endswith('.npy'):
                filepaths.append(os.path.join(dirpath, f))

    if not filepaths:
        raise ValueError(f"No .npy files found in directory: {root_dir}")

    print(f"Found {len(filepaths)} .npy files. Starting calculation for {num_channels} channels...")

    # Use PyTorch for efficient computation
    sum_channels = torch.zeros(num_channels) # --- MODIFIED: Use num_channels
    sum_sq_channels = torch.zeros(num_channels) # --- MODIFIED: Use num_channels
    pixel_count = 0

    # Iterate through all files with a progress bar
    for path in tqdm(filepaths, desc="Calculating Stats"):
        image_np = np.load(path)

        # Ensure it's in the correct format (C, W, H)
        if image_np.ndim != 3 or image_np.shape[0] != num_channels: # --- MODIFIED: Use num_channels
            print(f"\nSkipping file with incorrect shape: {path} has shape {image_np.shape}")
            continue

        image_tensor = torch.from_numpy(image_np.astype(np.float32))

        # Sum over the width and height dimensions (1 and 2)
        sum_channels += torch.sum(image_tensor, dim=[1, 2])
        sum_sq_channels += torch.sum(image_tensor ** 2, dim=[1, 2])
        pixel_count += image_tensor.shape[1] * image_tensor.shape[2]

    if pixel_count == 0:
        raise ValueError("Could not process any images. Check file shapes and paths.")

    # Calculate mean and standard deviation
    mean = sum_channels / pixel_count
    var = (sum_sq_channels / pixel_count) - (mean ** 2)
    std = torch.sqrt(var)

    return mean.tolist(), std.tolist()


if __name__ == "__main__":
    # --- ADDED: Set up argument parser ---
    parser = argparse.ArgumentParser(description="Calculate mean and std for a dataset of .npy files.")
    parser.add_argument("data_dir", type=str, help="Path to the root directory of the training dataset.")
    parser.add_argument("--channels", type=int, default=6, help="Number of channels in the image files (default: 6).")
    args = parser.parse_args()
    # -------------------------------------

    try:
        # --- MODIFIED: Call function with parsed arguments ---
        mean, std = calculate_mean_std(args.data_dir, args.channels)

        print("\n" + "="*40)
        print("✅ Calculation Complete!")
        print("="*40)
        print("\nCopy these values into your dataloader script:\n")
        print(f"mean = {mean}")
        print(f"std  = {std}")
        print("\n" + "="*40)

    except (ValueError, FileNotFoundError) as e:
        print(f"\nError: {e}")