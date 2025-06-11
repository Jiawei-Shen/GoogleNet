import os
import numpy as np
import torch # Using torch for efficient tensor operations
from tqdm import tqdm # For a progress bar

def calculate_channel_stats(data_root_dir):
    """
    Calculates the mean and standard deviation for each channel across all
    .npy files in the 'train' subdirectory of the given data_root_dir.

    Args:
        data_root_dir (str): The root directory of the dataset.
                             This directory should contain a 'train' subdirectory,
                             which in turn contains class-specific subdirectories
                             with .npy files.

    Returns:
        tuple: A tuple containing two lists: (means, stds).
               'means' is a list of 4 float values (mean for each channel).
               'stds' is a list of 4 float values (std dev for each channel).
               Returns (None, None) if no .npy files are found.
    """
    train_dir = os.path.join(data_root_dir, "train")

    if not os.path.isdir(train_dir):
        print(f"Error: 'train' directory not found in {data_root_dir}")
        return None, None

    npy_files_paths = []
    for root, _, files in os.walk(train_dir):
        for file in files:
            if file.lower().endswith(".npy"):
                npy_files_paths.append(os.path.join(root, file))

    if not npy_files_paths:
        print(f"Error: No .npy files found in {train_dir}")
        return None, None

    print(f"Found {len(npy_files_paths)} .npy files in {train_dir} to process.")

    # Initialize accumulators for sums and sums of squares for 4 channels
    # Using torch tensors for potentially faster accumulation if on GPU,
    # but will work on CPU too.
    channel_sum = torch.zeros(4, dtype=torch.float64) # Sum of pixel values
    channel_sum_sq = torch.zeros(4, dtype=torch.float64) # Sum of squared pixel values
    pixel_count_per_channel = torch.zeros(4, dtype=torch.int64) # Total number of pixels per channel

    # Expected shape (can be dynamic if needed, but fixed for this example)
    # If your H and W vary, this needs adjustment or per-file calculation.
    # For now, let's assume they are somewhat consistent or we load one to check.
    # We can also dynamically sum pixels.

    for file_path in tqdm(npy_files_paths, desc="Processing .npy files"):
        try:
            # Load the .npy file
            data_np = np.load(file_path)

            # Ensure it's a numpy array
            if not isinstance(data_np, np.ndarray):
                print(f"Warning: Skipping {file_path}, not a NumPy array.")
                continue

            # Validate shape (Channels, Height, Width)
            if data_np.ndim != 3 or data_np.shape[0] != 4:
                print(f"Warning: Skipping {file_path}, unexpected shape {data_np.shape}. Expected (4, H, W).")
                continue

            # Convert to a float PyTorch tensor
            data_tensor = torch.from_numpy(data_np.astype(np.float64)) # Use float64 for precision

            # Accumulate sums and sums of squares per channel
            # Sum over Height and Width dimensions (dim 1 and 2 for a C, H, W tensor)
            channel_sum += torch.sum(data_tensor, dim=[1, 2])
            channel_sum_sq += torch.sum(data_tensor**2, dim=[1, 2])

            # Accumulate pixel counts for each channel
            # data_tensor.shape[1] is Height, data_tensor.shape[2] is Width
            num_pixels_in_this_file_per_channel = data_tensor.shape[1] * data_tensor.shape[2]
            pixel_count_per_channel += num_pixels_in_this_file_per_channel

        except Exception as e:
            print(f"Warning: Could not process file {file_path}. Error: {e}")
            continue

    if torch.any(pixel_count_per_channel == 0):
        print("Error: Zero pixels counted for one or more channels. Cannot calculate stats.")
        # Identify which channels had zero pixels
        for i in range(4):
            if pixel_count_per_channel[i] == 0:
                print(f"  Channel {i+1} had 0 pixels.")
        return None, None

    # Calculate mean for each channel
    means = channel_sum / pixel_count_per_channel.double() # Ensure double for division

    # Calculate variance for each channel: Var(X) = E[X^2] - (E[X])^2
    variances = (channel_sum_sq / pixel_count_per_channel.double()) - (means**2)

    # Handle potential negative variances due to floating point inaccuracies if variance is very close to 0
    variances[variances < 0] = 0

    # Calculate standard deviation for each channel
    stds = torch.sqrt(variances)

    return means.tolist(), stds.tolist()

if __name__ == "__main__":
    # IMPORTANT: Replace this with the actual path to your dataset's root directory
    # This is the directory that contains the 'train' and 'val' subdirectories.
    # Example: "/home/jiawei/Documents/Dockers/GoogleNet/data/COLO829T_SNV_chr1_tensors_4channel"
    # dataset_root = input("Enter the root path to your 4-channel dataset directory: ")
    dataset_root = "/home/jiawei/Documents/Dockers/GoogleNet/data/COLO829T_SNV_chr1_tensors_4channel"

    if not os.path.isdir(dataset_root):
        print(f"Error: The provided path '{dataset_root}' is not a valid directory.")
    else:
        print(f"\nCalculating normalization statistics for dataset at: {dataset_root}")
        calculated_means, calculated_stds = calculate_channel_stats(dataset_root)

        if calculated_means and calculated_stds:
            print("\n--- Calculated Statistics ---")
            print(f"Means per channel: {calculated_means}")
            print(f"Std Devs per channel: {calculated_stds}")
            print("\nReminder: Use these values in your transforms.Normalize() step for your DataLoader.")
            print("Example:")
            print(f"transforms.Normalize(mean={calculated_means}, std={calculated_stds})")
        else:
            print("\nCould not calculate statistics. Please check errors above.")

