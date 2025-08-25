import os
import json
import shutil
import argparse
from pathlib import Path


def organize_node_data(source_dir, output_dir):
    """
    Organizes node data by moving and renaming .npy files and merging
    variant_summary.json files.

    Args:
        source_dir (str): The path to the directory containing node_id subdirectories.
        output_dir (str): The path to the directory where organized data will be saved.
    """
    # --- 1. Setup Output Directories ---
    source_path = Path(source_dir)
    output_path = Path(output_dir)
    tensors_path = output_path / "tensors"

    if not source_path.is_dir():
        print(f"Error: Source directory not found at '{source_dir}'")
        return

    # Create output directories if they don't exist
    output_path.mkdir(exist_ok=True)
    tensors_path.mkdir(exist_ok=True)
    print(f"Output will be saved in: '{output_path}'")
    print(f"Tensors will be moved to: '{tensors_path}'")

    # --- 2. Process Data and Merge JSONs ---
    merged_summary_data = {}
    total_npy_files_moved = 0

    # Get a list of all subdirectories that seem to be node_ids (i.e., are numbers)
    node_dirs = [d for d in source_path.iterdir() if d.is_dir() and d.name.isdigit()]

    if not node_dirs:
        print(f"Warning: No node_id subdirectories found in '{source_dir}'")
        return

    print(f"\nFound {len(node_dirs)} node directories to process...")

    for i, node_dir in enumerate(node_dirs):
        node_id = node_dir.name
        print(f"  ({i + 1}/{len(node_dirs)}) Processing node: {node_id}")

        # --- Move and rename .npy files ---
        npy_files = list(node_dir.glob("*.npy"))
        for npy_file in npy_files:
            # Create the new filename with the node_id as a prefix
            new_filename = f"{node_id}_{npy_file.name}"
            destination_path = tensors_path / new_filename

            # Move the file
            shutil.move(str(npy_file), str(destination_path))
            total_npy_files_moved += 1

        # --- Read and merge the variant_summary.json ---
        summary_file = node_dir / "variant_summary.json"
        if summary_file.exists():
            try:
                with open(summary_file, 'r') as f:
                    data = json.load(f)
                    # Use the node_id as the key for the merged dictionary
                    merged_summary_data[node_id] = data
            except json.JSONDecodeError:
                print(f"    Warning: Could not parse JSON file at '{summary_file}'")
            except Exception as e:
                print(f"    Warning: An error occurred reading '{summary_file}': {e}")
        else:
            print(f"    Warning: 'variant_summary.json' not found in node {node_id}")

    # --- 3. Write the Final Merged JSON ---
    output_json_path = output_path / "merged_variant_summary.json"
    try:
        with open(output_json_path, 'w') as f:
            json.dump(merged_summary_data, f, indent=2)
    except Exception as e:
        print(f"\nError: Failed to write merged JSON file: {e}")
        return

    print("\n--- Organization Complete ---")
    print(f"Total .npy files moved: {total_npy_files_moved}")
    print(f"Total nodes merged in JSON: {len(merged_summary_data)}")
    print(f"Merged summary saved to: '{output_json_path}'")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Organize node data by moving .npy files and merging summary JSONs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("source_directory", help="The source directory containing the node_id subfolders.")
    parser.add_argument("output_directory", help="The directory where the organized data will be saved.")

    args = parser.parse_args()

    organize_node_data(args.source_directory, args.output_directory)
