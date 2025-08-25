import os
import json
import shutil
import argparse
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm


def restore_single_node(node_id, node_data, tensors_source_path, restore_base_path):
    """
    Restores a single node's directory structure from the organized data.
    This function is designed to be run in a separate thread.

    Args:
        node_id (str): The node ID to process.
        node_data (dict): The JSON data corresponding to this node.
        tensors_source_path (Path): The path to the 'tensors' directory.
        restore_base_path (Path): The base path where the restored node directory will be created.

    Returns:
        int: The number of .npy files successfully moved for this node.
    """
    npy_files_moved = 0
    node_restore_path = restore_base_path / node_id
    node_restore_path.mkdir(exist_ok=True)

    # --- 1. Write the individual variant_summary.json ---
    summary_file_path = node_restore_path / "variant_summary.json"
    try:
        with open(summary_file_path, 'w') as f:
            json.dump(node_data, f, indent=2)
    except Exception as e:
        raise IOError(f"Failed to write JSON for node {node_id}: {e}")

    # --- 2. Move and rename .npy files back ---
    variants = node_data.get("variants_passing_af_filter", [])
    for variant_info in variants:
        original_npy_filename = variant_info.get("tensor_file")
        if not original_npy_filename:
            continue

        prefixed_filename = f"{node_id}_{original_npy_filename}"
        source_npy_path = tensors_source_path / prefixed_filename
        destination_npy_path = node_restore_path / original_npy_filename

        if source_npy_path.exists():
            try:
                shutil.move(str(source_npy_path), str(destination_npy_path))
                npy_files_moved += 1
            except Exception as e:
                # Log a warning but continue trying to move other files for this node
                tqdm.write(f"\nWarning: Could not move file '{source_npy_path}': {e}")
        else:
            tqdm.write(f"\nWarning: Source tensor file not found: '{source_npy_path}'")

    return npy_files_moved


def restore_organized_data(source_dir, output_dir, num_workers):
    """
    Restores the original directory structure from the organized data folder in parallel.

    Args:
        source_dir (str): The organized directory containing 'tensors' and 'merged_variant_summary.json'.
        output_dir (str): The directory where the original structure will be restored.
        num_workers (int): The number of parallel threads to use.
    """
    # --- 1. Validate Paths and Load Merged JSON ---
    source_path = Path(source_dir)
    output_path = Path(output_dir)
    tensors_source_path = source_path / "tensors"
    merged_json_path = source_path / "merged_variant_summary.json"

    if not all([source_path.is_dir(), tensors_source_path.is_dir(), merged_json_path.is_file()]):
        print(f"Error: Source directory '{source_dir}' is not valid.")
        print("It must contain a 'tensors' subdirectory and a 'merged_variant_summary.json' file.")
        return

    output_path.mkdir(exist_ok=True)
    print(f"Restored data will be saved in: '{output_path}'")

    try:
        with open(merged_json_path, 'r') as f:
            all_node_data = json.load(f)
    except (json.JSONDecodeError, Exception) as e:
        print(f"Error: Could not read or parse '{merged_json_path}': {e}")
        return

    # --- 2. Process Data in Parallel ---
    total_npy_files_moved = 0
    print(f"\nFound {len(all_node_data)} nodes to restore with {num_workers} workers...")

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        future_to_node = {
            executor.submit(restore_single_node, node_id, data, tensors_source_path, output_path): node_id
            for node_id, data in all_node_data.items()
        }

        for future in tqdm(as_completed(future_to_node), total=len(all_node_data), desc="Restoring nodes"):
            node_id_str = future_to_node[future]
            try:
                npy_count = future.result()
                total_npy_files_moved += npy_count
            except Exception as exc:
                tqdm.write(f"\nError restoring node {node_id_str}: {exc}")

    print("\n--- Restoration Complete ---")
    print(f"Total .npy files moved back: {total_npy_files_moved}")
    print(f"Total node directories created: {len(all_node_data)}")
    print(f"Restored data is located in: '{output_path}'")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Restore original node directory structure from an organized dataset.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("source_directory", help="The organized source directory (contains 'tensors' and merged JSON).")
    parser.add_argument("output_directory", help="The directory where the original structure will be restored.")
    parser.add_argument("--workers", type=int, default=os.cpu_count(),
                        help="Number of parallel worker threads to use.")

    args = parser.parse_args()

    restore_organized_data(args.source_directory, args.output_directory, args.workers)
