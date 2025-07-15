import os
import json
import argparse


def run_full_evaluation(json_path, data_folder_path, comparison_folder_path=None):
    """
    Calculates full classification metrics, lists all item names, and optionally
    compares false positives with an additional folder.

    Args:
        json_path (str): File path for the JSON with prediction lists.
        data_folder_path (str): File path for the root data folder ('val/true' and 'val/false').
        comparison_folder_path (str, optional): Path to a folder to compare
                                                false positives against. Defaults to None.
    """
    # --- 1. Load Model Predictions from JSON ---
    try:
        with open(json_path, 'r') as f:
            predictions = json.load(f)
        predicted_true = set(predictions.get('true', []))
        predicted_false = set(predictions.get('false', []))
    except FileNotFoundError:
        print(f"Error: The file '{json_path}' was not found.")
        return
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from '{json_path}'.")
        return

    # --- 2. Get Ground Truth Files ---
    actual_true_dir = os.path.join(data_folder_path, 'val', 'true')
    actual_false_dir = os.path.join(data_folder_path, 'val', 'false')
    if not os.path.isdir(actual_true_dir) or not os.path.isdir(actual_false_dir):
        print(f"Error: Ensure both '{actual_true_dir}' and '{actual_false_dir}' exist.")
        return
    actual_true = {os.path.basename(f) for f in os.listdir(actual_true_dir)}
    actual_false = {os.path.basename(f) for f in os.listdir(actual_false_dir)}

    # --- 3. Identify TP, TN, FP, FN ---
    true_positives = predicted_true.intersection(actual_true)
    true_negatives = predicted_false.intersection(actual_false)
    false_positives = predicted_true.intersection(actual_false)
    false_negatives = predicted_false.intersection(actual_true)
    tp_count, tn_count = len(true_positives), len(true_negatives)
    fp_count, fn_count = len(false_positives), len(false_negatives)

    # --- 4. Calculate Performance Metrics ---
    precision = tp_count / (tp_count + fp_count) if (tp_count + fp_count) > 0 else 0
    recall = tp_count / (tp_count + fn_count) if (tp_count + fn_count) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    # --- 5. Print Core Results ---
    print("=" * 60)
    print("CLASSIFICATION METRICS & COMPLETE FILE LISTS")
    print("=" * 60)
    print(
        f"\n📊 Performance Metrics:\n  - Precision: {precision:.4f}\n  - Recall:    {recall:.4f}\n  - F1 Score:  {f1_score:.4f}")
    print(
        f"\n📋 Confusion Matrix Counts:\n  - True Positives (TP):  {tp_count}\n  - True Negatives (TN):  {tn_count}\n  - False Positives (FP): {fp_count}\n  - False Negatives (FN): {fn_count}")

    # --- 6. (OPTIONAL) Compare False Positives with another folder ---
    if comparison_folder_path:
        print("\n" + "=" * 60)
        print("FALSE POSITIVE COMPARISON")
        print("=" * 60)
        if not os.path.isdir(comparison_folder_path):
            print(f"\nWarning: Comparison folder not found at '{comparison_folder_path}'")
        else:
            comparison_files = {os.path.basename(f) for f in os.listdir(comparison_folder_path)}
            overlapped_items = false_positives.intersection(comparison_files)
            overlap_rate = len(overlapped_items) / fp_count if fp_count > 0 else 0

            print(
                f"\nComparing {fp_count} false positives against {len(comparison_files)} items in '{os.path.basename(comparison_folder_path)}'.")
            print(f"📈 Overlap Rate: {overlap_rate:.2%}")

            print(f"\n--- Overlapping Items ({len(overlapped_items)} found) ---")
            if overlapped_items:
                for item in sorted(list(overlapped_items)):
                    print(f"  {item}")
            else:
                print("  None")

    # --- 7. Print Detailed File Lists ---
    def print_full_file_list(name, file_set):
        print(f"\n--- {name} ({len(file_set)} files) ---")
        if file_set:
            for filename in sorted(list(file_set)):
                print(f"  {filename}")
        else:
            print("  None")

    print("\n" + "=" * 60)
    print("📂 Detailed File Lists")
    print("=" * 60)
    print_full_file_list("✅ True Positives", true_positives)
    print_full_file_list("✅ True Negatives", true_negatives)
    print_full_file_list("❌ False Positives (Type I Error)", false_positives)
    print_full_file_list("❌ False Negatives (Type II Error)", false_negatives)
    print("\n" + "=" * 60)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run a full model evaluation.")
    parser.add_argument("json_file", help="Path to the JSON file with prediction results.")
    parser.add_argument("data_folder", help="Path to the root data folder.")
    parser.add_argument("--compare_folder",
                        help="Optional: Path to a second folder to compare False Positives against.")

    args = parser.parse_args()
    run_full_evaluation(args.json_file, args.data_folder, args.compare_folder)