import os
import json
import argparse

def calculate_and_list_all(json_path, data_folder_path):
    """
    Calculates full classification metrics and lists ALL item names for
    TP, TN, FP, and FN categories.

    Args:
        json_path (str): File path for the JSON with 'true' and 'false' prediction lists.
        data_folder_path (str): File path for the root data folder, containing
                                'val/true' and 'val/false' subdirectories.
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

    # --- 2. Get Ground Truth Files from Folders ---
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

    tp_count = len(true_positives)
    tn_count = len(true_negatives)
    fp_count = len(false_positives)
    fn_count = len(false_negatives)

    # --- 4. Calculate Performance Metrics ---
    precision = tp_count / (tp_count + fp_count) if (tp_count + fp_count) > 0 else 0
    recall = tp_count / (tp_count + fn_count) if (tp_count + fn_count) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    # --- 5. Print All Results ---
    print("=" * 60)
    print("CLASSIFICATION METRICS & COMPLETE FILE LISTS")
    print("=" * 60)

    # Print Scores
    print("\n📊 Performance Metrics:")
    print(f"  - Precision: {precision:.4f}")
    print(f"  - Recall:    {recall:.4f}")
    print(f"  - F1 Score:  {f1_score:.4f}")

    # Print Counts
    print("\n📋 Confusion Matrix Counts:")
    print(f"  - True Positives (TP):  {tp_count}")
    print(f"  - True Negatives (TN):  {tn_count}")
    print(f"  - False Positives (FP): {fp_count}")
    print(f"  - False Negatives (FN): {fn_count}")

    # Helper function to print a full list of filenames
    def print_full_file_list(name, file_set):
        print(f"\n--- {name} ({len(file_set)} files) ---")
        if file_set:
            for filename in sorted(list(file_set)):
                print(f"  {filename}")
        else:
            print("  None")

    print("\n" + "-" * 60)
    print("📂 Detailed File Lists")
    print("-" * 60)
    print_full_file_list("✅ True Positives", true_positives)
    print_full_file_list("✅ True Negatives", true_negatives)
    print_full_file_list("❌ False Positives (Type I Error)", false_positives)
    print_full_file_list("❌ False Negatives (Type II Error)", false_negatives)
    print("\n" + "=" * 60)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Calculate full classification metrics and list all item names."
    )
    parser.add_argument("json_file", help="Path to the JSON file with prediction results.")
    parser.add_argument("data_folder", help="Path to the root data folder.")

    args = parser.parse_args()
    calculate_and_list_all(args.json_file, args.data_folder)