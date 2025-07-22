#!/usr/bin/env python3
import argparse
import torch
import sys
import os
import json
from tqdm import tqdm
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

# Add the parent directory to the path to find custom modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from mynet import ConvNeXtCBAMClassifier
from dataset_pansoma_npy_6ch import get_data_loader

# Set the device for computation
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def print_and_log(message, log_path):
    """Prints a message to the console and appends it to a log file."""
    print(message)
    with open(log_path, 'a', encoding='utf-8') as f:
        f.write(message + '\n')


def run_evaluation(model, data_loader, genotype_map, output_path):
    """
    Evaluates the model on the provided data loader and saves the results.

    Args:
        model (torch.nn.Module): The trained model to evaluate.
        data_loader (torch.utils.data.DataLoader): DataLoader for the test set.
        genotype_map (dict): A dictionary mapping class names to integer labels.
        output_path (str): Directory to save the log and results files.
    """
    model.eval()  # Set the model to evaluation mode
    all_labels = []
    all_preds = []
    all_paths = []
    all_probs = []

    log_file = os.path.join(output_path, "test_log.txt")
    print_and_log(f"Starting evaluation on device: {device}", log_file)

    with torch.no_grad():  # Disable gradient calculation
        progress_bar = tqdm(data_loader, desc="Evaluating", leave=True)
        for images, labels, paths in progress_bar:
            images, labels = images.to(device), labels.to(device).float()

            # Forward pass
            outputs = model(images)
            if isinstance(outputs, tuple):  # Handle models with auxiliary outputs
                outputs = outputs[0]

            outputs = outputs.squeeze(1)

            # Get probabilities (scores) and thresholded predictions
            probs = torch.sigmoid(outputs)
            preds = (probs > 0.5).long()

            # Store results for later analysis
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            all_paths.extend(paths)

    # --- Metrics Calculation (for logging) ---
    all_labels = np.array(all_labels).astype(int)
    all_preds = np.array(all_preds)
    class_names = sorted(genotype_map.keys(), key=lambda k: genotype_map[k])
    report = classification_report(all_labels, all_preds, target_names=class_names, digits=4)
    cm = confusion_matrix(all_labels, all_preds)

    # --- Logging Summary ---
    print_and_log("\n" + "=" * 30 + " EVALUATION SUMMARY " + "=" * 30, log_file)
    print_and_log(f"\nTotal samples evaluated: {len(all_labels)}", log_file)
    print_and_log("\nClassification Report:", log_file)
    print_and_log(report, log_file)
    print_and_log("\nConfusion Matrix:", log_file)
    header = " " * 5 + " ".join([f"{name:<10}" for name in class_names])
    print_and_log(header, log_file)
    print_and_log("-" * len(header), log_file)
    for i, row in enumerate(cm):
        row_str = f"{class_names[i]:<5}" + " ".join([f"{val:<10}" for val in row])
        print_and_log(row_str, log_file)
    print_and_log("=" * 80, log_file)

    # --- MODIFIED: Save predictions in the requested JSON format ---
    predictions_data = []
    for i in range(len(all_paths)):
        predicted_class_name = class_names[all_preds[i]]
        # The raw sigmoid probability is used as the confidence score
        confidence_score = float(all_probs[i])

        predictions_data.append({
            "file_name": os.path.basename(all_paths[i]),
            "predicted_type": predicted_class_name,
            "score": round(confidence_score, 6)  # Round for cleanliness
        })

    results_path = os.path.join(output_path, "predictions.json")
    with open(results_path, 'w') as f:
        json.dump(predictions_data, f, indent=4)
    print_and_log(f"\n✅ Predictions saved in DeepVariant format to: {results_path}", log_file)


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate a trained classifier and produce a DeepVariant-style JSON output.")
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to the trained model checkpoint (.pth file).")
    parser.add_argument("--data_path", type=str, required=True,
                        help="Path to the dataset directory (must contain a 'test' subdirectory).")
    parser.add_argument("--output_path", type=str, required=True,
                        help="Directory to save the evaluation results and log file.")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size for evaluation.")
    parser.add_argument("--num_workers", type=int, default=4,
                        help="Number of worker processes for data loading.")
    args = parser.parse_args()

    # --- Setup ---
    os.makedirs(args.output_path, exist_ok=True)
    log_file = os.path.join(args.output_path, "test_log.txt")
    if os.path.exists(log_file):
        os.remove(log_file)  # Clear old log file

    # --- Load Model ---
    if not os.path.isfile(args.model_path):
        print_and_log(f"Error: Model file not found at {args.model_path}", log_file)
        return

    print_and_log(f"Loading model from: {args.model_path}", log_file)
    checkpoint = torch.load(args.model_path, map_location=device)

    try:
        genotype_map = checkpoint['genotype_map']
        in_channels = checkpoint.get('in_channels', 6)
        model = ConvNeXtCBAMClassifier(in_channels=in_channels, class_num=1, depths=[3, 3, 27, 3],
                                       dims=[128, 256, 512, 1024]).to(device)
        model.load_state_dict(checkpoint['model_state_dict'])

    except KeyError as e:
        print_and_log(f"Error: Checkpoint is missing a required key: {e}. Cannot build model.", log_file)
        return
    except Exception as e:
        print_and_log(f"An error occurred while loading the model: {e}", log_file)
        return

    # --- Load Data ---
    print_and_log(f"Loading test data from: {args.data_path}", log_file)
    try:
        test_loader, _ = get_data_loader(
            data_dir=args.data_path,
            dataset_type="test",
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            shuffle=False,
            return_paths=True
        )
    except Exception as e:
        print_and_log(f"FATAL: Could not create test data loader. Error: {e}", log_file)
        return

    # --- Run Evaluation ---
    run_evaluation(model, test_loader, genotype_map, args.output_path)


if __name__ == "__main__":
    main()