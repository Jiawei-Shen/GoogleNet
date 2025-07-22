#!/usr/bin/env python3
import argparse
import torch
import torch.optim as optim
import torch.nn as nn
import sys
import os
from tqdm import tqdm
from collections import defaultdict
import torch.nn.functional as F
import json

# --- MODIFIED: Import new schedulers ---
from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from mynet import ConvNeXtCBAMClassifier
from dataset_pansoma_npy_6ch import get_data_loader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class MultiClassFocalLoss(nn.Module):
    """
    Focal Loss for multi-class classification.
    """
    def __init__(self, gamma=2.0, weight=None, reduction='mean'):
        super(MultiClassFocalLoss, self).__init__()
        self.gamma = gamma
        self.weight = weight
        self.reduction = reduction

    def forward(self, logits, targets):
        log_probs = F.log_softmax(logits, dim=1)
        log_pt = log_probs.gather(1, targets.view(-1, 1)).squeeze(1)
        pt = torch.exp(log_pt)

        if self.weight is not None:
            at = self.weight.gather(0, targets)
            log_pt = log_pt * at

        focal_loss = -1 * (1 - pt)**self.gamma * log_pt

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss


class CombinedFocalWeightedCELoss(nn.Module):
    def __init__(self, initial_lr, pos_weight=None, gamma=2.0):
        super().__init__()
        self.initial_lr = initial_lr
        self.focal_loss = MultiClassFocalLoss(gamma=gamma, weight=pos_weight)
        self.wce_loss = nn.CrossEntropyLoss(pos_weight=pos_weight)

    def forward(self, logits, targets, current_lr):
        focal_weight = 1.0 - (current_lr / self.initial_lr)
        wce_weight = 1.0 - focal_weight
        loss_focal = self.focal_loss(logits, targets)
        loss_wce = self.wce_loss(logits, targets)
        return focal_weight * loss_focal + wce_weight * loss_wce


def init_weights(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)


def print_and_log(message, log_path):
    print(message)
    with open(log_path, 'a', encoding='utf-8') as f:
        f.write(message + '\n')


# --- MODIFIED: Updated function signature with new arguments ---
def train_model(data_path, output_path, save_val_results=False, num_epochs=100, learning_rate=0.0001,
                batch_size=32, num_workers=4, model_save_milestone=50, loss_type='weighted_ce',
                warmup_epochs=10, weight_decay=0.05, depths=None, dims=None):
    os.makedirs(output_path, exist_ok=True)
    log_file = os.path.join(output_path, "training_log_6ch.txt")
    if os.path.exists(log_file):
        os.remove(log_file)

    print_and_log(f"Using device: {device}", log_file)
    print_and_log(f"Initial Learning Rate: {learning_rate:.1e}", log_file)
    # --- MODIFIED: Updated log message for new scheduler ---
    print_and_log(
        f"Using Cosine Annealing scheduler with a {warmup_epochs}-epoch linear warmup.",
        log_file)
    print_and_log(f"Checkpoints will be saved every {model_save_milestone} epochs into: {output_path}", log_file)
    print_and_log(f"Using {num_workers} workers for data loading.", log_file)
    if save_val_results:
        print_and_log("Validation classification results will be saved at each milestone.", log_file)

    train_loader, genotype_map = get_data_loader(
        data_dir=data_path, dataset_type="train", batch_size=batch_size,
        num_workers=num_workers, shuffle=True
    )

    try:
        val_loader, _ = get_data_loader(
            data_dir=data_path, dataset_type="val", batch_size=batch_size,
            num_workers=num_workers, shuffle=False, return_paths=True
        )
    except Exception as e:
        print_and_log(f"\nFATAL: Could not create validation data loader with 'return_paths=True'.", log_file)
        print_and_log("Please ensure your 'dataset_pansoma_npy_6ch.py' can handle this flag.", log_file)
        print_and_log(f"Error details: {e}", log_file)
        return

    if not genotype_map:
        print_and_log("Error: genotype_map is empty. Check dataloader.", log_file)
        return
    num_classes = len(genotype_map)
    if num_classes == 0:
        print_and_log("Error: Number of classes is 0. Check dataloader.", log_file)
        return
    print_and_log(f"Number of classes: {num_classes}", log_file)
    sorted_class_names_from_map = sorted(genotype_map.keys(), key=lambda k: genotype_map[k])

    model = ConvNeXtCBAMClassifier(in_channels=6, class_num=num_classes,
                                   depths=depths, dims=dims).to(device)

    model.apply(init_weights)
    false_count = 48736
    true_count = 268
    pos_weight_value = min(88.0, false_count / true_count)
    # Convert the float to a tensor and move it to the correct device
    pos_weight_tensor = torch.tensor(pos_weight_value).to(device)

    if loss_type == "combined":
        criterion = CombinedFocalWeightedCELoss(initial_lr=learning_rate, pos_weight=pos_weight_tensor)
        print_and_log(f"Using Combined Focal Loss and Weighted CE Loss.", log_file)
    elif loss_type == "weighted_ce":
        weight = torch.tensor([1.0, pos_weight_value]).to(device)
        criterion = nn.CrossEntropyLoss(weight=weight)
        print_and_log(f"Using Weighted CE Loss.", log_file)
    else:
        raise ValueError(f"Unsupported loss_type: {loss_type}")

    # --- MODIFIED: Use AdamW optimizer ---
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    print_and_log(f"Using AdamW optimizer with weight decay: {weight_decay}", log_file)

    # --- MODIFIED: Set up schedulers for warmup and cosine annealing ---
    warmup_scheduler = LinearLR(optimizer, start_factor=0.01, total_iters=warmup_epochs)
    main_scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs - warmup_epochs, eta_min=0)
    scheduler = SequentialLR(optimizer, schedulers=[warmup_scheduler, main_scheduler], milestones=[warmup_epochs])

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0

        current_lr = optimizer.param_groups[0]['lr']
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs} LR: {current_lr:.1e}", leave=True)

        batch_count = 0
        for images, labels in progress_bar:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)

            # AFTER
            # Check if using the special combined loss that needs current_lr
            if loss_type == "combined":
                if isinstance(outputs, tuple) and len(outputs) == 3:
                    main_output, aux1, aux2 = outputs
                    loss1 = criterion(main_output, labels, current_lr)
                    loss2 = criterion(aux1, labels, current_lr)
                    loss3 = criterion(aux2, labels, current_lr)
                    loss = loss1 + 0.3 * loss2 + 0.3 * loss3
                    outputs_for_acc = main_output
                elif isinstance(outputs, torch.Tensor):
                    loss = criterion(outputs, labels, current_lr)
                    outputs_for_acc = outputs
                else:
                    progress_bar.close()
                    raise TypeError(f"Model output type not recognized: {type(outputs)}")
            else:  # For other standard losses
                if isinstance(outputs, tuple) and len(outputs) == 3:
                    main_output, aux1, aux2 = outputs
                    loss1 = criterion(main_output, labels)
                    loss2 = criterion(aux1, labels)
                    loss3 = criterion(aux2, labels)
                    loss = loss1 + 0.3 * loss2 + 0.3 * loss3
                    outputs_for_acc = main_output
                elif isinstance(outputs, torch.Tensor):
                    loss = criterion(outputs, labels)
                    outputs_for_acc = outputs
                else:
                    progress_bar.close()
                    raise TypeError(f"Model output type not recognized: {type(outputs)}")

            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            batch_count += 1
            _, predicted = torch.max(outputs_for_acc, 1)
            correct_train += (predicted == labels).sum().item()
            total_train += labels.size(0)

            if total_train > 0 and batch_count > 0:
                avg_loss_train = running_loss / batch_count
                avg_acc_train = (correct_train / total_train) * 100
                progress_bar.set_postfix(loss=f"{avg_loss_train:.4f}", acc=f"{avg_acc_train:.2f}%")

        epoch_train_loss = (running_loss / batch_count) if batch_count > 0 else 0.0
        epoch_train_acc = (correct_train / total_train) * 100 if total_train > 0 else 0.0

        val_loss, val_acc, class_performance_stats_val, val_inference_results = evaluate_model(
            model, val_loader, criterion, genotype_map, log_file, loss_type, current_lr
        )

        if class_performance_stats_val:
            print_and_log("\nClass-wise Validation Accuracy:", log_file)
            for class_name in sorted_class_names_from_map:
                stats = class_performance_stats_val.get(class_name, {})
                print_and_log(
                    f"  {class_name} (Index {stats.get('idx', 'N/A')}): {stats.get('acc', 0):.2f}% ({stats.get('correct', 0)}/{stats.get('total', 0)})",
                    log_file)

        summary_msg = (
            f"Epoch {epoch + 1}/{num_epochs} Summary - Train Loss: {epoch_train_loss:.4f}, Train Acc: {epoch_train_acc:.2f}%, "
            f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}% (LR: {current_lr:.1e})")
        print_and_log(summary_msg, log_file)

        scheduler.step()

        if (epoch + 1) % model_save_milestone == 0 or (epoch + 1) == num_epochs:
            milestone_path = os.path.join(output_path, f"model_epoch_{epoch + 1}.pth")
            torch.save({
                'epoch': epoch + 1, 'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(), 'scheduler_state_dict': scheduler.state_dict(),
                'genotype_map': genotype_map, 'in_channels': 6
            }, milestone_path)
            print_and_log(f"\nMilestone model saved at: {milestone_path}", log_file)

            if save_val_results:
                result_path = os.path.join(output_path, f"validation_results_epoch_{epoch + 1}.json")
                try:
                    with open(result_path, 'w') as f:
                        json.dump(val_inference_results, f, indent=4)
                    print_and_log(f"Saved validation results for epoch {epoch + 1} to {result_path}", log_file)
                except Exception as e:
                    print_and_log(f"Error saving validation results: {e}", log_file)

        print_and_log("-" * 30, log_file)

    print_and_log(f"Training complete. Final model located in: {output_path}", log_file)


def evaluate_model(model, data_loader, criterion, genotype_map, log_file, loss_type, current_lr):
    model.eval()
    running_loss_eval = 0.0
    correct_eval = 0
    total_eval = 0
    class_correct_counts = defaultdict(int)
    class_total_counts = defaultdict(int)
    batch_count_eval = 0

    inference_results = defaultdict(list)
    idx_to_class = {v: k for k, v in genotype_map.items()}

    if not data_loader or len(data_loader) == 0:
        return 0.0, 0.0, {}, {}

    with torch.no_grad():
        for images, labels, paths in data_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            if isinstance(outputs, tuple):
                outputs = outputs[0]

            # loss = criterion(outputs, labels)
            # Update the loss calculation here
            if loss_type == "combined":
                loss = criterion(outputs, labels, current_lr)
            else:
                loss = criterion(outputs, labels)

            running_loss_eval += loss.item()
            batch_count_eval += 1
            _, predicted = torch.max(outputs, 1)
            correct_eval += (predicted == labels).sum().item()
            total_eval += labels.size(0)

            for i, pred_idx_tensor in enumerate(predicted):
                pred_idx = pred_idx_tensor.item()
                true_idx = labels[i].item()
                path = paths[i]

                class_total_counts[true_idx] += 1
                if pred_idx == true_idx:
                    class_correct_counts[true_idx] += 1

                predicted_class_name = idx_to_class[pred_idx]
                inference_results[predicted_class_name].append(os.path.basename(path))

    avg_loss_eval = (running_loss_eval / batch_count_eval) if batch_count_eval > 0 else 0.0
    overall_accuracy_eval = (correct_eval / total_eval) * 100 if total_eval > 0 else 0.0

    class_performance_stats = {}
    if genotype_map:
        for class_name, class_idx in genotype_map.items():
            correct_c = class_correct_counts[class_idx]
            total_c = class_total_counts[class_idx]
            acc_c = (correct_c / total_c) * 100 if total_c > 0 else 0.0
            class_performance_stats[class_name] = {'acc': acc_c, 'correct': correct_c, 'total': total_c,
                                                   'idx': class_idx}
    else:
        print_and_log("Warning: genotype_map is missing in evaluate_model.", log_file)

    return avg_loss_eval, overall_accuracy_eval, class_performance_stats, inference_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a Classifier on 6-channel custom .npy dataset")
    parser.add_argument("data_path", type=str, help="Path to the dataset")
    parser.add_argument("-o", "--output_path", default="./saved_models_6channel", type=str, help="Path to save model")
    parser.add_argument("--depths", type=int, nargs='+', default=[3, 3, 27, 3],
                        help="A list of depths for the ConvNeXt stages (e.g., 3 3 27 3)")
    parser.add_argument("--dims", type=int, nargs='+', default=[192, 384, 768, 1536],
                        help="A list of dimensions for the ConvNeXt stages (e.g., 192 384 768 1536)")

    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=0.001, help="Initial learning rate")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--num_workers", type=int, default=8, help="Number of workers for data loading")
    parser.add_argument("--milestone", type=int, default=10, help="Save model every N epochs")

    # --- MODIFIED: Added arguments for new optimizer and scheduler ---
    parser.add_argument("--warmup_epochs", type=int, default=5, help="Number of epochs for linear LR warmup")
    parser.add_argument("--weight_decay", type=float, default=0.05, help="Weight decay for AdamW optimizer")

    # --- REMOVED: Obsolete arguments for StepLR ---
    # parser.add_argument("--lr_decay_epochs", ...)
    # parser.add_argument("--lr_decay_factor", ...)

    parser.add_argument("--save_val_results", action='store_true', help="Save validation results at each milestone.")
    parser.add_argument("--loss_type", type=str, default="weighted_ce", choices=["combined", "weighted_ce"],
                        help="Loss function to use")
    args = parser.parse_args()

    train_model(
        data_path=args.data_path, output_path=args.output_path,
        save_val_results=args.save_val_results,
        num_epochs=args.epochs, learning_rate=args.lr,
        batch_size=args.batch_size, num_workers=args.num_workers,
        model_save_milestone=args.milestone,
        loss_type=args.loss_type,
        # --- MODIFIED: Pass new arguments to the train function ---
        warmup_epochs=args.warmup_epochs,
        weight_decay=args.weight_decay,
        depths=args.depths,
        dims=args.dims,
    )