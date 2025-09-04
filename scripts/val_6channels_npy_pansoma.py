#!/usr/bin/env python3
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os
import json
from tqdm import tqdm
from collections import defaultdict
from torch.utils.data import DataLoader, Subset

# ------------------------------------------------------------
# Imports from your project
# ------------------------------------------------------------
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from mynet import ConvNeXtCBAMClassifier
from dataset_pansoma_npy_6ch import get_data_loader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ------------------------------------------------------------
# Optional losses (for reporting val loss only)
# ------------------------------------------------------------
class MultiClassFocalLoss(nn.Module):
    def __init__(self, gamma=2.0, weight=None, reduction='mean'):
        super().__init__()
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

        loss = -1 * (1 - pt) ** self.gamma * log_pt
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss


class CombinedFocalWeightedCELoss(nn.Module):
    """
    Same as before, but for validation we pass current_lr=0 so it
    behaves like pure weighted CE (wce_weight=1.0) unless you override.
    """
    def __init__(self, initial_lr=1e-4, pos_weight=None, gamma=2.0):
        super().__init__()
        self.initial_lr = initial_lr
        self.focal_loss = MultiClassFocalLoss(gamma=gamma, weight=pos_weight)
        self.wce_loss = nn.CrossEntropyLoss(weight=pos_weight)

    def forward(self, logits, targets, current_lr=0.0):
        focal_weight = 1.0 - (current_lr / self.initial_lr) if self.initial_lr > 0 else 1.0
        wce_weight = 1.0 - focal_weight
        lf = self.focal_loss(logits, targets)
        lw = self.wce_loss(logits, targets)
        return focal_weight * lf + wce_weight * lw


# ------------------------------------------------------------
# Utility
# ------------------------------------------------------------
def print_and_log(msg, log_path):
    print(msg)
    with open(log_path, 'a', encoding='utf-8') as f:
        f.write(msg + '\n')


def _read_paths_file(file_path):
    paths = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                s = line.strip()
                if not s or s.startswith('#'):
                    continue
                paths.append(os.path.abspath(os.path.expanduser(s)))
    except Exception:
        pass
    return paths


# ------------------------------------------------------------
# Evaluation
# ------------------------------------------------------------
@torch.no_grad()
def evaluate_model(model, data_loader, criterion, genotype_map, log_file, loss_type):
    model.eval()
    running_loss = 0.0
    n_batches = 0
    correct = 0
    total = 0

    class_correct = defaultdict(int)
    class_total = defaultdict(int)

    # for P/R/F1
    tp = defaultdict(int)
    fp = defaultdict(int)
    fn = defaultdict(int)

    inference_results = defaultdict(list)
    idx_to_class = {v: k for k, v in genotype_map.items()}

    if not data_loader or len(data_loader) == 0:
        metrics = {
            'precision_macro': 0.0, 'recall_macro': 0.0, 'f1_macro': 0.0,
            'precision_weighted': 0.0, 'recall_weighted': 0.0, 'f1_weighted': 0.0
        }
        return 0.0, 0.0, {}, {}, metrics

    for images, labels, paths in tqdm(data_loader, desc="Validating", leave=True):
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        if isinstance(outputs, tuple):
            outputs_for_acc = outputs[0]
        else:
            outputs_for_acc = outputs

        # compute "val loss" for reference
        if loss_type == "combined":
            loss = criterion(outputs_for_acc, labels, current_lr=0.0)
        else:
            loss = criterion(outputs_for_acc, labels)

        running_loss += float(loss.item())
        n_batches += 1

        _, pred = torch.max(outputs_for_acc, 1)
        correct += (pred == labels).sum().item()
        total += labels.size(0)

        for i in range(labels.size(0)):
            pred_idx = int(pred[i].item())
            true_idx = int(labels[i].item())
            path = paths[i]

            class_total[true_idx] += 1
            if pred_idx == true_idx:
                class_correct[true_idx] += 1
                tp[true_idx] += 1
            else:
                fp[pred_idx] += 1
                fn[true_idx] += 1

            predicted_class_name = idx_to_class.get(pred_idx, str(pred_idx))
            inference_results[predicted_class_name].append(os.path.basename(path))

    avg_loss = running_loss / n_batches if n_batches > 0 else 0.0
    acc = (correct / total) * 100 if total > 0 else 0.0

    # per-class stats
    class_stats = {}
    for cname, cidx in genotype_map.items():
        ctot = class_total[cidx]
        ccorr = class_correct[cidx]
        cacc = (ccorr / ctot) * 100 if ctot > 0 else 0.0
        class_stats[cname] = {'acc': cacc, 'correct': ccorr, 'total': ctot, 'idx': cidx}

    # macro / weighted P/R/F1
    class_indices = list(genotype_map.values())
    precisions, recalls, f1s, supports = [], [], [], []
    for c in class_indices:
        tpc, fpc, fnc = tp[c], fp[c], fn[c]
        p = (tpc / (tpc + fpc)) if (tpc + fpc) > 0 else 0.0
        r = (tpc / (tpc + fnc)) if (tpc + fnc) > 0 else 0.0
        f1 = (2 * p * r / (p + r)) if (p + r) > 0 else 0.0
        precisions.append(p); recalls.append(r); f1s.append(f1); supports.append(tpc + fnc)

    if len(class_indices) > 0:
        precision_macro = sum(precisions) / len(precisions)
        recall_macro = sum(recalls) / len(recalls)
        f1_macro = sum(f1s) / len(f1s)
    else:
        precision_macro = recall_macro = f1_macro = 0.0

    total_support = sum(supports)
    if total_support > 0:
        precision_weighted = sum(p * s for p, s in zip(precisions, supports)) / total_support
        recall_weighted = sum(r * s for r, s in zip(recalls, supports)) / total_support
        f1_weighted = sum(f * s for f, s in zip(f1s, supports)) / total_support
    else:
        precision_weighted = recall_weighted = f1_weighted = 0.0

    metrics = {
        'precision_macro': precision_macro,
        'recall_macro': recall_macro,
        'f1_macro': f1_macro,
        'precision_weighted': precision_weighted,
        'recall_weighted': recall_weighted,
        'f1_weighted': f1_weighted,
    }

    return avg_loss, acc, class_stats, inference_results, metrics


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Validate a saved classifier on 6-channel .npy dataset (VAL ONLY)")

    # Data (Mode A or Mode B: only val is needed)
    parser.add_argument("data_path", nargs="?", type=str,
                        help="Dataset root that contains 'val/' (Mode A).")
    parser.add_argument("--val_data_paths_file", type=str, default=None,
                        help="Text file listing VAL dataset roots (one per line). (Mode B)")

    # Model checkpoint (required)
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to the saved checkpoint (.pth) with model_state_dict etc.")
    parser.add_argument("-o", "--output_path", default="./val_only_output", type=str,
                        help="Path to write logs / optional JSON.")

    # Dataloader
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for validation")
    parser.add_argument("--num_workers", type=int, default=8, help="#workers for dataloader")

    # Arch (used only if checkpoint lacks dims/depths; otherwise ignored)
    parser.add_argument("--depths", type=int, nargs='+', default=[3, 3, 27, 3],
                        help="ConvNeXt stage depths, used if ckpt missing this info")
    parser.add_argument("--dims", type=int, nargs='+', default=[192, 384, 768, 1536],
                        help="ConvNeXt dims, used if ckpt missing this info")

    # Loss reporting options
    parser.add_argument("--loss_type", type=str, default="weighted_ce",
                        choices=["combined", "weighted_ce"],
                        help="Which loss to report on validation")
    parser.add_argument("--initial_lr", type=float, default=1e-4,
                        help="Only for combined loss weighting; val uses current_lr=0.0")
    parser.add_argument("--class_weights_csv", type=str, default=None,
                        help="Optional comma-separated class weights, e.g. '1,88' to weight CE/Focal.")

    # Output toggles
    parser.add_argument("--save_val_results_json", action="store_true",
                        help="If set, save inference_results + metrics JSON to output_path.")

    args = parser.parse_args()
    os.makedirs(args.output_path, exist_ok=True)
    log_file = os.path.join(args.output_path, "val_log_6ch.txt")
    if os.path.exists(log_file):
        os.remove(log_file)

    # ---------- Build VAL loader ----------
    if (args.data_path is None) == (args.val_data_paths_file is None):
        parser.error("Provide exactly one of: positional data_path (Mode A) OR --val_data_paths_file (Mode B).")

    if args.data_path is not None:
        val_source = os.path.abspath(os.path.expanduser(args.data_path))
    else:
        val_roots = _read_paths_file(args.val_data_paths_file)
        if not val_roots:
            parser.error(f"--val_data_paths_file is empty or unreadable: {args.val_data_paths_file}")
        # get_data_loader will look for 'val' subfolder under each root per your dataloader
        val_source = val_roots

    try:
        val_loader, genotype_map = get_data_loader(
            data_dir=val_source,
            dataset_type="val",
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            shuffle=False,
            return_paths=True
        )
    except Exception as e:
        print_and_log(f"FATAL: could not create VAL dataloader with return_paths=True\nError: {e}", log_file)
        sys.exit(1)

    if not genotype_map:
        print_and_log("Error: genotype_map is empty (from dataloader).", log_file)
        sys.exit(1)
    num_classes = len(genotype_map)
    print_and_log(f"Using device: {device}", log_file)
    print_and_log(f"Detected {num_classes} classes from dataloader.", log_file)

    # ---------- Load checkpoint ----------
    ckpt_path = os.path.abspath(os.path.expanduser(args.model_path))
    if not os.path.isfile(ckpt_path):
        print_and_log(f"Checkpoint not found: {ckpt_path}", log_file)
        sys.exit(1)

    ckpt = torch.load(ckpt_path, map_location="cpu")
    ckpt_in_channels = ckpt.get('in_channels', 6)
    ckpt_genotype_map = ckpt.get('genotype_map', None)
    ckpt_depths = ckpt.get('depths', None)
    ckpt_dims = ckpt.get('dims', None)

    # If ckpt carries its own genotype_map, prefer it (and warn if mismatch)
    if ckpt_genotype_map:
        if ckpt_genotype_map != genotype_map:
            print_and_log("Warning: genotype_map in checkpoint differs from dataloader map. "
                          "Using checkpoint's map for class names/indices.", log_file)
        genotype_map = ckpt_genotype_map
        num_classes = len(genotype_map)

    depths = ckpt_depths if ckpt_depths is not None else args.depths
    dims = ckpt_dims if ckpt_dims is not None else args.dims

    # ---------- Build model and load weights ----------
    model = ConvNeXtCBAMClassifier(in_channels=ckpt_in_channels, class_num=num_classes,
                                   depths=depths, dims=dims).to(device)

    state = ckpt.get('model_state_dict', ckpt)  # allow raw state_dict file
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print_and_log(f"Note: missing keys when loading state_dict: {missing}", log_file)
    if unexpected:
        print_and_log(f"Note: unexpected keys when loading state_dict: {unexpected}", log_file)

    # ---------- Build criterion (for loss reporting only) ----------
    class_weights = None
    if args.class_weights_csv:
        try:
            ws = [float(x.strip()) for x in args.class_weights_csv.split(",")]
            class_weights = torch.tensor(ws, dtype=torch.float32, device=device)
            if len(ws) != num_classes:
                print_and_log("Warning: class_weights length != num_classes; ignoring.", log_file)
                class_weights = None
        except Exception as e:
            print_and_log(f"Warning: failed to parse --class_weights_csv: {e}", log_file)
            class_weights = None

    if args.loss_type == "combined":
        criterion = CombinedFocalWeightedCELoss(initial_lr=args.initial_lr, pos_weight=class_weights)
        print_and_log("Validation loss: Combined(Focal+Weighted CE) with current_lr=0.0.", log_file)
    else:
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        print_and_log("Validation loss: Weighted CE.", log_file)

    # ---------- Evaluate ----------
    val_loss, val_acc, class_stats, inference_results, metrics = evaluate_model(
        model, val_loader, criterion, genotype_map, log_file, args.loss_type
    )

    # ---------- Report ----------
    print_and_log("\n=== Validation Summary ===", log_file)
    print_and_log(f"Val Loss: {val_loss:.4f}", log_file)
    print_and_log(f"Val Acc : {val_acc:.2f}%", log_file)
    print_and_log(
        f"Precision (macro): {metrics['precision_macro']*100:.2f}% | "
        f"Recall (macro): {metrics['recall_macro']*100:.2f}% | "
        f"F1 (macro): {metrics['f1_macro']*100:.2f}%", log_file
    )
    print_and_log(
        f"Precision (weighted): {metrics['precision_weighted']*100:.2f}% | "
        f"Recall (weighted): {metrics['recall_weighted']*100:.2f}% | "
        f"F1 (weighted): {metrics['f1_weighted']*100:.2f}%", log_file
    )

    if class_stats:
        print_and_log("\nPer-class Accuracy:", log_file)
        for cname, stats in sorted(class_stats.items(), key=lambda kv: kv[1]['idx']):
            print_and_log(
                f"  {cname} (idx {stats['idx']}): {stats['acc']:.2f}% "
                f"({stats['correct']}/{stats['total']})", log_file
            )

    if args.save_val_results_json:
        out_json = os.path.join(args.output_path, "validation_results.json")
        payload = {
            'val_loss': val_loss,
            'val_acc': val_acc,
            'metrics': metrics,
            'class_stats': class_stats,
            'inference_results': inference_results,
            'genotype_map': genotype_map,
        }
        try:
            with open(out_json, 'w') as f:
                json.dump(payload, f, indent=2)
            print_and_log(f"\nSaved validation JSON to: {out_json}", log_file)
        except Exception as e:
            print_and_log(f"Error saving validation_results.json: {e}", log_file)


if __name__ == "__main__":
    main()
