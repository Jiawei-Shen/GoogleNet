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
from torch.utils.data import Subset, DataLoader
from torch.utils.data.distributed import DistributedSampler  # NEW
import torch.distributed as dist                             # NEW
from torch.nn.parallel import DistributedDataParallel        # NEW
import json

# --- MODIFIED: Import new schedulers ---
from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from mynet import ConvNeXtCBAMClassifier
from dataset_pansoma_npy_6ch import get_data_loader

# Globals updated in __main__ when using DDP
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IS_MAIN_PROCESS = True  # rank-0 logging only when DDP is enabled


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
        self.wce_loss = nn.CrossEntropyLoss(weight=pos_weight)

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
    # Only rank 0 logs in DDP
    if not IS_MAIN_PROCESS:
        return
    print(message)
    with open(log_path, 'a', encoding='utf-8') as f:
        f.write(message + '\n')


def _state_dict(m):
    # Works for nn.DataParallel and DDP, or plain nn.Module
    return m.module.state_dict() if hasattr(m, "module") else m.state_dict()


def _load_state_dict(m, state):
    # Load into model whether wrapped or not
    if hasattr(m, "module"):
        m.module.load_state_dict(state)
    else:
        m.load_state_dict(state)


def train_model(data_path, output_path, save_val_results=False, num_epochs=100, learning_rate=0.0001,
                batch_size=32, num_workers=4, loss_type='weighted_ce',
                warmup_epochs=10, weight_decay=0.05, depths=None, dims=None,
                training_data_ratio=1.0, ddp=False, data_parallel=False, local_rank=0,
                resume=None):
    os.makedirs(output_path, exist_ok=True)
    log_file = os.path.join(output_path, "training_log_6ch.txt")
    if os.path.exists(log_file) and IS_MAIN_PROCESS:
        os.remove(log_file)

    MIN_SAVE_EPOCH = 5  # save first checkpoint after 5 epochs

    if not (0 < training_data_ratio <= 1.0):
        raise ValueError(f"--training_data_ratio must be in (0,1], got {training_data_ratio}")

    print_and_log(f"Using device: {device}", log_file)
    print_and_log(f"Initial Learning Rate: {learning_rate:.1e}", log_file)
    print_and_log(f"Using Cosine Annealing scheduler with a {warmup_epochs}-epoch linear warmup.", log_file)
    print_and_log(
        f"Checkpointing: snapshot at epoch {MIN_SAVE_EPOCH}, then save only when validation F1(true) improves.",
        log_file)
    print_and_log(f"Using {num_workers} workers for data loading.", log_file)
    if save_val_results:
        print_and_log("Will save validation results when a new best is found.", log_file)

    # Build loaders (train)
    train_loader, genotype_map = get_data_loader(
        data_dir=data_path, dataset_type="train", batch_size=batch_size,
        num_workers=num_workers, shuffle=True
    )

    # Randomly subsample training data if requested
    if training_data_ratio < 1.0:
        full_ds = train_loader.dataset
        n = len(full_ds)
        k = max(1, int(round(n * training_data_ratio)))
        idx = torch.randperm(n, device=device)[:k].cpu().tolist()
        subset = Subset(full_ds, idx)
        train_loader = DataLoader(subset, batch_size=batch_size, shuffle=True,
                                  num_workers=num_workers, pin_memory=True)
        print_and_log(f"Training subset: using {k}/{n} samples (~{training_data_ratio:.2f} of data).", log_file)

    # Build loader (val)
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

    # Optional DataParallel (single-process, multi-GPU), ignored when DDP
    if (not ddp) and data_parallel and torch.cuda.is_available():
        n = torch.cuda.device_count()
        if n > 1:
            print_and_log(f"Wrapping model in DataParallel across {n} GPUs.", log_file)
            model = nn.DataParallel(model)
        else:
            print_and_log("DataParallel requested but only one CUDA device found; running on a single GPU.", log_file)

    # DDP (multi-process) takes precedence if requested
    if ddp:
        print_and_log(f"Wrapping model in DistributedDataParallel on cuda:{local_rank}.", log_file)
        model = DistributedDataParallel(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            gradient_as_bucket_view=True,  # keep grads contiguous for perf
            broadcast_buffers=False,
        )

        # Attach DistributedSampler to train/val datasets
        train_dataset = train_loader.dataset
        train_sampler = DistributedSampler(train_dataset, shuffle=True, drop_last=False)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers,
                                  pin_memory=True, sampler=train_sampler)

        val_dataset = val_loader.dataset
        val_sampler = DistributedSampler(val_dataset, shuffle=False, drop_last=False)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=num_workers,
                                pin_memory=True, sampler=val_sampler)
    else:
        train_sampler = None  # for uniform API below

    model.apply(init_weights)
    false_count = 48736
    true_count = 268
    pos_weight_value = min(88.0, false_count / true_count)
    class_weights = torch.tensor([1.0, pos_weight_value], device=device)

    if loss_type == "combined":
        criterion = CombinedFocalWeightedCELoss(initial_lr=learning_rate, pos_weight=class_weights)
        print_and_log(f"Using Combined Focal Loss and Weighted CE Loss.", log_file)
    elif loss_type == "weighted_ce":
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        print_and_log(f"Using Weighted CE Loss.", log_file)
    else:
        raise ValueError(f"Unsupported loss_type: {loss_type}")

    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    print_and_log(f"Using AdamW optimizer with weight decay: {weight_decay}", log_file)

    warmup_scheduler = LinearLR(optimizer, start_factor=0.01, total_iters=warmup_epochs)
    main_scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs - warmup_epochs, eta_min=0)
    scheduler = SequentialLR(optimizer, schedulers=[warmup_scheduler, main_scheduler], milestones=[warmup_epochs])

    # ---- Resume support ----
    start_epoch = 0
    best_epoch = 0
    best_f1_true = float("-inf")
    best_val_acc = float("-inf")
    best_val_loss = float("inf")
    best_prec_true = 0.0
    best_rec_true = 0.0

    if resume is not None and os.path.isfile(resume):
        try:
            checkpoint = torch.load(resume, map_location=device)
            _load_state_dict(model, checkpoint['model_state_dict'])
            if 'optimizer_state_dict' in checkpoint:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            if 'scheduler_state_dict' in checkpoint:
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            # epoch in checkpoints was saved as the last finished epoch (1-based)
            start_epoch = int(checkpoint.get('epoch', 0))
            # restore best tracking if present
            best_f1_true = float(checkpoint.get('best_f1_true', best_f1_true))
            best_rec_true = float(checkpoint.get('best_rec_true', best_rec_true))
            best_val_acc = float(checkpoint.get('best_val_acc', best_val_acc))
            best_val_loss = float(checkpoint.get('best_val_loss', best_val_loss))
            best_epoch = int(checkpoint.get('epoch', best_epoch))
            print_and_log(f"Resumed from '{resume}' at epoch {start_epoch}.", log_file)
        except Exception as e:
            print_and_log(f"WARNING: Failed to load checkpoint '{resume}': {e}", log_file)

    # ---- Training loop ----
    for epoch in range(start_epoch, num_epochs):
        model.train()

        # Ensure different shuffles each epoch with DDP
        if ddp and train_sampler is not None:
            train_sampler.set_epoch(epoch)

        running_loss = 0.0
        correct_train = 0
        total_train = 0

        current_lr = optimizer.param_groups[0]['lr']
        progress_bar = tqdm(
            train_loader,
            desc=f"Epoch {epoch + 1}/{num_epochs} LR: {current_lr:.1e}",
            leave=True,
            disable=not IS_MAIN_PROCESS  # tqdm only on rank 0
        )

        batch_count = 0
        for images, labels in progress_bar:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)

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
                    if IS_MAIN_PROCESS:
                        progress_bar.close()
                    raise TypeError(f"Model output type not recognized: {type(outputs)}")
            else:
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
                    if IS_MAIN_PROCESS:
                        progress_bar.close()
                    raise TypeError(f"Model output type not recognized: {type(outputs)}")

            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            batch_count += 1
            _, predicted = torch.max(outputs_for_acc, 1)
            correct_train += (predicted == labels).sum().item()
            total_train += labels.size(0)

            if IS_MAIN_PROCESS and total_train > 0 and batch_count > 0:
                avg_loss_train = running_loss / batch_count
                avg_acc_train = (correct_train / total_train) * 100
                progress_bar.set_postfix(loss=f"{avg_loss_train:.4f}", acc=f"{avg_acc_train:.2f}%")

        epoch_train_loss = (running_loss / batch_count) if batch_count > 0 else 0.0
        epoch_train_acc = (correct_train / total_train) * 100 if total_train > 0 else 0.0

        world_size = dist.get_world_size() if ddp and dist.is_initialized() else 1
        val_loss, val_acc, class_performance_stats_val, val_inference_results, val_metrics = evaluate_model(
            model, val_loader, criterion, genotype_map, log_file, loss_type, current_lr,
            ddp=ddp, world_size=world_size
        )

        if IS_MAIN_PROCESS and class_performance_stats_val:
            print_and_log("\nClass-wise Validation Accuracy:", log_file)
            for class_name in sorted_class_names_from_map:
                stats = class_performance_stats_val.get(class_name, {})
                print_and_log(
                    f"  {class_name} (Index {stats.get('idx', 'N/A')}): {stats.get('acc', 0):.2f}% "
                    f"({stats.get('correct', 0)}/{stats.get('total', 0)})",
                    log_file)

        # True-class metrics
        val_prec_true = val_metrics.get('precision_true', 0.0)
        val_rec_true  = val_metrics.get('recall_true', 0.0)
        val_f1_true   = val_metrics.get('f1_true', 0.0)
        pos_idx       = val_metrics.get('pos_class_idx', None)

        summary_msg = (
            f"Epoch {epoch + 1}/{num_epochs} Summary - "
            f"Train Loss: {epoch_train_loss:.4f}, Train Acc: {epoch_train_acc:.2f}%, "
            f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}% | "
            f"Prec(true): {val_prec_true * 100:.2f}%, "
            f"Rec(true): {val_rec_true * 100:.2f}%, "
            f"F1(true): {val_f1_true * 100:.2f}% "
            f"(LR: {current_lr:.1e}"
            + (f", pos_idx={pos_idx}" if pos_idx is not None else "")
            + ")"
        )
        print_and_log(summary_msg, log_file)

        # Snapshot at epoch 5 (rank 0 only)
        if (epoch + 1) == MIN_SAVE_EPOCH and IS_MAIN_PROCESS:
            snap_path = os.path.join(output_path, f"model_epoch_{MIN_SAVE_EPOCH}.pth")
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': _state_dict(model),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'genotype_map': genotype_map,
                'in_channels': 6
            }, snap_path)
            print_and_log(f"\nSnapshot saved at epoch {epoch + 1}: {snap_path}", log_file)

        # Save on validation improvement by F1(true); tie-breakers: higher recall_true, then lower val_loss
        improved = (val_f1_true > best_f1_true) or \
                   (val_f1_true == best_f1_true and val_rec_true > best_rec_true) or \
                   (val_f1_true == best_f1_true and val_rec_true == best_rec_true and val_loss < best_val_loss)

        if (epoch + 1) >= MIN_SAVE_EPOCH and improved:
            best_f1_true = val_f1_true
            best_rec_true = val_rec_true
            best_val_acc = val_acc
            best_val_loss = val_loss
            best_epoch = epoch + 1

            if IS_MAIN_PROCESS:
                best_path = os.path.join(output_path, "model_best.pth")
                torch.save({
                    'epoch': best_epoch,
                    'model_state_dict': _state_dict(model),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'genotype_map': genotype_map,
                    'in_channels': 6,
                    'best_f1_true': best_f1_true,
                    'best_rec_true': best_rec_true,
                    'best_val_acc': best_val_acc,
                    'best_val_loss': best_val_loss,
                }, best_path)
                print_and_log(
                    f"\nNew BEST at epoch {best_epoch}: "
                    f"F1(true) {best_f1_true*100:.2f}% | Rec(true) {best_rec_true*100:.2f}% | "
                    f"Val Acc {best_val_acc:.2f}% | Val Loss {best_val_loss:.4f}. "
                    f"Saved: {best_path}", log_file)

                if save_val_results:
                    result_path = os.path.join(output_path, "validation_results_best.json")
                    try:
                        with open(result_path, 'w') as f:
                            json.dump({
                                'epoch': best_epoch,
                                'f1_true': best_f1_true,
                                'recall_true': best_rec_true,
                                'val_acc': best_val_acc,
                                'val_loss': best_val_loss,
                                'inference_results': val_inference_results
                            }, f, indent=4)
                        print_and_log(f"Saved best validation results to {result_path}", log_file)
                    except Exception as e:
                        print_and_log(f"Error saving best validation results: {e}", log_file)

        scheduler.step()
        print_and_log("-" * 30, log_file)

    print_and_log(
        f"Training complete. Best epoch: {best_epoch} "
        f"| F1(true) {best_f1_true*100:.2f}% | Rec(true) {best_rec_true*100:.2f}% "
        f"| Val Acc {best_val_acc:.2f}% | Val Loss {best_val_loss:.4f}. "
        f"Best model: {os.path.join(output_path, 'model_best.pth')}",
        log_file
    )


def evaluate_model(model, data_loader, criterion, genotype_map, log_file, loss_type, current_lr,
                   ddp=False, world_size=1):
    model.eval()
    batch_count_eval = 0

    num_classes = len(genotype_map) if genotype_map else 0

    # Tensors for global metrics (on device)
    correct_eval = torch.zeros(1, device=device, dtype=torch.long)
    total_eval   = torch.zeros(1, device=device, dtype=torch.long)
    loss_sum     = torch.zeros(1, device=device, dtype=torch.float)

    # Per-class counters for metrics
    tp = torch.zeros(num_classes, device=device, dtype=torch.long)
    fp = torch.zeros(num_classes, device=device, dtype=torch.long)
    fn = torch.zeros(num_classes, device=device, dtype=torch.long)
    class_correct_counts = torch.zeros(num_classes, device=device, dtype=torch.long)
    class_total_counts   = torch.zeros(num_classes, device=device, dtype=torch.long)

    inference_results = defaultdict(list)
    idx_to_class = {v: k for k, v in genotype_map.items()} if genotype_map else {}

    if not data_loader:
        metrics = {
            'precision_true': 0.0, 'recall_true': 0.0, 'f1_true': 0.0, 'pos_class_idx': None
        }
        return 0.0, 0.0, {}, {}, metrics

    with torch.no_grad():
        for batch in data_loader:
            # Support both (images, labels) and (images, labels, paths)
            if len(batch) == 3:
                images, labels, paths = batch
            else:
                images, labels = batch
                paths = [""] * labels.size(0)

            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            if isinstance(outputs, tuple):
                outputs = outputs[0]

            # Loss
            loss = criterion(outputs, labels, current_lr) if loss_type == "combined" else criterion(outputs, labels)
            loss_sum += loss.detach()
            batch_count_eval += 1

            # Predictions
            _, predicted = torch.max(outputs, 1)
            correct_eval += (predicted == labels).sum()
            total_eval += labels.size(0)

            # Per-class counters
            for i in range(labels.size(0)):
                pred_idx = int(predicted[i])
                true_idx = int(labels[i])

                class_total_counts[true_idx] += 1
                if pred_idx == true_idx:
                    class_correct_counts[true_idx] += 1
                    tp[true_idx] += 1
                else:
                    if pred_idx < num_classes:
                        fp[pred_idx] += 1
                    fn[true_idx] += 1

                if idx_to_class and paths[i]:
                    predicted_class_name = idx_to_class.get(pred_idx, str(pred_idx))
                    inference_results[predicted_class_name].append(os.path.basename(paths[i]))

    # --- DDP: reduce sums across all processes ---
    if ddp and world_size > 1 and dist.is_initialized():
        dist.all_reduce(correct_eval, op=dist.ReduceOp.SUM)
        dist.all_reduce(total_eval,   op=dist.ReduceOp.SUM)
        dist.all_reduce(loss_sum,     op=dist.ReduceOp.SUM)

        if num_classes > 0:
            dist.all_reduce(tp, op=dist.ReduceOp.SUM)
            dist.all_reduce(fp, op=dist.ReduceOp.SUM)
            dist.all_reduce(fn, op=dist.ReduceOp.SUM)
            dist.all_reduce(class_correct_counts, op=dist.ReduceOp.SUM)
            dist.all_reduce(class_total_counts,   op=dist.ReduceOp.SUM)

    # Global averages
    denom_batches = max(1, batch_count_eval * (world_size if (ddp and world_size > 1) else 1))
    avg_loss_eval = (loss_sum.item() / denom_batches)
    overall_accuracy_eval = (correct_eval.item() / max(1, total_eval.item())) * 100.0

    # Class-wise stats
    class_performance_stats = {}
    if genotype_map:
        for class_name, class_idx in genotype_map.items():
            correct_c = int(class_correct_counts[class_idx].item())
            total_c   = int(class_total_counts[class_idx].item())
            acc_c = (correct_c / total_c * 100.0) if total_c > 0 else 0.0
            class_performance_stats[class_name] = {
                'acc': acc_c, 'correct': correct_c, 'total': total_c, 'idx': class_idx
            }

    # ---- NEW: precision / recall / F1 for the positive ("true") class only ----
    # Choose pos_idx:
    # 1) prefer class named "true" (case-insensitive)
    # 2) else pick index 1 if present
    # 3) else choose minority (smallest support)
    pos_idx = None
    if genotype_map:
        for name, idx in genotype_map.items():
            if str(name).lower() == "true":
                pos_idx = idx
                break
    if pos_idx is None:
        if 1 < num_classes:
            pos_idx = 1
        elif num_classes > 0:
            supports = class_total_counts.clone()
            if supports.sum() > 0:
                pos_idx = int(torch.nonzero(supports == supports[supports > 0].min(), as_tuple=False)[0].item())
            else:
                pos_idx = 0
        else:
            pos_idx = 0

    tpc = float(tp[pos_idx].item() if pos_idx < num_classes else 0.0)
    fpc = float(fp[pos_idx].item() if pos_idx < num_classes else 0.0)
    fnc = float(fn[pos_idx].item() if pos_idx < num_classes else 0.0)

    precision_true = (tpc / (tpc + fpc)) if (tpc + fpc) > 0 else 0.0
    recall_true    = (tpc / (tpc + fnc)) if (tpc + fnc) > 0 else 0.0
    f1_true        = (2 * precision_true * recall_true / (precision_true + recall_true)) \
                     if (precision_true + recall_true) > 0 else 0.0

    metrics = {
        'precision_true': precision_true,
        'recall_true': recall_true,
        'f1_true': f1_true,
        'pos_class_idx': pos_idx,
    }

    return avg_loss_eval, overall_accuracy_eval, class_performance_stats, inference_results, metrics


# --- Helpers for path resolution (single, non-duplicated) ---
def _read_paths_file(file_path):
    paths = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith('#'):
                continue
            paths.append(os.path.abspath(os.path.expanduser(s)))
    return paths


def _resolve_data_roots(primary_path, extra_paths, paths_file):
    candidates = []
    if primary_path:
        candidates.append(os.path.abspath(os.path.expanduser(primary_path)))
    if extra_paths:
        for p in extra_paths:
            candidates.append(os.path.abspath(os.path.expanduser(p)))
    if paths_file:
        candidates.extend(_read_paths_file(paths_file))
    seen = set()
    deduped = []
    for p in candidates:
        if p not in seen:
            seen.add(p)
            deduped.append(p)
    if len(deduped) == 0:
        return primary_path
    if len(deduped) == 1:
        return deduped[0]
    return deduped


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a Classifier on 6-channel custom .npy dataset")

    # Make data_path OPTIONAL; we will enforce XOR with the files mode
    parser.add_argument("data_path", nargs="?", type=str,
                        help="Dataset root containing 'train/' and 'val/' (Mode A).")

    parser.add_argument("-o", "--output_path", default="./saved_models_6channel", type=str, help="Path to save model")
    parser.add_argument("--depths", type=int, nargs='+', default=[3, 3, 27, 3],
                        help="A list of depths for the ConvNeXt stages (e.g., 3 3 27 3)")
    parser.add_argument("--dims", type=int, nargs='+', default=[192, 384, 768, 1536],
                        help="A list of dimensions for the ConvNeXt stages (e.g., 192 384 768 1536)")

    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=0.0001, help="Initial learning rate")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--num_workers", type=int, default=8, help="Number of workers for data loading")

    # Optimizer / scheduler (unchanged)
    parser.add_argument("--warmup_epochs", type=int, default=3, help="Number of epochs for linear LR warmup")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay for AdamW optimizer")
    parser.add_argument("--save_val_results", action='store_true', help="Save validation results when best is found.")
    parser.add_argument("--loss_type", type=str, default="weighted_ce", choices=["combined", "weighted_ce"],
                        help="Loss function to use")

    # Mode B (files) — both must be provided together when using files mode
    parser.add_argument("--train_data_paths_file", type=str, default=None,
                        help="Text file listing TRAIN dataset roots (one per line).")
    parser.add_argument("--val_data_paths_file", type=str, default=None,
                        help="Text file listing VAL dataset roots (one per line).")

    # Subsample ratio
    parser.add_argument("--training_data_ratio", type=float, default=1.0,
                        help="Proportion of training data to use (0–1]. Randomly subsamples the training set.")

    # Multi-GPU switches
    parser.add_argument("--ddp", action="store_true",
                        help="Use DistributedDataParallel (multi-process). Launch with torchrun.")
    parser.add_argument("--data_parallel", action="store_true",
                        help="Use nn.DataParallel across all visible GPUs (single process). Ignored if --ddp is set.")

    # NEW: resume from checkpoint
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to a checkpoint .pth to resume training (loads model/optimizer/scheduler/epoch).")

    args = parser.parse_args()

    # ---- Enforce: exactly one of (data_path) OR (both files) ----
    has_base = args.data_path is not None
    has_both_files = (args.train_data_paths_file is not None) and (args.val_data_paths_file is not None)

    if not (0 < args.training_data_ratio <= 1.0):
        parser.error(f"--training_data_ratio must be in (0,1], got {args.training_data_ratio}")

    if has_base and has_both_files:
        parser.error("Provide either positional data_path (Mode A) OR both --train_data_paths_file and "
                     "--val_data_paths_file (Mode B), not both.")
    if not has_base and not has_both_files:
        parser.error("You must provide exactly one input mode:\n"
                     "  • Mode A: data_path\n"
                     "  • Mode B: --train_data_paths_file and --val_data_paths_file")

    # Build the argument passed into train_model:
    #  - Mode A: a single string root (backward compatible)
    #  - Mode B: a pair (train_roots, val_roots) for the revised get_data_loader
    if has_base:
        data_path_or_pair = os.path.abspath(os.path.expanduser(args.data_path))
    else:
        train_roots = _read_paths_file(args.train_data_paths_file)
        val_roots   = _read_paths_file(args.val_data_paths_file)
        if not train_roots:
            parser.error(f"--train_data_paths_file is empty or unreadable: {args.train_data_paths_file}")
        if not val_roots:
            parser.error(f"--val_data_paths_file is empty or unreadable: {args.val_data_paths_file}")
        # Pair: get_data_loader(dataset_type="train"/"val") will pick the right side and
        # include BOTH 'train' and 'val' subfolders from each root (per your revised dataloader)
        data_path_or_pair = (train_roots, val_roots)

    # --- DDP init (takes precedence over DataParallel) ---
    if args.ddp:
        if not torch.cuda.is_available():
            raise RuntimeError("DDP requires CUDA available.")
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
        dist.init_process_group(backend="nccl", init_method="env://")
        IS_MAIN_PROCESS = (dist.get_rank() == 0)
        if IS_MAIN_PROCESS:
            print(f"[DDP] World size={dist.get_world_size()} | Local rank={local_rank} | Global rank={dist.get_rank()}")
    else:
        local_rank = 0
        IS_MAIN_PROCESS = True
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Hand off; train_model still discovers loaders via get_data_loader(...)
    train_model(
        data_path=data_path_or_pair, output_path=args.output_path,
        save_val_results=args.save_val_results,
        num_epochs=args.epochs, learning_rate=args.lr,
        batch_size=args.batch_size, num_workers=args.num_workers,
        loss_type=args.loss_type,
        warmup_epochs=args.warmup_epochs,
        weight_decay=args.weight_decay,
        depths=args.depths,
        dims=args.dims,
        training_data_ratio=args.training_data_ratio,
        ddp=args.ddp, data_parallel=args.data_parallel, local_rank=local_rank,
        resume=args.resume,
    )

    # Clean up DDP
    if args.ddp and dist.is_initialized():
        dist.destroy_process_group()