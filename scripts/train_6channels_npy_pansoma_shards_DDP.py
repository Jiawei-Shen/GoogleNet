#!/usr/bin/env python3
import argparse
import json
import os
import sys
import random
from collections import defaultdict

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel
from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR
from torch.utils.data import DataLoader, Subset, ConcatDataset
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

# ---- env + backend knobs (helps speed) ----
os.environ.setdefault("OMP_NUM_THREADS", "1")
torch.backends.cudnn.benchmark = True

# local imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
try:
    from mynet import ConvNeXtCBAMClassifier
    from dataset_pansoma_npy_sharded_6ch_DDP import get_data_loader  # returns (loader, genotype_map)
except ImportError:
    # Fallback for standalone testing if local files missing
    print("Warning: Local imports failed. Ensure 'mynet' and 'dataset_pansoma_npy_6ch' are in path.")
    pass

# Globals updated in __main__ with DDP
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IS_MAIN_PROCESS = True  # rank-0 logging only


# ---------------- Losses ----------------
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

        focal_loss = -1 * (1 - pt) ** self.gamma * log_pt
        if self.reduction == 'mean':
            return focal_loss.mean()
        if self.reduction == 'sum':
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
        return focal_weight * self.focal_loss(logits, targets) + wce_weight * self.wce_loss(logits, targets)


# ---------------- Utils ----------------
def init_weights(m):
    if isinstance(m, (nn.Conv2d, nn.Linear)):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)


def print_and_log(message, log_path):
    if not IS_MAIN_PROCESS:
        return
    print(message, flush=True)
    with open(log_path, 'a', encoding='utf-8') as f:
        f.write(message + '\n')


def _state_dict(m):
    return m.module.state_dict() if hasattr(m, "module") else m.state_dict()


def _load_state_dict(m, state):
    if hasattr(m, "module"):
        m.module.load_state_dict(state)
    else:
        m.load_state_dict(state)


def _unique_path(path: str) -> str:
    if not os.path.exists(path):
        return path
    base, ext = os.path.splitext(path)
    n = 2
    cand = f"{base}_v{n}{ext}"
    while os.path.exists(cand):
        n += 1
        cand = f"{base}_v{n}{ext}"
    return cand


def _read_paths_file(file_path):
    paths = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith('#'):
                continue
            paths.append(os.path.abspath(os.path.expanduser(s)))
    return paths


# ---- central helper to build loaders with fast knobs ----
def _make_loader(dataset,
                 batch_size,
                 shuffle,
                 num_workers,
                 pin_memory=True,
                 persistent_workers=True,
                 prefetch_factor=16,
                 multiprocessing_context=None,
                 sampler=None):
    kwargs = dict(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=(shuffle and sampler is None),
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers if num_workers > 0 else False,
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
    )
    if multiprocessing_context is not None:
        kwargs["multiprocessing_context"] = multiprocessing_context
    if sampler is not None:
        kwargs["sampler"] = sampler
        kwargs["shuffle"] = False
    return DataLoader(**kwargs)


# ---- NEW: Build list of individual loaders (Shards) instead of Concatenating ----
def _build_sharded_loaders_from_roots(roots, split, batch_size, num_workers, shuffle,
                                      prefetch_factor, mp_ctx, ddp=False, training_data_ratio=1.0):
    loaders = []
    genotype_map = None

    for r in roots:
        # 1. Get the dataset object for this specific root (shard)
        ld, gm = get_data_loader(
            data_dir=r, dataset_type=split, batch_size=batch_size,
            num_workers=num_workers, shuffle=False,
            return_paths=False
        )
        ds = getattr(ld, "dataset", None)

        # Check validity
        if ds is None:
            continue
        try:
            if len(ds) == 0: continue
        except Exception:
            pass

        # Check map consistency
        if genotype_map is None:
            genotype_map = gm
        elif gm != genotype_map:
            raise ValueError(f"Inconsistent genotype_map between roots; offending root: {r}")

        # Optional: Subsample this shard if ratio < 1.0
        if training_data_ratio < 1.0:
            n = len(ds)
            k = max(1, int(round(n * training_data_ratio)))
            # Deterministic subset seed
            gen = torch.Generator()
            gen.manual_seed(123456)
            idx_list = torch.randperm(n, generator=gen)[:k].tolist()
            ds = Subset(ds, idx_list)

        # 2. Create Sampler for this specific shard
        sampler = None
        if ddp:
            # shuffle=True here means shuffle WITHIN the shard
            sampler = DistributedSampler(ds, shuffle=True, drop_last=False)

        # 3. Create DataLoader for this shard
        loader = _make_loader(
            dataset=ds,
            batch_size=batch_size,
            shuffle=shuffle,  # Ignored if sampler is set
            num_workers=num_workers,
            prefetch_factor=prefetch_factor,
            multiprocessing_context=mp_ctx,
            pin_memory=True,
            persistent_workers=True,
            sampler=sampler
        )
        loaders.append(loader)

    if not loaders:
        raise ValueError(f"No valid datasets found for split='{split}' in provided roots.")

    return loaders, genotype_map


# ---- Old builder (kept for Validation, which is faster if concatenated) ----
def _build_concatenated_loader_from_roots(roots, split, batch_size, num_workers, shuffle,
                                          prefetch_factor, mp_ctx, return_paths=False, ddp=False):
    datasets = []
    genotype_map = None
    for r in roots:
        ld, gm = get_data_loader(
            data_dir=r, dataset_type=split, batch_size=batch_size,
            num_workers=num_workers, shuffle=False,
            return_paths=return_paths
        )
        ds = getattr(ld, "dataset", None)
        if ds and len(ds) > 0:
            if genotype_map is None:
                genotype_map = gm
            elif gm != genotype_map:
                raise ValueError(f"Inconsistent genotype_map in {r}")
            datasets.append(ds)

    if not datasets: raise ValueError(f"No datasets for {split}")

    concat = ConcatDataset(datasets)

    sampler = None
    if ddp:
        sampler = DistributedSampler(concat, shuffle=False, drop_last=False)

    loader = _make_loader(
        dataset=concat, batch_size=batch_size, shuffle=shuffle,
        num_workers=num_workers, prefetch_factor=prefetch_factor,
        multiprocessing_context=mp_ctx, pin_memory=True,
        persistent_workers=True, sampler=sampler
    )
    return loader, genotype_map


def _build_mode_c_loaders_sharded(data_paths, batch_size, num_workers, prefetch_factor, mp_ctx, ddp, ratio):
    """
    Mode C:
      Train = List of Loaders (one per root).
      Val   = Single Concatenated Loader (union of all).
    """
    roots = [os.path.abspath(os.path.expanduser(p)) for p in data_paths]

    # Train: Keep separate
    train_loaders, gm_tr = _build_sharded_loaders_from_roots(
        roots, "train", batch_size, num_workers, shuffle=True,
        prefetch_factor=prefetch_factor, mp_ctx=mp_ctx, ddp=ddp, training_data_ratio=ratio
    )

    # Val: Concatenate (Standard eval behavior)
    val_loader, gm_val = _build_concatenated_loader_from_roots(
        roots, "val", batch_size, num_workers, shuffle=False,
        prefetch_factor=prefetch_factor, mp_ctx=mp_ctx, return_paths=True, ddp=ddp
    )

    if gm_val != gm_tr:
        raise ValueError("Inconsistent genotype_map between train and val.")

    return train_loaders, val_loader, gm_tr


# ---------------- Train / Eval ----------------
def train_model(data_path, output_path, save_val_results=False, num_epochs=100, learning_rate=1e-4,
                batch_size=32, num_workers=4, loss_type='weighted_ce',
                warmup_epochs=10, weight_decay=0.05, depths=None, dims=None,
                training_data_ratio=1.0, ddp=False, data_parallel=False, local_rank=0,
                resume=None, pos_weight=88.0,
                prefetch_factor=4, mp_context=None):
    os.makedirs(output_path, exist_ok=True)
    log_file = os.path.join(output_path, "training_log_6ch.txt")
    if os.path.exists(log_file) and IS_MAIN_PROCESS:
        os.remove(log_file)

    MIN_SAVE_EPOCH = 5

    print_and_log(f"Using device: {device}", log_file)
    print_and_log(f"Initial Learning Rate: {learning_rate:.1e}", log_file)
    print_and_log(f"Loading strategy: Shard-by-shard training (random order per epoch).", log_file)

    # ---------------- Build loaders ----------------
    # train_loaders will be a LIST of DataLoaders
    # val_loader will be a SINGLE DataLoader (concatenated)

    train_loaders = []
    val_loader = None
    genotype_map = None

    if isinstance(data_path, str):
        # Mode A: Single root (List of length 1)
        roots = [data_path]
        train_loaders, genotype_map = _build_sharded_loaders_from_roots(
            roots, "train", batch_size, num_workers, True, prefetch_factor, mp_context, ddp, training_data_ratio
        )
        val_loader, _ = _build_concatenated_loader_from_roots(
            roots, "val", batch_size, num_workers, False, prefetch_factor, mp_context, True, ddp
        )
    elif isinstance(data_path, tuple) and len(data_path) == 2:
        # Mode B: Explicit lists
        train_roots, val_roots = data_path
        train_loaders, genotype_map = _build_sharded_loaders_from_roots(
            train_roots, "train", batch_size, num_workers, True, prefetch_factor, mp_context, ddp, training_data_ratio
        )
        val_loader, gm_val = _build_concatenated_loader_from_roots(
            val_roots, "val", batch_size, num_workers, False, prefetch_factor, mp_context, True, ddp
        )
        if gm_val != genotype_map: raise ValueError("Map mismatch.")
    elif isinstance(data_path, (list, tuple)):
        # Mode C: Multi roots
        train_loaders, val_loader, genotype_map = _build_mode_c_loaders_sharded(
            data_path, batch_size, num_workers, prefetch_factor, mp_context, ddp, training_data_ratio
        )
    else:
        raise ValueError("Unsupported data_path type.")

    # Calculate total length for progress bar
    total_batches_per_epoch = sum([len(loader) for loader in train_loaders])
    print_and_log(f"Total shards (files/roots): {len(train_loaders)}", log_file)
    print_and_log(f"Total batches per epoch: {total_batches_per_epoch}", log_file)

    # ---- Model / parallelism ----
    num_classes = len(genotype_map)
    print_and_log(f"Number of classes: {num_classes}", log_file)

    model = ConvNeXtCBAMClassifier(in_channels=6, class_num=num_classes, depths=depths, dims=dims).to(device)

    if (not ddp) and data_parallel and torch.cuda.is_available():
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)

    if ddp:
        model = DistributedDataParallel(
            model, device_ids=[local_rank], output_device=local_rank,
            gradient_as_bucket_view=True, broadcast_buffers=False,
        )

    model.apply(init_weights)

    # ---- Loss / Optim / Sched ----
    pos_weight_value = float(pos_weight)
    class_weights = torch.ones(num_classes, device=device, dtype=torch.float32)
    if num_classes >= 2:
        class_weights[1] = pos_weight_value

    if loss_type == "combined":
        criterion = CombinedFocalWeightedCELoss(initial_lr=learning_rate, pos_weight=class_weights)
    else:
        criterion = nn.CrossEntropyLoss(weight=class_weights)

    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    warmup_scheduler = LinearLR(optimizer, start_factor=0.01, total_iters=warmup_epochs)
    main_scheduler = CosineAnnealingLR(optimizer, T_max=max(1, num_epochs - warmup_epochs), eta_min=0)
    scheduler = SequentialLR(optimizer, schedulers=[warmup_scheduler, main_scheduler], milestones=[warmup_epochs])

    # ---- Resume ----
    start_epoch = 0
    best_f1_true = float("-inf")
    best_val_acc = float("-inf")
    best_val_loss = float("inf")
    best_rec_true = 0.0

    if resume and os.path.isfile(resume):
        checkpoint = torch.load(resume, map_location=device)
        _load_state_dict(model, checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = int(checkpoint.get('epoch', 0))
        best_f1_true = float(checkpoint.get('best_f1_true', best_f1_true))
        print_and_log(f"Resumed from '{resume}' at epoch {start_epoch}.", log_file)

    # ---- Train loop ----
    sorted_class_names = sorted(genotype_map.keys(), key=lambda k: genotype_map[k])

    for epoch in range(start_epoch, num_epochs):
        model.train()

        running_loss = 0.0
        correct_train = 0
        total_train = 0
        current_lr = optimizer.param_groups[0]['lr']

        # 1. Deterministic Shard Shuffling
        # All ranks must agree on the order of shards so they train on the same data type at the same time
        shard_indices = list(range(len(train_loaders)))

        # Use a Generator with seed = epoch to ensure all ranks shuffle shards identically
        g_shuffle = torch.Generator()
        g_shuffle.manual_seed(epoch + 1000)
        perm = torch.randperm(len(shard_indices), generator=g_shuffle)
        shard_indices = [shard_indices[i] for i in perm.tolist()]

        # Unified progress bar for the whole epoch
        progress_bar = tqdm(total=total_batches_per_epoch,
                            desc=f"Epoch {epoch + 1}/{num_epochs} [Shards Randomized]",
                            leave=True, disable=not IS_MAIN_PROCESS)

        batch_count_epoch = 0

        # 2. Iterate through Shards one by one
        for shard_idx in shard_indices:
            loader = train_loaders[shard_idx]

            # Important: Set epoch for DistributedSampler inside this shard
            if ddp and hasattr(loader.sampler, 'set_epoch'):
                loader.sampler.set_epoch(epoch)

            # 3. Iterate through batches in this specific shard
            for images, labels in loader:
                images = images.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)

                optimizer.zero_grad(set_to_none=True)
                outputs = model(images)

                # Loss calculation logic
                loss_val = 0
                outputs_for_acc = outputs

                # Handle tuple outputs (aux heads)
                if isinstance(outputs, tuple):
                    main_out = outputs[0]
                    outputs_for_acc = main_out
                    if loss_type == "combined":
                        loss_val = criterion(main_out, labels, current_lr)
                        for aux in outputs[1:]:
                            loss_val += 0.3 * criterion(aux, labels, current_lr)
                    else:
                        loss_val = criterion(main_out, labels)
                        for aux in outputs[1:]:
                            loss_val += 0.3 * criterion(aux, labels)
                else:
                    if loss_type == "combined":
                        loss_val = criterion(outputs, labels, current_lr)
                    else:
                        loss_val = criterion(outputs, labels)

                loss_val.backward()
                optimizer.step()

                running_loss += loss_val.item()
                batch_count_epoch += 1

                _, predicted = torch.max(outputs_for_acc, 1)
                correct_train += (predicted == labels).sum().item()
                total_train += labels.size(0)

                if IS_MAIN_PROCESS:
                    avg_loss = running_loss / batch_count_epoch
                    avg_acc = (correct_train / total_train) * 100
                    progress_bar.set_postfix(loss=f"{avg_loss:.4f}", acc=f"{avg_acc:.2f}%")
                    progress_bar.update(1)

        if IS_MAIN_PROCESS:
            progress_bar.close()

        # --- End of Training Epoch ---

        epoch_train_loss = (running_loss / batch_count_epoch) if batch_count_epoch > 0 else 0.0
        epoch_train_acc = (correct_train / max(1, total_train)) * 100.0

        world_size = dist.get_world_size() if ddp and dist.is_initialized() else 1
        val_loss, val_acc, class_stats_val, val_infer_lists, val_metrics = evaluate_model(
            model, val_loader, criterion, genotype_map, log_file, loss_type, current_lr,
            ddp=ddp, world_size=world_size
        )

        val_f1_true = val_metrics.get('f1_true', 0.0)
        val_rec_true = val_metrics.get('recall_true', 0.0)

        print_and_log(
            f"Epoch {epoch + 1} Summary "
            f"| Train: Loss {epoch_train_loss:.4f} Acc {epoch_train_acc:.2f}% "
            f"| Val: Loss {val_loss:.4f} Acc {val_acc:.2f}% F1(True) {val_f1_true:.4f}",
            log_file
        )

        # Checkpointing logic
        improved = (val_f1_true > best_f1_true) or \
                   (val_f1_true == best_f1_true and val_rec_true > best_rec_true) or \
                   (val_f1_true == best_f1_true and val_rec_true == best_rec_true and val_loss < best_val_loss)

        # Regular Save
        if (epoch + 1) == MIN_SAVE_EPOCH and IS_MAIN_PROCESS:
            snap_path = _unique_path(os.path.join(output_path, f"model_epoch_{MIN_SAVE_EPOCH}.pth"))
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': _state_dict(model),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'genotype_map': genotype_map,
                'best_f1_true': best_f1_true
            }, snap_path)

        # Best Save
        if (epoch + 1) >= MIN_SAVE_EPOCH and improved and IS_MAIN_PROCESS:
            best_f1_true = val_f1_true
            best_rec_true = val_rec_true
            best_val_acc = val_acc
            best_val_loss = val_loss

            best_path = _unique_path(os.path.join(output_path, f"model_best_f1_{best_f1_true:.4f}.pth"))
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': _state_dict(model),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'genotype_map': genotype_map,
                'best_f1_true': best_f1_true
            }, best_path)
            print_and_log(f"Saved BEST model to {best_path}", log_file)

        scheduler.step()
        print_and_log("-" * 30, log_file)


def evaluate_model(model, data_loader, criterion, genotype_map, log_file, loss_type, current_lr,
                   ddp=False, world_size=1):
    model.eval()
    batch_count_eval = 0
    num_classes = len(genotype_map) if genotype_map else 0

    correct_eval = torch.zeros(1, device=device, dtype=torch.long)
    total_eval = torch.zeros(1, device=device, dtype=torch.long)
    loss_sum = torch.zeros(1, device=device, dtype=torch.float)

    tp = torch.zeros(num_classes, device=device, dtype=torch.long)
    fp = torch.zeros(num_classes, device=device, dtype=torch.long)
    fn = torch.zeros(num_classes, device=device, dtype=torch.long)
    class_correct = torch.zeros(num_classes, device=device, dtype=torch.long)
    class_total = torch.zeros(num_classes, device=device, dtype=torch.long)

    inference_results = defaultdict(list)
    idx_to_class = {v: k for k, v in genotype_map.items()} if genotype_map else {}

    if not data_loader:
        return 0.0, 0.0, {}, {}, {'f1_true': 0.0, 'recall_true': 0.0}

    with torch.no_grad():
        for batch in data_loader:
            if len(batch) == 3:
                images, labels, paths = batch
            else:
                images, labels = batch
                paths = [""] * labels.size(0)

            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            outputs = model(images)

            if isinstance(outputs, tuple): outputs = outputs[0]

            loss = criterion(outputs, labels, current_lr) if loss_type == "combined" else criterion(outputs, labels)
            loss_sum += loss.detach()
            batch_count_eval += 1

            _, predicted = torch.max(outputs, 1)
            correct_eval += (predicted == labels).sum()
            total_eval += labels.size(0)

            for i in range(labels.size(0)):
                pred_idx = int(predicted[i])
                true_idx = int(labels[i])
                class_total[true_idx] += 1
                if pred_idx == true_idx:
                    class_correct[true_idx] += 1
                    tp[true_idx] += 1
                else:
                    if pred_idx < num_classes: fp[pred_idx] += 1
                    fn[true_idx] += 1

                if idx_to_class and paths[i]:
                    inference_results[idx_to_class.get(pred_idx)].append(os.path.basename(paths[i]))

    if ddp and world_size > 1 and dist.is_initialized():
        dist.all_reduce(correct_eval, op=dist.ReduceOp.SUM)
        dist.all_reduce(total_eval, op=dist.ReduceOp.SUM)
        dist.all_reduce(loss_sum, op=dist.ReduceOp.SUM)
        if num_classes > 0:
            dist.all_reduce(tp, op=dist.ReduceOp.SUM)
            dist.all_reduce(fp, op=dist.ReduceOp.SUM)
            dist.all_reduce(fn, op=dist.ReduceOp.SUM)
            dist.all_reduce(class_correct, op=dist.ReduceOp.SUM)
            dist.all_reduce(class_total, op=dist.ReduceOp.SUM)

    denom = max(1, batch_count_eval * world_size)
    avg_loss = loss_sum.item() / denom
    acc = (correct_eval.item() / max(1, total_eval.item())) * 100.0

    class_stats = {}
    for class_name, idx in genotype_map.items():
        c = int(class_correct[idx].item())
        t = int(class_total[idx].item())
        class_stats[class_name] = {'acc': (c / t * 100) if t > 0 else 0, 'correct': c, 'total': t, 'idx': idx}

    pos_idx = next((idx for name, idx in genotype_map.items() if str(name).lower() == "true"), 1)

    tpc = float(tp[pos_idx].item())
    fpc = float(fp[pos_idx].item())
    fnc = float(fn[pos_idx].item())

    prec = tpc / (tpc + fpc) if (tpc + fpc) > 0 else 0.0
    rec = tpc / (tpc + fnc) if (tpc + fnc) > 0 else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0

    return avg_loss, acc, class_stats, inference_results, {'f1_true': f1, 'recall_true': rec, 'precision_true': prec}


# ---------------- Main ----------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Sharded")
    # Input modes
    parser.add_argument("data_path", nargs="?", type=str, help="Dataset root")
    parser.add_argument("--data_paths", type=str, nargs='+', default=None)
    parser.add_argument("--train_data_paths_file", type=str, default=None)
    parser.add_argument("--val_data_paths_file", type=str, default=None)

    # Params
    parser.add_argument("-o", "--output_path", default="./saved_models_sharded", type=str)
    parser.add_argument("--depths", type=int, nargs='+', default=[3, 3, 27, 3])
    parser.add_argument("--dims", type=int, nargs='+', default=[192, 384, 768, 1536])
    parser.add_argument("--epochs", type=int, default=70)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--prefetch_factor", type=int, default=4)
    parser.add_argument("--mp_context", type=str, default=None)
    parser.add_argument("--warmup_epochs", type=int, default=3)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--save_val_results", action='store_true')
    parser.add_argument("--loss_type", type=str, default="weighted_ce", choices=["combined", "weighted_ce"])
    parser.add_argument("--training_data_ratio", type=float, default=1.0)
    parser.add_argument("--ddp", action="store_true")
    parser.add_argument("--data_parallel", action="store_true")
    parser.add_argument("--local_rank", type=int, default=None)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--pos_weight", type=float, default=88.0)

    args, _ = parser.parse_known_args()

    has_base = args.data_path is not None
    has_files = (args.train_data_paths_file is not None) and (args.val_data_paths_file is not None)
    has_multi = args.data_paths is not None

    if sum([has_base, has_files, has_multi]) != 1:
        print("Error: Provide exactly one input mode (positional, file lists, or multi-roots).")
        sys.exit(1)

    if has_base:
        data_in = os.path.abspath(os.path.expanduser(args.data_path))
    elif has_files:
        tr = _read_paths_file(args.train_data_paths_file)
        va = _read_paths_file(args.val_data_paths_file)
        data_in = (tr, va)
    else:
        data_in = args.data_paths

    # DDP setup
    if args.ddp:
        if not torch.cuda.is_available():
            raise RuntimeError("DDP requires CUDA.")
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
        dist.init_process_group(backend="nccl", init_method="env://")
        IS_MAIN_PROCESS = (dist.get_rank() == 0)
        if IS_MAIN_PROCESS:
            print(f"DDP Init: Size={dist.get_world_size()} Local={local_rank}")
    else:
        local_rank = 0
        IS_MAIN_PROCESS = True
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_model(
        data_path=data_in, output_path=args.output_path,
        save_val_results=args.save_val_results,
        num_epochs=args.epochs, learning_rate=args.lr,
        batch_size=args.batch_size, num_workers=args.num_workers,
        loss_type=args.loss_type, warmup_epochs=args.warmup_epochs,
        weight_decay=args.weight_decay, depths=args.depths, dims=args.dims,
        training_data_ratio=args.training_data_ratio,
        ddp=args.ddp, data_parallel=args.data_parallel, local_rank=local_rank,
        resume=args.resume, pos_weight=args.pos_weight,
        prefetch_factor=args.prefetch_factor,
        mp_context=(None if args.mp_context in (None, "None") else args.mp_context)
    )

    if args.ddp and dist.is_initialized():
        dist.destroy_process_group()