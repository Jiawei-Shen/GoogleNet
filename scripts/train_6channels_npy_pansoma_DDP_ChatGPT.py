#!/usr/bin/env python3
import argparse
import json
import os
import sys
from collections import defaultdict
import glob
import math

import numpy as np
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
from mynet import ConvNeXtCBAMClassifier
from dataset_pansoma_npy_sharded_6ch_DDP_Gemini import get_data_loader  # legacy non-sharded loader

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


# ---------------- Shard helpers (.npy) ----------------
def _discover_shards(split_root):
    """
    Robustly pair shard_*_data.npy and shard_*_labels.npy by base ID.
    Skips shards that don't have both files, and logs them.
    """
    data_paths = sorted(glob.glob(os.path.join(split_root, "shard_*_data.npy")))
    label_paths = sorted(glob.glob(os.path.join(split_root, "shard_*_labels.npy")))

    data_map = {
        os.path.basename(p).replace("_data.npy", ""): p
        for p in data_paths
    }
    label_map = {
        os.path.basename(p).replace("_labels.npy", ""): p
        for p in label_paths
    }

    shard_ids = sorted(set(data_map.keys()) | set(label_map.keys()))
    pairs = []
    missing_data = []
    missing_labels = []

    for sid in shard_ids:
        dp = data_map.get(sid)
        lp = label_map.get(sid)
        if dp is None:
            missing_data.append(sid)
        elif lp is None:
            missing_labels.append(sid)
        else:
            pairs.append((dp, lp))

    if not pairs:
        raise ValueError(f"No valid shard pairs found in {split_root}.")

    print(f"[discover_shards] {split_root}: "
          f"{len(pairs)} valid pairs, {len(missing_data)} missing_data, {len(missing_labels)} missing_labels")
    if missing_data:
        print("  Shards missing data:", ", ".join(missing_data))
    if missing_labels:
        print("  Shards missing labels:", ", ".join(missing_labels))

    return pairs


def _discover_shards_multi(roots, split):
    """
    Discover shards across multiple roots for the given split ("train" or "val").
    Each root is expected to contain <root>/<split>/shard_*_data.npy and shard_*_labels.npy.
    """
    all_pairs = []
    for r in roots:
        base = os.path.abspath(os.path.expanduser(r))
        split_root = os.path.join(base, split)
        if not os.path.isdir(split_root):
            print(f"[discover_shards_multi] WARNING: {split_root} does not exist or is not a directory; skipping.")
            continue
        pairs = _discover_shards(split_root)
        all_pairs.extend(pairs)

    if not all_pairs:
        raise ValueError(f"No valid shard pairs found for split='{split}' in any of roots: {roots}")

    return all_pairs


def _shard_stats(shards, batch_size):
    """
    Compute total samples and total batches across all shards.
    """
    total_samples = 0
    total_batches = 0
    for _, lp in shards:
        y = np.load(lp, mmap_mode="r")
        n = int(y.shape[0])
        total_samples += n
        total_batches += (n + batch_size - 1) // batch_size
    return total_samples, total_batches


def _infer_genotype_map_from_shards(train_shards):
    """
    Infer num_classes and a simple genotype_map from labels in the first shard.
    Binary {0,1} -> {"false":0, "true":1}
    Else -> {"class_0":0, "class_1":1, ...}
    """
    _, labels_path = train_shards[0]
    y = np.load(labels_path, mmap_mode="r")
    uniq = sorted(int(v) for v in np.unique(y))
    genotype_map = {}
    if uniq == [0, 1]:
        genotype_map["false"] = 0
        genotype_map["true"] = 1
    else:
        for cls_idx in uniq:
            genotype_map[f"class_{cls_idx}"] = cls_idx
    return genotype_map


def _iter_sharded_batches(
    shards,
    batch_size,
    device,
    seed=None,
    shuffle_shards=True,
    shuffle_within_shard=True,
):
    """
    Single-process / single-GPU shard iterator:
      - Randomizes shard order each epoch (if shuffle_shards)
      - Randomizes sample order within each shard (if shuffle_within_shard)
      - Yields (x, y) mini-batches of size batch_size.
    """
    rng = np.random.default_rng(seed)
    shard_indices = np.arange(len(shards))
    if shuffle_shards:
        rng.shuffle(shard_indices)

    for si in shard_indices:
        data_path, labels_path = shards[si]
        xs = np.load(data_path, mmap_mode="r")   # (N, C, H, W)
        ys = np.load(labels_path, mmap_mode="r") # (N,)
        n = int(ys.shape[0])
        if n == 0:
            continue

        idxs = np.arange(n)
        if shuffle_within_shard:
            rng.shuffle(idxs)

        for start in range(0, n, batch_size):
            batch_idx = idxs[start:start + batch_size]
            batch_x = torch.from_numpy(xs[batch_idx]).float().to(device, non_blocking=True)
            batch_y = torch.from_numpy(ys[batch_idx]).long().to(device, non_blocking=True)
            yield batch_x, batch_y


def _iter_sharded_batches_ddp(
    shards,
    batch_size,
    device,
    epoch,
    rank,
    world_size,
    shuffle_shards=True,
):
    """
    DDP-safe shard iterator:
      - All ranks use the same shuffled shard order per epoch.
      - For each shard, we make one random permutation of indices (same on all ranks).
      - That permutation is evenly split across ranks (disjoint subsets).
      - Each rank only iterates over its subset, batched by batch_size.
    """
    rng_shards = np.random.default_rng(epoch)
    shard_indices = np.arange(len(shards))
    if shuffle_shards:
        rng_shards.shuffle(shard_indices)

    for si in shard_indices:
        data_path, labels_path = shards[si]
        xs = np.load(data_path, mmap_mode="r")
        ys = np.load(labels_path, mmap_mode="r")
        n = int(ys.shape[0])
        if n == 0:
            continue

        # same permutation on *all* ranks
        seed = epoch * 1000003 + si
        rng = np.random.default_rng(seed)
        idxs = np.arange(n)
        rng.shuffle(idxs)

        # split across ranks
        per_rank = int(math.ceil(n / world_size))
        start = rank * per_rank
        end = min(start + per_rank, n)
        if start >= n:
            continue

        rank_idxs = idxs[start:end]

        for b_start in range(0, len(rank_idxs), batch_size):
            b_sel = rank_idxs[b_start:b_start + batch_size]
            batch_x = torch.from_numpy(xs[b_sel]).float().to(device, non_blocking=True)
            batch_y = torch.from_numpy(ys[b_sel]).long().to(device, non_blocking=True)
            yield batch_x, batch_y


def _iter_sharded_val_batches(shards, batch_size, device):
    """
    Validation iterator: deterministic order, no shuffling, yields (x, y, paths)
    with dummy paths (empty strings).
    """
    for data_path, labels_path in shards:
        xs = np.load(data_path, mmap_mode="r")
        ys = np.load(labels_path, mmap_mode="r")
        n = int(ys.shape[0])
        if n == 0:
            continue

        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            batch_x = torch.from_numpy(xs[start:end]).float().to(device, non_blocking=True)
            batch_y = torch.from_numpy(ys[start:end]).long().to(device, non_blocking=True)
            dummy_paths = [""] * batch_y.size(0)
            yield batch_x, batch_y, dummy_paths


# ---- central helper to build loaders with fast knobs (non-sharded) ----
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


def _build_loader_from_roots(roots, split, batch_size, num_workers, shuffle,
                             prefetch_factor, mp_ctx, return_paths=False):
    """
    Non-sharded Mode B/C helper: multiple roots using get_data_loader and ConcatDataset.
    """
    datasets = []
    genotype_map = None
    for r in roots:
        ld, gm = get_data_loader(
            data_dir=r, dataset_type=split, batch_size=batch_size,  # batch_size here is irrelevant; we rewrap below
            num_workers=num_workers, shuffle=False,  # avoid extra threads; just to obtain the dataset object
            return_paths=return_paths
        )
        ds = getattr(ld, "dataset", None)
        if ds is None:
            continue
        try:
            if len(ds) == 0:
                continue
        except Exception:
            pass

        if genotype_map is None:
            genotype_map = gm
        elif gm != genotype_map:
            raise ValueError(f"Inconsistent genotype_map between roots; offending root: {r}")
        datasets.append(ds)

    if not datasets:
        raise ValueError(f"No datasets found for split='{split}' in provided roots.")

    concat = ConcatDataset(datasets)
    loader = _make_loader(
        dataset=concat,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor,
        multiprocessing_context=mp_ctx,
        pin_memory=True,
        persistent_workers=True,
        sampler=None,
    )
    return loader, genotype_map


def _build_mode_c_loaders(data_paths, batch_size, num_workers, prefetch_factor, mp_ctx):
    """
    Mode C (non-sharded): multiple roots with classic directory layout.
      • Train = union of TRAIN from every root.
      • Val   = union of VAL   from every root (with return_paths=True).
    """
    roots = [os.path.abspath(os.path.expanduser(p)) for p in data_paths]
    train_loader, gm_tr = _build_loader_from_roots(
        roots, "train", batch_size, num_workers, shuffle=True,
        prefetch_factor=prefetch_factor, mp_ctx=mp_ctx, return_paths=False
    )
    val_loader, gm_val = _build_loader_from_roots(
        roots, "val", batch_size, num_workers, shuffle=False,
        prefetch_factor=prefetch_factor, mp_ctx=mp_ctx, return_paths=True
    )
    if gm_val != gm_tr:
        raise ValueError("Inconsistent genotype_map between combined train and combined val across roots.")
    return train_loader, val_loader, gm_tr


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

    if not (0 < training_data_ratio <= 1.0):
        raise ValueError(f"--training_data_ratio must be in (0,1], got {training_data_ratio}")

    print_and_log(f"Using device: {device}", log_file)
    print_and_log(f"Initial Learning Rate: {learning_rate:.1e}", log_file)
    print_and_log(f"Using CosineAnnealing with {warmup_epochs} warmup epochs.", log_file)
    print_and_log(
        f"DataLoader (non-sharded): workers={num_workers}, pin_memory=True, "
        f"persistent_workers=True, prefetch_factor={prefetch_factor}, mp_ctx={mp_context}", log_file
    )

    # ---- DDP rank/world size ----
    if ddp and dist.is_initialized():
        world_size = dist.get_world_size()
        rank = dist.get_rank()
    else:
        world_size = 1
        rank = 0

    # ---------------- Detect shard-based mode ----------------
    use_sharded_mode = False
    train_shards = None
    val_shards = None
    num_train_batches = None

    candidate_roots = None
    if isinstance(data_path, str):
        candidate_roots = [data_path]
    elif isinstance(data_path, (list, tuple)) and all(isinstance(p, str) for p in data_path):
        # Mode C multi-root: list of roots, all strings
        candidate_roots = list(data_path)

    if candidate_roots is not None:
        for r in candidate_roots:
            train_root_candidate = os.path.join(os.path.abspath(os.path.expanduser(r)), "train")
            pattern = os.path.join(train_root_candidate, "shard_*_data.npy")
            if os.path.isdir(train_root_candidate) and glob.glob(pattern):
                use_sharded_mode = True
                break

    # ---------------- Build loaders or shard sets ----------------
    if use_sharded_mode:
        roots = [os.path.abspath(os.path.expanduser(r)) for r in candidate_roots]

        train_shards = _discover_shards_multi(roots, split="train")
        val_shards = _discover_shards_multi(roots, split="val")

        total_train_samples, num_train_batches = _shard_stats(train_shards, batch_size)
        genotype_map = _infer_genotype_map_from_shards(train_shards)

        print_and_log(f"Shard-based training enabled (multi-root OK).", log_file)
        print_and_log(f"  Roots        : {len(roots)}", log_file)
        print_and_log(f"  Train shards : {len(train_shards)} (total samples={total_train_samples})", log_file)
        print_and_log(f"  Val shards   : {len(val_shards)}", log_file)
        print_and_log(f"  Approx train batches/epoch: {num_train_batches}", log_file)

        if training_data_ratio != 1.0:
            print_and_log(f"WARNING: training_data_ratio != 1.0 ignored in shard mode (using full data).", log_file)

        train_loader = None
        val_loader = None
    else:
        # ---- Original non-sharded logic (Modes A/B/C with NpyDataset) ----
        if isinstance(data_path, str):
            # Mode A: single root
            ld_tr, genotype_map = get_data_loader(
                data_dir=data_path, dataset_type="train", batch_size=batch_size,
                num_workers=num_workers, shuffle=False  # get dataset only
            )
            ld_va, _ = get_data_loader(
                data_dir=data_path, dataset_type="val", batch_size=batch_size,
                num_workers=num_workers, shuffle=False, return_paths=True
            )
            train_loader = _make_loader(
                ld_tr.dataset, batch_size, True, num_workers,
                prefetch_factor=prefetch_factor, multiprocessing_context=mp_context,
                pin_memory=True, persistent_workers=True
            )
            val_loader = _make_loader(
                ld_va.dataset, batch_size, False, num_workers,
                prefetch_factor=prefetch_factor, multiprocessing_context=mp_context,
                pin_memory=True, persistent_workers=True
            )
        elif isinstance(data_path, tuple) and len(data_path) == 2:
            # Mode B: explicit (train_roots, val_roots)
            train_roots, val_roots = data_path
            train_loader, genotype_map = _build_loader_from_roots(
                train_roots, "train", batch_size, num_workers, True,
                prefetch_factor=prefetch_factor, mp_ctx=mp_context, return_paths=False
            )
            val_loader, gm_val = _build_loader_from_roots(
                val_roots, "val", batch_size, num_workers, False,
                prefetch_factor=prefetch_factor, mp_ctx=mp_context, return_paths=True
            )
            if gm_val != genotype_map:
                raise ValueError("Inconsistent genotype_map between train_roots and val_roots.")
        elif isinstance(data_path, (list, tuple)):
            # Mode C: multiple roots, classic layout
            train_loader, val_loader, genotype_map = _build_mode_c_loaders(
                data_path, batch_size=batch_size, num_workers=num_workers,
                prefetch_factor=prefetch_factor, mp_ctx=mp_context
            )
            print_and_log("Mode C (non-sharded): Train=union(train), Val=union(val) across all roots.", log_file)
        else:
            raise ValueError("Unsupported data_path type.")

        # Optional subsample (DDP-safe, no collectives)
        if training_data_ratio < 1.0:
            full_ds = train_loader.dataset
            n = len(full_ds)
            k = max(1, int(round(n * training_data_ratio)))

            gen = torch.Generator()
            gen.manual_seed(123456)
            idx_list = torch.randperm(n, generator=gen)[:k].tolist()

            subset = Subset(full_ds, idx_list)
            train_loader = _make_loader(
                subset, batch_size, True, num_workers,
                prefetch_factor=prefetch_factor, multiprocessing_context=mp_context,
                pin_memory=True, persistent_workers=True
            )
            print_and_log(
                f"Training subset: using {len(subset)}/{n} samples (~{training_data_ratio:.2f} of data).", log_file
            )

    # ---- Model / parallelism ----
    if not genotype_map:
        print_and_log("Error: genotype_map is empty. Check dataloader or shard labels.", log_file)
        return
    num_classes = len(genotype_map)
    print_and_log(f"Number of classes: {num_classes}", log_file)

    model = ConvNeXtCBAMClassifier(in_channels=6, class_num=num_classes, depths=depths, dims=dims).to(device)

    if (not ddp) and data_parallel and torch.cuda.is_available():
        n = torch.cuda.device_count()
        if n > 1:
            print_and_log(f"DataParallel across {n} GPUs.", log_file)
            model = nn.DataParallel(model)
        else:
            print_and_log("DataParallel requested but single GPU detected; running single-GPU.", log_file)

    train_sampler = None
    if ddp:
        print_and_log(f"Wrapping model in DistributedDataParallel on cuda:{local_rank}.", log_file)
        model = DistributedDataParallel(
            model, device_ids=[local_rank], output_device=local_rank,
            gradient_as_bucket_view=True, broadcast_buffers=False,
        )
        if (not use_sharded_mode) and (train_loader is not None) and (val_loader is not None):
            train_dataset = train_loader.dataset
            val_dataset = val_loader.dataset
            train_sampler = DistributedSampler(train_dataset, shuffle=True, drop_last=False)
            val_sampler = DistributedSampler(val_dataset, shuffle=False, drop_last=False)
            train_loader = _make_loader(
                train_dataset, batch_size, True, num_workers,
                prefetch_factor=prefetch_factor, multiprocessing_context=mp_context,
                pin_memory=True, persistent_workers=True, sampler=train_sampler
            )
            val_loader = _make_loader(
                val_dataset, batch_size, False, num_workers,
                prefetch_factor=prefetch_factor, multiprocessing_context=mp_context,
                pin_memory=True, persistent_workers=True, sampler=val_sampler
            )

    model.apply(init_weights)

    # ---- Loss / Optim / Sched ----
    pos_weight_value = float(pos_weight)
    class_weights = torch.ones(num_classes, device=device, dtype=torch.float32)
    if num_classes >= 2:
        class_weights[1] = pos_weight_value

    print_and_log(f"Class weights: {class_weights.tolist()} (from --pos_weight={pos_weight_value})", log_file)

    if loss_type == "combined":
        criterion = CombinedFocalWeightedCELoss(initial_lr=learning_rate, pos_weight=class_weights)
        print_and_log("Using Combined(Focal + Weighted CE) Loss.", log_file)
    elif loss_type == "weighted_ce":
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        print_and_log("Using Weighted CE Loss.", log_file)
    else:
        raise ValueError(f"Unsupported loss_type: {loss_type}")

    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    warmup_scheduler = LinearLR(optimizer, start_factor=0.01, total_iters=warmup_epochs)
    main_scheduler = CosineAnnealingLR(optimizer, T_max=max(1, num_epochs - warmup_epochs), eta_min=0)
    scheduler = SequentialLR(optimizer, schedulers=[warmup_scheduler, main_scheduler], milestones=[warmup_epochs])

    # ---- Resume ----
    start_epoch = 0
    best_epoch = 0
    best_f1_true = float("-inf")
    best_val_acc = float("-inf")
    best_val_loss = float("inf")
    best_rec_true = 0.0
    last_best_ckpt_path = None

    if resume is not None and os.path.isfile(resume):
        try:
            checkpoint = torch.load(resume, map_location=device)
            _load_state_dict(model, checkpoint['model_state_dict'])
            if 'optimizer_state_dict' in checkpoint:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            if 'scheduler_state_dict' in checkpoint:
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            start_epoch = int(checkpoint.get('epoch', 0))
            best_f1_true = float(checkpoint.get('best_f1_true', best_f1_true))
            best_rec_true = float(checkpoint.get('best_rec_true', best_rec_true))
            best_val_acc = float(checkpoint.get('best_val_acc', best_val_acc))
            best_val_loss = float(checkpoint.get('best_val_loss', best_val_loss))
            best_epoch = int(checkpoint.get('epoch', best_epoch))
            print_and_log(f"Resumed from '{resume}' at epoch {start_epoch}.", log_file)
        except Exception as e:
            print_and_log(f"WARNING: Failed to load checkpoint '{resume}': {e}", log_file)

    sorted_class_names_from_map = sorted(genotype_map.keys(), key=lambda k: genotype_map[k])

    # ---------------- Train loop ----------------
    for epoch in range(start_epoch, num_epochs):
        model.train()
        if ddp and train_sampler is not None and (not use_sharded_mode):
            train_sampler.set_epoch(epoch)

        running_loss = 0.0
        correct_train = 0
        total_train = 0
        batch_count = 0

        current_lr = optimizer.param_groups[0]['lr']

        # Choose training iterator depending on mode + DDP
        if use_sharded_mode:
            if ddp and world_size > 1:
                train_iter = _iter_sharded_batches_ddp(
                    train_shards,
                    batch_size=batch_size,
                    device=device,
                    epoch=epoch,
                    rank=rank,
                    world_size=world_size,
                    shuffle_shards=True,
                )
            else:
                train_iter = _iter_sharded_batches(
                    train_shards,
                    batch_size=batch_size,
                    device=device,
                    seed=epoch,
                    shuffle_shards=True,
                    shuffle_within_shard=True,
                )

            progress_bar = tqdm(
                train_iter,
                desc=f"Epoch {epoch + 1}/{num_epochs} LR: {current_lr:.1e}",
                leave=True,
                disable=not IS_MAIN_PROCESS,
            )
        else:
            train_iter = train_loader
            progress_bar = tqdm(
                train_iter,
                desc=f"Epoch {epoch + 1}/{num_epochs} LR: {current_lr:.1e}",
                leave=True,
                disable=not IS_MAIN_PROCESS,
            )

        for images, labels in progress_bar:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            outputs = model(images)

            if loss_type == "combined":
                if isinstance(outputs, tuple) and len(outputs) == 3:
                    main_output, aux1, aux2 = outputs
                    loss = (criterion(main_output, labels, current_lr)
                            + 0.3 * criterion(aux1, labels, current_lr)
                            + 0.3 * criterion(aux2, labels, current_lr))
                    outputs_for_acc = main_output
                else:
                    loss = criterion(outputs, labels, current_lr)
                    outputs_for_acc = outputs
            else:
                if isinstance(outputs, tuple) and len(outputs) == 3:
                    main_output, aux1, aux2 = outputs
                    loss = (criterion(main_output, labels)
                            + 0.3 * criterion(aux1, labels)
                            + 0.3 * criterion(aux2, labels))
                    outputs_for_acc = main_output
                else:
                    loss = criterion(outputs, labels)
                    outputs_for_acc = outputs

            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            batch_count += 1
            _, predicted = torch.max(outputs_for_acc, 1)
            correct_train += (predicted == labels).sum().item()
            total_train += labels.size(0)

            if IS_MAIN_PROCESS and total_train > 0 and batch_count > 0:
                avg_loss_train = running_loss / batch_count
                avg_acc_train = (correct_train / total_train) * 100.0
                progress_bar.set_postfix(loss=f"{avg_loss_train:.4f}", acc=f"{avg_acc_train:.2f}%")

        epoch_train_loss = (running_loss / max(1, batch_count))
        epoch_train_acc = (correct_train / max(1, total_train)) * 100.0

        # Build validation iterator
        if use_sharded_mode:
            val_iter_for_eval = _iter_sharded_val_batches(
                val_shards, batch_size=batch_size, device=device
            )
        else:
            val_iter_for_eval = val_loader

        val_loss, val_acc, class_stats_val, val_infer_lists, val_metrics = evaluate_model(
            model, val_iter_for_eval, criterion, genotype_map, log_file, loss_type, current_lr,
            ddp=ddp, world_size=world_size
        )

        if IS_MAIN_PROCESS and class_stats_val:
            print_and_log("\nClass-wise Validation Accuracy:", log_file)
            for class_name in sorted_class_names_from_map:
                s = class_stats_val.get(class_name, {})
                print_and_log(
                    f"  {class_name} (idx {s.get('idx','N/A')}): {s.get('acc',0):.2f}% "
                    f"({s.get('correct',0)}/{s.get('total',0)})", log_file)

        val_prec_true = val_metrics.get('precision_true', 0.0)
        val_rec_true = val_metrics.get('recall_true', 0.0)
        val_f1_true = val_metrics.get('f1_true', 0.0)
        pos_idx = val_metrics.get('pos_class_idx', None)

        print_and_log(
            f"Epoch {epoch + 1}/{num_epochs} "
            f"| Train Loss {epoch_train_loss:.4f} Acc {epoch_train_acc:.2f}% "
            f"| Val Loss {val_loss:.4f} Acc {val_acc:.2f}% "
            f"| Prec(true) {val_prec_true*100:.2f}% Rec(true) {val_rec_true*100:.2f}% F1(true) {val_f1_true:.4f} "
            f"(LR {current_lr:.1e}{', pos_idx='+str(pos_idx) if pos_idx is not None else ''})",
            log_file
        )

        improved = (val_f1_true > best_f1_true) or \
                   (val_f1_true == best_f1_true and val_rec_true > best_rec_true) or \
                   (val_f1_true == best_f1_true and val_rec_true == best_rec_true and val_loss < best_val_loss)

        if (epoch + 1) == MIN_SAVE_EPOCH and IS_MAIN_PROCESS:
            snap_path = _unique_path(os.path.join(output_path, f"model_epoch_{MIN_SAVE_EPOCH}.pth"))
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': _state_dict(model),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'genotype_map': genotype_map,
                'in_channels': 6
            }, snap_path)
            print_and_log(f"Snapshot saved at epoch {epoch + 1}: {snap_path}", log_file)

        if (epoch + 1) >= MIN_SAVE_EPOCH and improved and IS_MAIN_PROCESS:
            best_f1_true = val_f1_true
            best_rec_true = val_rec_true
            best_val_acc = val_acc
            best_val_loss = val_loss
            best_epoch = epoch + 1

            best_path = _unique_path(os.path.join(output_path, f"model_e{best_epoch:03d}_f1_{best_f1_true:.4f}.pth"))
            payload = {
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
            }
            torch.save(payload, best_path)
            last_best_ckpt_path = best_path
            print_and_log(
                f"New BEST @epoch {best_epoch}: F1(true) {best_f1_true:.4f} | "
                f"Rec(true) {best_rec_true*100:.2f}% | Val Acc {best_val_acc:.2f}% | "
                f"Val Loss {best_val_loss:.4f}. Saved: {best_path}", log_file
            )

            if save_val_results:
                result_path = _unique_path(os.path.join(
                    output_path, f"validation_results_e{best_epoch:03d}_f1_{best_f1_true:.4f}.json"
                ))
                try:
                    with open(result_path, 'w') as f:
                        json.dump({
                            'epoch': best_epoch,
                            'f1_true': best_f1_true,
                            'recall_true': best_rec_true,
                            'val_acc': best_val_acc,
                            'val_loss': best_val_loss,
                            'inference_results': val_infer_lists
                        }, f, indent=4)
                    print_and_log(f"Saved best validation results to {result_path}", log_file)
                except Exception as e:
                    print_and_log(f"Error saving best validation results: {e}", log_file)

        scheduler.step()
        print_and_log("-" * 30, log_file)

    final_msg = (
        f"Training complete. Best epoch: {best_epoch} "
        f"| F1(true) {best_f1_true:.4f} | Rec(true) {best_rec_true*100:.2f}% "
        f"| Val Acc {best_val_acc:.2f}% | Val Loss {best_val_loss:.4f}. "
        f"{'Best model: '+last_best_ckpt_path if last_best_ckpt_path else 'No best checkpoint saved.'}"
    )
    print_and_log(final_msg, log_file)


def evaluate_model(model, data_iter, criterion, genotype_map, log_file, loss_type, current_lr,
                   ddp=False, world_size=1):
    model.eval()
    batch_count_eval = 0
    num_classes = len(genotype_map) if genotype_map else 0

    # global accumulators on device
    correct_eval = torch.zeros(1, device=device, dtype=torch.long)
    total_eval = torch.zeros(1, device=device, dtype=torch.long)
    loss_sum = torch.zeros(1, device=device, dtype=torch.float)

    tp = torch.zeros(num_classes, device=device, dtype=torch.long)
    fp = torch.zeros(num_classes, device=device, dtype=torch.long)
    fn = torch.zeros(num_classes, device=device, dtype=torch.long)
    class_correct_counts = torch.zeros(num_classes, device=device, dtype=torch.long)
    class_total_counts = torch.zeros(num_classes, device=device, dtype=torch.long)

    inference_results = defaultdict(list)
    idx_to_class = {v: k for k, v in genotype_map.items()} if genotype_map else {}

    # allow any iterable / generator
    if data_iter is None:
        metrics = {'precision_true': 0.0, 'recall_true': 0.0, 'f1_true': 0.0, 'pos_class_idx': None}
        return 0.0, 0.0, {}, {}, metrics

    with torch.no_grad():
        for batch in data_iter:
            if len(batch) == 3:
                images, labels, paths = batch
            else:
                images, labels = batch
                paths = [""] * labels.size(0)

            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            outputs = model(images)
            if isinstance(outputs, tuple):
                outputs = outputs[0]

            loss = criterion(outputs, labels, current_lr) if loss_type == "combined" else criterion(outputs, labels)
            loss_sum += loss.detach()
            batch_count_eval += 1

            _, predicted = torch.max(outputs, 1)
            correct_eval += (predicted == labels).sum()
            total_eval += labels.size(0)

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

    if ddp and world_size > 1 and dist.is_initialized():
        dist.all_reduce(correct_eval, op=dist.ReduceOp.SUM)
        dist.all_reduce(total_eval, op=dist.ReduceOp.SUM)
        dist.all_reduce(loss_sum, op=dist.ReduceOp.SUM)
        if num_classes > 0:
            dist.all_reduce(tp, op=dist.ReduceOp.SUM)
            dist.all_reduce(fp, op=dist.ReduceOp.SUM)
            dist.all_reduce(fn, op=dist.ReduceOp.SUM)
            dist.all_reduce(class_correct_counts, op=dist.ReduceOp.SUM)
            dist.all_reduce(class_total_counts, op=dist.ReduceOp.SUM)

    denom_batches = max(1, batch_count_eval * (world_size if (ddp and world_size > 1) else 1))
    avg_loss_eval = (loss_sum.item() / denom_batches)
    overall_accuracy_eval = (correct_eval.item() / max(1, total_eval.item())) * 100.0

    class_performance_stats = {}
    if genotype_map:
        for class_name, class_idx in genotype_map.items():
            correct_c = int(class_correct_counts[class_idx].item())
            total_c = int(class_total_counts[class_idx].item())
            acc_c = (correct_c / total_c * 100.0) if total_c > 0 else 0.0
            class_performance_stats[class_name] = {
                'acc': acc_c, 'correct': correct_c, 'total': total_c, 'idx': class_idx
            }

    # choose positive class index
    pos_idx = None
    if genotype_map:
        for name, idx in genotype_map.items():
            if str(name).lower() == "true":
                pos_idx = idx
                break
    if pos_idx is None:
        pos_idx = 1 if num_classes > 1 else 0

    tpc = float(tp[pos_idx].item() if pos_idx < num_classes else 0.0)
    fpc = float(fp[pos_idx].item() if pos_idx < num_classes else 0.0)
    fnc = float(fn[pos_idx].item() if pos_idx < num_classes else 0.0)

    precision_true = (tpc / (tpc + fpc)) if (tpc + fpc) > 0 else 0.0
    recall_true = (tpc / (tpc + fnc)) if (tpc + fnc) > 0 else 0.0
    f1_true = (2 * precision_true * recall_true / (precision_true + recall_true)) if (precision_true + recall_true) > 0 else 0.0

    metrics = {
        'precision_true': precision_true,
        'recall_true': recall_true,
        'f1_true': f1_true,
        'pos_class_idx': pos_idx,
    }
    return avg_loss_eval, overall_accuracy_eval, class_performance_stats, inference_results, metrics


# ---------------- Main ----------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a Classifier on 6-channel custom .npy dataset (sharded or classic)")

    # Input modes
    parser.add_argument("data_path", nargs="?", type=str,
                        help="Dataset root containing 'train/' and 'val/' (Mode A, classic or sharded).")
    parser.add_argument("--data_paths", type=str, nargs='+', default=None,
                        help="MULTI roots (each contains 'train/' and 'val/'). "
                             "Mode C: Train=union(train), Val=union(val) or shard mode.")
    parser.add_argument("--train_data_paths_file", type=str, default=None,
                        help="Text file listing TRAIN dataset roots (one per line). (Mode B, classic only)")
    parser.add_argument("--val_data_paths_file", type=str, default=None,
                        help="Text file listing VAL dataset roots (one per line). (Mode B, classic only)")

    # Model / train
    parser.add_argument("-o", "--output_path", default="./saved_models_6channel", type=str, help="Path to save model")
    parser.add_argument("--depths", type=int, nargs='+', default=[3, 3, 27, 3])
    parser.add_argument("--dims", type=int, nargs='+', default=[192, 384, 768, 1536])
    parser.add_argument("--epochs", type=int, default=70)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch_size", type=int, default=32)

    # Loader knobs
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--prefetch_factor", type=int, default=4)
    parser.add_argument("--mp_context", type=str, default=None, choices=[None, "fork", "forkserver", "spawn"])

    # Optimizer / scheduler
    parser.add_argument("--warmup_epochs", type=int, default=3)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--save_val_results", action='store_true')
    parser.add_argument("--loss_type", type=str, default="weighted_ce", choices=["combined", "weighted_ce"])

    # Subsample (non-sharded only)
    parser.add_argument("--training_data_ratio", type=float, default=1.0)

    # Parallel
    parser.add_argument("--ddp", action="store_true")
    parser.add_argument("--data_parallel", action="store_true")
    parser.add_argument("--local_rank", type=int, default=None, help="(Ignored) Torch launcher may pass this.")

    # Resume / weights
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--pos_weight", type=float, default=88.0)

    args, _unknown = parser.parse_known_args()

    has_base = args.data_path is not None
    has_both_files = (args.train_data_paths_file is not None) and (args.val_data_paths_file is not None)
    has_multi = args.data_paths is not None

    if not (0 < args.training_data_ratio <= 1.0):
        parser.error(f"--training_data_ratio must be in (0,1], got {args.training_data_ratio}")

    if sum([has_base, has_both_files, has_multi]) != 1:
        parser.error("Provide exactly one input mode:\n"
                     "  • Mode A: positional data_path\n"
                     "  • Mode B: --train_data_paths_file and --val_data_paths_file\n"
                     "  • Mode C: --data_paths root1 root2 ...")

    # Build param for train_model
    if has_base:
        data_path_or_pair = os.path.abspath(os.path.expanduser(args.data_path))
    elif has_both_files:
        train_roots = _read_paths_file(args.train_data_paths_file)
        val_roots = _read_paths_file(args.val_data_paths_file)
        if not train_roots:
            parser.error(f"--train_data_paths_file is empty or unreadable: {args.train_data_paths_file}")
        if not val_roots:
            parser.error(f"--val_data_paths_file is empty or unreadable: {args.val_data_paths_file}")
        data_path_or_pair = (train_roots, val_roots)
    else:
        if len(args.data_paths) < 1:
            parser.error("--data_paths needs at least one root.")
        data_path_or_pair = args.data_paths

    # DDP init
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

    train_model(
        data_path=data_path_or_pair, output_path=args.output_path,
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
