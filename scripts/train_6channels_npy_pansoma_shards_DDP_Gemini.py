#!/usr/bin/env python3
import argparse
import json
import os
import sys
import glob
import random
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel
from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision import transforms
from tqdm import tqdm

# ---- env + backend knobs ----
os.environ.setdefault("OMP_NUM_THREADS", "1")
torch.backends.cudnn.benchmark = True

# local imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
try:
    from mynet import ConvNeXtCBAMClassifier
    # Ensure your dataset file is named 'dataset_pansoma_npy_6ch.py' or update this import
    from dataset_pansoma_npy_6ch import get_data_loader, NpyDataset
except ImportError:
    print("Warning: Local imports failed. Ensure 'mynet' and dataset file are in python path.")
    pass

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IS_MAIN_PROCESS = True


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
        if self.reduction == 'mean': return focal_loss.mean()
        if self.reduction == 'sum': return focal_loss.sum()
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
        if m.bias is not None: nn.init.constant_(m.bias, 0)


def print_and_log(message, log_path):
    if not IS_MAIN_PROCESS: return
    print(message, flush=True)
    with open(log_path, 'a', encoding='utf-8') as f: f.write(message + '\n')


def _state_dict(m):
    return m.module.state_dict() if hasattr(m, "module") else m.state_dict()


def _load_state_dict(m, state):
    if hasattr(m, "module"):
        m.module.load_state_dict(state)
    else:
        m.load_state_dict(state)


def _unique_path(path: str) -> str:
    if not os.path.exists(path): return path
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
            if s and not s.startswith('#'):
                paths.append(os.path.abspath(os.path.expanduser(s)))
    return paths


# ---------------- Loader Builders ----------------
def _get_normalization_transform():
    return transforms.Compose([
        transforms.Normalize(
            mean=[18.417816162109375, 12.649129867553711, -0.5452527403831482,
                  24.723854064941406, 4.690611362457275, 0.2813551473402196],
            std=[25.028322219848633, 14.809632301330566, 0.6181337833404541,
                 29.972835540771484, 7.9231791496276855, 0.7659083659074717]
        )
    ])


def _find_individual_shard_files(roots, subfolder="train"):
    """Recursively finds all *_data.npy files."""
    shard_files = []
    for r in roots:
        r = os.path.abspath(os.path.expanduser(r))
        # Case 1: Root itself has files
        direct_shards = glob.glob(os.path.join(r, "*_data.npy"))
        if direct_shards:
            shard_files.extend(direct_shards)
            continue
        # Case 2: Root has 'train' folder
        target = os.path.join(r, subfolder)
        if os.path.exists(target):
            sub_shards = glob.glob(os.path.join(target, "*_data.npy"))
            shard_files.extend(sub_shards)
    return sorted(list(set(shard_files)))


def _build_list_of_shard_loaders(shard_files, batch_size, num_workers, prefetch_factor, mp_ctx, ddp=False):
    loaders = []
    transform = _get_normalization_transform()
    genotype_map = None

    # print_and_log(f"Building {len(shard_files)} individual shard loaders (RAM Preload Enabled)...", "stdout")

    for fpath in shard_files:
        try:
            # --- ENABLE RAM PRELOAD HERE ---
            # This uses the NpyDataset class directly to control preloading
            ds = NpyDataset(root_dir=fpath, transform=transform, return_paths=False, preload_ram=True)
        except Exception as e:
            # This catches missing label files (like shard_61) and skips them safely
            if IS_MAIN_PROCESS:
                print(f"WARNING: Skipping bad shard {os.path.basename(fpath)}: {e}")
            continue

        if genotype_map is None:
            genotype_map = ds.class_to_idx

        sampler = None
        if ddp:
            sampler = DistributedSampler(ds, shuffle=True, drop_last=False)

        loader = DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=(sampler is None),
            sampler=sampler,
            num_workers=num_workers,
            prefetch_factor=prefetch_factor if num_workers > 0 else None,
            multiprocessing_context=mp_ctx,
            pin_memory=True,
            persistent_workers=(num_workers > 0)
        )
        loaders.append(loader)

    return loaders, genotype_map


# ---------------- Train / Eval ----------------
def train_model(data_path, output_path, save_val_results=False, num_epochs=100, learning_rate=1e-4,
                batch_size=32, num_workers=4, loss_type='weighted_ce',
                warmup_epochs=10, weight_decay=0.05, depths=None, dims=None,
                training_data_ratio=1.0, ddp=False, data_parallel=False, local_rank=0,
                resume=None, pos_weight=88.0,
                prefetch_factor=4, mp_context=None):
    os.makedirs(output_path, exist_ok=True)
    log_file = os.path.join(output_path, "training_log.txt")
    if os.path.exists(log_file) and IS_MAIN_PROCESS: os.remove(log_file)

    MIN_SAVE_EPOCH = 5

    print_and_log(f"Device: {device} | LR: {learning_rate} | Epochs: {num_epochs}", log_file)
    print_and_log("Strategy: Shard-by-Shard (Preloaded into RAM).", log_file)

    # 1. Parse Inputs
    if isinstance(data_path, (list, tuple)) and len(data_path) == 2:
        train_roots_raw = data_path[0] if isinstance(data_path[0], list) else [data_path[0]]
        val_roots_raw = data_path[1]
    elif isinstance(data_path, list):
        train_roots_raw = data_path
        val_roots_raw = data_path
    else:
        train_roots_raw = [data_path]
        val_roots_raw = [data_path]

    # 2. Build Training Loaders
    train_shard_files = _find_individual_shard_files(train_roots_raw, subfolder="train")
    if not train_shard_files:
        raise ValueError(f"No *_data.npy files found in train roots: {train_roots_raw}")

    print_and_log(f"Found {len(train_shard_files)} shards. Loading into RAM...", log_file)

    train_loaders, genotype_map = _build_list_of_shard_loaders(
        train_shard_files, batch_size, num_workers, prefetch_factor, mp_context, ddp
    )

    if not train_loaders:
        raise RuntimeError("All shards failed to load! Check your data paths.")

    # 3. Build Validation Loader (Standard, no preload)
    val_loader, gm_val = get_data_loader(
        val_roots_raw, "val", batch_size, num_workers, False, return_paths=True
    )

    if genotype_map is None: genotype_map = gm_val

    total_batches = sum([len(l) for l in train_loaders])
    print_and_log(f"Train: {len(train_loaders)} valid shards | {total_batches} batches/epoch.", log_file)

    # 4. Model
    num_classes = len(genotype_map)
    model = ConvNeXtCBAMClassifier(in_channels=6, class_num=num_classes, depths=depths, dims=dims).to(device)

    if (not ddp) and data_parallel and torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    if ddp:
        model = DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank)

    model.apply(init_weights)

    # 5. Optimizer
    class_weights = torch.ones(num_classes, device=device)
    if num_classes >= 2: class_weights[1] = float(pos_weight)

    if loss_type == "combined":
        criterion = CombinedFocalWeightedCELoss(learning_rate, class_weights)
    else:
        criterion = nn.CrossEntropyLoss(weight=class_weights)

    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    warmup_scheduler = LinearLR(optimizer, start_factor=0.01, total_iters=warmup_epochs)
    main_scheduler = CosineAnnealingLR(optimizer, T_max=max(1, num_epochs - warmup_epochs), eta_min=0)
    scheduler = SequentialLR(optimizer, schedulers=[warmup_scheduler, main_scheduler], milestones=[warmup_epochs])

    # 6. Resume
    start_epoch = 0
    best_f1 = -1.0
    best_val_loss = float('inf')
    if resume and os.path.isfile(resume):
        ckpt = torch.load(resume, map_location=device)
        _load_state_dict(model, ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        start_epoch = int(ckpt.get('epoch', 0))
        best_f1 = float(ckpt.get('best_f1_true', -1.0))
        print_and_log(f"Resumed from epoch {start_epoch}", log_file)

    # 7. Loop
    for epoch in range(start_epoch, num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        batches_done = 0
        current_lr = optimizer.param_groups[0]['lr']

        g = torch.Generator()
        g.manual_seed(epoch + 1000)
        perm = torch.randperm(len(train_loaders), generator=g).tolist()

        pbar = tqdm(total=total_batches, desc=f"Ep {epoch + 1}", disable=not IS_MAIN_PROCESS)

        for shard_idx in perm:
            loader = train_loaders[shard_idx]
            if ddp: loader.sampler.set_epoch(epoch)

            for images, labels in loader:
                images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)

                optimizer.zero_grad(set_to_none=True)
                out = model(images)

                if isinstance(out, tuple):
                    main_out = out[0]
                    loss = criterion(main_out, labels, current_lr) if loss_type == "combined" else criterion(main_out,
                                                                                                             labels)
                    for aux in out[1:]:
                        loss += 0.3 * (
                            criterion(aux, labels, current_lr) if loss_type == "combined" else criterion(aux, labels))
                    out_acc = main_out
                else:
                    loss = criterion(out, labels, current_lr) if loss_type == "combined" else criterion(out, labels)
                    out_acc = out

                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                _, pred = torch.max(out_acc, 1)
                correct += (pred == labels).sum().item()
                total += labels.size(0)
                batches_done += 1

                if IS_MAIN_PROCESS:
                    pbar.set_postfix(loss=f"{running_loss / batches_done:.4f}", acc=f"{correct / total * 100:.2f}%")
                    pbar.update(1)

        if IS_MAIN_PROCESS: pbar.close()

        epoch_loss = running_loss / max(1, batches_done)
        epoch_acc = correct / max(1, total) * 100

        world_size = dist.get_world_size() if ddp and dist.is_initialized() else 1
        val_loss, val_acc, _, _, val_metrics = evaluate_model(
            model, val_loader, criterion, genotype_map, log_file, loss_type, current_lr, ddp, world_size
        )

        f1 = val_metrics.get('f1_true', 0.0)

        print_and_log(
            f"Ep {epoch + 1} | Tr Loss {epoch_loss:.4f} Acc {epoch_acc:.1f}% | Val Loss {val_loss:.4f} Acc {val_acc:.1f}% F1 {f1:.4f}",
            log_file)

        improved = (f1 > best_f1) or (f1 == best_f1 and val_loss < best_val_loss)
        if IS_MAIN_PROCESS:
            state = {
                'epoch': epoch + 1, 'model_state_dict': _state_dict(model),
                'optimizer_state_dict': optimizer.state_dict(), 'scheduler_state_dict': scheduler.state_dict(),
                'best_f1_true': best_f1, 'genotype_map': genotype_map
            }
            if (epoch + 1) == MIN_SAVE_EPOCH:
                torch.save(state, _unique_path(os.path.join(output_path, f"model_epoch_{MIN_SAVE_EPOCH}.pth")))
            if (epoch + 1) >= MIN_SAVE_EPOCH and improved:
                best_f1 = f1
                best_val_loss = val_loss
                path = _unique_path(os.path.join(output_path, f"model_best_f1_{best_f1:.4f}.pth"))
                torch.save(state, path)
                print_and_log(f"Saved Best: {path}", log_file)

        scheduler.step()


def evaluate_model(model, loader, criterion, map, log, loss_type, lr, ddp, world_size):
    model.eval()
    loss_sum, corr, tot = torch.zeros(3, device=device)
    tp = torch.zeros(len(map), device=device)
    fp = torch.zeros(len(map), device=device)
    fn = torch.zeros(len(map), device=device)

    with torch.no_grad():
        for batch in loader:
            imgs, lbls = batch[0].to(device), batch[1].to(device)
            out = model(imgs)
            if isinstance(out, tuple): out = out[0]

            l = criterion(out, lbls, lr) if loss_type == "combined" else criterion(out, lbls)
            loss_sum += l
            _, pred = torch.max(out, 1)
            corr += (pred == lbls).sum()
            tot += lbls.size(0)

            for i in range(lbls.size(0)):
                p, t = int(pred[i]), int(lbls[i])
                if p == t:
                    tp[t] += 1
                else:
                    if p < len(map): fp[p] += 1
                    fn[t] += 1

    if ddp and world_size > 1:
        dist.all_reduce(loss_sum);
        dist.all_reduce(corr);
        dist.all_reduce(tot)
        dist.all_reduce(tp);
        dist.all_reduce(fp);
        dist.all_reduce(fn)

    avg_loss = loss_sum.item() / max(1, len(loader) * world_size)
    acc = corr.item() / max(1, tot.item()) * 100

    pos_idx = next((v for k, v in map.items() if str(k).lower() == "true"), 1)
    tpc, fpc, fnc = tp[pos_idx].item(), fp[pos_idx].item(), fn[pos_idx].item()
    prec = tpc / (tpc + fpc) if (tpc + fpc) > 0 else 0
    rec = tpc / (tpc + fnc) if (tpc + fnc) > 0 else 0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0

    return avg_loss, acc, {}, {}, {'f1_true': f1, 'recall_true': rec}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("data_path", nargs="?", type=str)
    parser.add_argument("--data_paths", type=str, nargs='+')
    parser.add_argument("--train_data_paths_file", type=str)
    parser.add_argument("--val_data_paths_file", type=str)
    parser.add_argument("-o", "--output_path", default="./saved_models", type=str)
    parser.add_argument("--epochs", type=int, default=70)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--ddp", action="store_true")
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument("--resume", type=str)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--loss_type", default="weighted_ce")

    # Additional args
    parser.add_argument("--depths", type=int, nargs='+', default=[3, 3, 27, 3])
    parser.add_argument("--dims", type=int, nargs='+', default=[192, 384, 768, 1536])
    parser.add_argument("--warmup_epochs", type=int, default=3)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--save_val_results", action='store_true')
    parser.add_argument("--training_data_ratio", type=float, default=1.0)
    parser.add_argument("--data_parallel", action="store_true")
    parser.add_argument("--pos_weight", type=float, default=88.0)
    parser.add_argument("--prefetch_factor", type=int, default=4)
    parser.add_argument("--mp_context", type=str, default=None)

    args, _ = parser.parse_known_args()

    if args.data_path:
        data_in = args.data_path
    elif args.train_data_paths_file:
        data_in = (_read_paths_file(args.train_data_paths_file), _read_paths_file(args.val_data_paths_file))
    else:
        data_in = args.data_paths

    if args.ddp:
        torch.cuda.set_device(args.local_rank)
        dist.init_process_group(backend="nccl")
        IS_MAIN_PROCESS = (dist.get_rank() == 0)
        if IS_MAIN_PROCESS: print(f"DDP Init: Size={dist.get_world_size()} Local={args.local_rank}")
    else:
        IS_MAIN_PROCESS = True

    train_model(
        data_in, args.output_path, num_epochs=args.epochs, batch_size=args.batch_size,
        num_workers=args.num_workers, ddp=args.ddp, local_rank=args.local_rank,
        resume=args.resume, learning_rate=args.lr, loss_type=args.loss_type,
        warmup_epochs=args.warmup_epochs, weight_decay=args.weight_decay,
        depths=args.depths, dims=args.dims, training_data_ratio=args.training_data_ratio,
        data_parallel=args.data_parallel, pos_weight=args.pos_weight,
        prefetch_factor=args.prefetch_factor, mp_context=args.mp_context
    )

    if args.ddp and dist.is_initialized(): dist.destroy_process_group()