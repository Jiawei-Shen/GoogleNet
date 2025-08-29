#!/usr/bin/env python3
"""
Unified, FAST validation script that can:
  • Re-use your existing get_data_loader(...) validation pipeline (organized train/val folders), OR
  • Validate over a scattered tree with thousands of subfolders (no fixed split), with parent/regex/CSV labels.

Speed tricks:
  • GPU-side normalization + AMP (optional) to cut CPU work.
  • Tuned DataLoader (pin_memory, persistent_workers, prefetch_factor).
  • Optional --preload for scattered mode: stack all arrays into one pinned tensor (RAM required).

Examples
--------
# Mode A: Use your existing folder-structured val loader, but with faster loader params
python validate_unified_fast.py \
  --mode val_loader \
  --data_path /data/my_dataset_root \
  --ckpt /path/to/model_best.pth \
  --batch_size 256 --num_workers 16 --fp16 --gpu_norm

# Mode B: Scattered files, labels by parent dir ("true"/"false" live as parent)
python validate_unified_fast.py \
  --mode scattered \
  --roots /big/tree1 /big/tree2 \
  --label_mode parent \
  --ckpt /path/to/model_best.pth \
  --batch_size 256 --num_workers 16 --mmap --fp16 --gpu_norm --preload

# Mode B: Scattered files, labels via regex over full path
python validate_unified_fast.py \
  --mode scattered \
  --roots /big/tree \
  --label_mode regex --label_regex ".*/(?P<label>true|false)/.*" \
  --ckpt /path/to/model_best.pth

# Mode B: Labels from CSV with columns path,label (paths may be relative to first root)
python validate_unified_fast.py \
  --mode scattered \
  --roots /big/tree \
  --label_mode csv --labels_csv labels.csv \
  --ckpt /path/to/model_best.pth
"""

import argparse
import os
import re
import csv
import json
import time
from typing import List, Tuple, Dict, Any, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset
from tqdm import tqdm

# --- Model & your val loader ---
THIS_DIR = os.path.abspath(os.path.dirname(__file__))
import sys
sys.path.append(os.path.abspath(os.path.join(THIS_DIR, "..")))
from mynet import ConvNeXtCBAMClassifier  # noqa: E402
from dataset_pansoma_npy_6ch import get_data_loader  # your existing val loader

# --- 6-channel normalization (Testing values from your messages) ---
VAL_MEAN = torch.tensor([
    18.417816162109375, 12.649129867553711, -0.5452527403831482,
    24.723854064941406, 4.690611362457275, 0.2813551473402196
], dtype=torch.float32)
VAL_STD = torch.tensor([
    25.028322219848633, 14.809632301330566, 0.6181337833404541,
    29.972835540771484, 7.9231791496276855, 0.7659083659074717
], dtype=torch.float32)

# ------------------ Scattered-mode helpers (fast recursive scan) ------------------ #

def _scan_tree(start_dir: str, ext: str = ".npy") -> List[str]:
    stack = [start_dir]
    out: List[str] = []
    while stack:
        d = stack.pop()
        try:
            with os.scandir(d) as it:
                for e in it:
                    try:
                        if e.is_dir(follow_symlinks=False):
                            stack.append(e.path)
                        else:
                            if e.name.lower().endswith(ext):
                                out.append(e.path)
                    except OSError:
                        continue
        except (PermissionError, FileNotFoundError):
            continue
    return out


def list_files_parallel(roots: List[str], ext: str, workers: int) -> List[str]:
    roots = [os.path.abspath(os.path.expanduser(r)) for r in roots]
    files: List[str] = []
    if workers <= 1 or len(roots) <= 1:
        for r in roots:
            files.extend(_scan_tree(r, ext))
        files.sort()
        return files

    start = time.time()
    with ThreadPoolExecutor(max_workers=max(1, workers)) as ex, \
         tqdm(total=len(roots), desc="Scan", unit="root", leave=False, dynamic_ncols=True) as bar:
        futs = [ex.submit(_scan_tree, r, ext) for r in roots]
        for fut in as_completed(futs):
            files.extend(fut.result())
            done = bar.n + 1
            elapsed = time.time() - start
            speed = done / max(1e-9, elapsed)
            remain = (len(roots) - done) / max(1e-9, speed)
            bar.set_postfix_str(f"speed={speed:.1f} root/s ETA={int(remain)}s")
            bar.update(1)
    files.sort()
    return files

# ----------------------------- Labels for scattered ----------------------------- #

def labels_from_parent(files: List[str]) -> Tuple[List[str], List[str]]:
    return files, [os.path.basename(os.path.dirname(p)) for p in files]


def labels_from_regex(files: List[str], pattern: str) -> Tuple[List[str], List[str]]:
    rx = re.compile(pattern)
    labels: List[str] = []
    for p in files:
        m = rx.match(p)
        if not m or ("label" not in m.groupdict()):
            raise ValueError(f"Regex must capture a named group (?P<label>...): {p}")
        labels.append(str(m.group("label")))
    return files, labels


def labels_from_csv(csv_path: str, roots: List[str]) -> Tuple[List[str], List[str]]:
    base = os.path.abspath(os.path.expanduser(roots[0])) if roots else os.getcwd()
    files: List[str] = []
    labels: List[str] = []
    with open(csv_path, "r", newline="", encoding="utf-8") as f:
        rdr = csv.DictReader(f)
        if not {"path", "label"}.issubset(set(rdr.fieldnames or [])):
            raise ValueError("CSV must contain columns: path,label")
        for row in rdr:
            rp = row["path"].strip()
            ap = os.path.abspath(os.path.join(base, rp)) if not os.path.isabs(rp) else rp
            files.append(ap)
            labels.append(str(row["label"]).strip())
    return files, labels

# -------------------------- Core tensor prep helpers --------------------------- #

def ensure_chw(x: np.ndarray, channels: int, tile_single_channel: bool) -> np.ndarray:
    if x.ndim == 3:
        if x.shape[0] == channels:
            chw = x
        elif x.shape[-1] == channels:
            chw = np.transpose(x, (2, 0, 1))
        elif x.shape[0] == 1 and tile_single_channel and channels > 1:
            chw = np.tile(x, (channels, 1, 1))
        elif x.shape[-1] == 1 and tile_single_channel and channels > 1:
            chw = np.transpose(x, (2, 0, 1))
            chw = np.tile(chw, (channels, 1, 1))
        else:
            raise ValueError(f"Cannot infer channel axis for shape {x.shape} with channels={channels}")
    elif x.ndim == 2:
        if tile_single_channel and channels > 1:
            chw = np.tile(x[None, ...], (channels, 1, 1))
        else:
            if channels != 1:
                raise ValueError(f"Input is 2D but channels={channels}. Use --tile_single_channel to tile.")
            chw = x[None, ...]
    else:
        raise ValueError(f"Unsupported tensor ndim={x.ndim}, shape={x.shape}")

    if chw.dtype != np.float32:
        chw = chw.astype(np.float32, copy=False)
    if not chw.flags['C_CONTIGUOUS']:
        chw = np.ascontiguousarray(chw)
    return chw

# ------------------------------- Datasets --------------------------------- #

class ScatteredNpyDataset(Dataset):
    def __init__(self, files: List[str], labels: List[str], class_to_idx: Dict[str, int],
                 channels: int = 6, tile_single_channel: bool = False, mmap: bool = True):
        assert len(files) == len(labels)
        self.files = files
        self.labels = labels
        self.class_to_idx = class_to_idx
        self.channels = channels
        self.tile_single_channel = tile_single_channel
        self.mmap = mmap

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx: int):
        p = self.files[idx]
        y = self.class_to_idx[self.labels[idx]]
        arr = np.load(p, allow_pickle=False, mmap_mode=('r' if self.mmap else None))
        chw = ensure_chw(arr, 6, self.tile_single_channel)
        t = torch.from_numpy(chw).float()
        return t, torch.tensor(y, dtype=torch.long), p


def preload_all(files: List[str], labels: List[str], class_to_idx: Dict[str, int],
                tile_single_channel: bool, mmap: bool) -> Tuple[TensorDataset, List[str]]:
    # Peek shape
    first = np.load(files[0], allow_pickle=False, mmap_mode=('r' if mmap else None))
    first_chw = ensure_chw(first, 6, tile_single_channel)
    C, H, W = first_chw.shape
    X = torch.empty((len(files), C, H, W), dtype=torch.float32, pin_memory=True)
    y = torch.empty((len(files),), dtype=torch.long)
    X[0].copy_(torch.from_numpy(first_chw))
    y[0] = class_to_idx[labels[0]]
    for i in range(1, len(files)):
        arr = np.load(files[i], allow_pickle=False, mmap_mode=('r' if mmap else None))
        chw = ensure_chw(arr, 6, tile_single_channel)
        X[i].copy_(torch.from_numpy(chw))
        y[i] = class_to_idx[labels[i]]
    ds = TensorDataset(X, y, torch.arange(len(files)))
    return ds, files

# ------------------------------- Model ----------------------------------- #

def load_model(ckpt: str, device: torch.device, num_classes: int, depths: List[int], dims: List[int]) -> torch.nn.Module:
    ckpt_obj = torch.load(ckpt, map_location=device)
    in_ch = int(ckpt_obj.get('in_channels', 6))
    if in_ch != 6:
        raise ValueError(f"Checkpoint in_channels={in_ch}, but this script assumes 6.")
    # allow genotype_map to define num_classes if present
    gm = ckpt_obj.get('genotype_map')
    if gm:
        num_classes = max(int(v) for v in gm.values()) + 1
    model = ConvNeXtCBAMClassifier(in_channels=6, class_num=num_classes, depths=depths, dims=dims).to(device)
    state = ckpt_obj.get('model_state_dict', ckpt_obj)
    model.load_state_dict(state, strict=True)
    model.eval()
    return model

# ------------------------------- Validate -------------------------------- #

@torch.inference_mode()
def validate_loop(model: torch.nn.Module, dl: DataLoader, device: torch.device, class_names: List[str],
                  gpu_norm: bool, fp16: bool, assume_already_normalized: bool = False,
                  save_lists: bool = False) -> Tuple[float, float, Dict[str, Any], Dict[str, List[str]], Dict[str, Any]]:
    mean_dev = VAL_MEAN.to(device).view(-1,1,1)
    std_dev  = VAL_STD.to(device).view(-1,1,1)

    num_classes = len(class_names)
    ce = torch.nn.CrossEntropyLoss(reduction='mean')

    total_loss = 0.0
    batches = 0
    correct = 0
    total = 0

    tp = torch.zeros(num_classes, device=device, dtype=torch.long)
    fp = torch.zeros(num_classes, device=device, dtype=torch.long)
    fn = torch.zeros(num_classes, device=device, dtype=torch.long)
    class_total = torch.zeros(num_classes, device=device, dtype=torch.long)

    lists: Dict[str, List[str]] = {c: [] for c in class_names}

    use_amp = (fp16 and device.type == 'cuda')

    with tqdm(total=len(dl), desc='Validate', unit='batch', dynamic_ncols=True, leave=True) as bar:
        for batch in dl:
            if len(batch) == 3:
                images, labels, paths_or_idx = batch
                if isinstance(paths_or_idx, list) or (isinstance(paths_or_idx, torch.Tensor) and paths_or_idx.ndim==1 and paths_or_idx.dtype==torch.long):
                    pass
            else:
                images, labels = batch
                paths_or_idx = None

            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            if gpu_norm and not assume_already_normalized:
                if use_amp:
                    with torch.cuda.amp.autocast():
                        images = (images - mean_dev) / std_dev
                        logits = model(images)
                else:
                    images = (images - mean_dev) / std_dev
                    logits = model(images)
            else:
                # either CPU transforms already normalized them, or user asked to skip
                if use_amp:
                    with torch.cuda.amp.autocast():
                        logits = model(images)
                else:
                    logits = model(images)

            if isinstance(logits, tuple):
                logits = logits[0]

            loss = ce(logits, labels)
            total_loss += float(loss.item())
            batches += 1

            preds = torch.argmax(logits, dim=1)
            correct += int((preds == labels).sum().item())
            total += int(labels.numel())

            for i in range(labels.size(0)):
                t = int(labels[i])
                p = int(preds[i])
                class_total[t] += 1
                if p == t:
                    tp[t] += 1
                else:
                    if p < num_classes:
                        fp[p] += 1
                    fn[t] += 1
                if save_lists and paths_or_idx is not None:
                    if isinstance(paths_or_idx, list):
                        fname = os.path.basename(paths_or_idx[i])
                    else:
                        # When using TensorDataset preload, third field is an index; user can remap if desired
                        fname = str(int(paths_or_idx[i]))
                    lists[class_names[p]].append(fname)

            bar.update(1)

    avg_loss = total_loss / max(1, batches)
    acc = (correct / max(1, total)) * 100.0

    # Choose positive class index (prefer a class literally named 'true')
    pos_idx = class_names.index('true') if 'true' in class_names else (1 if len(class_names)>1 else 0)
    tpc = float(tp[pos_idx].item())
    fpc = float(fp[pos_idx].item())
    fnc = float(fn[pos_idx].item())
    precision_true = (tpc / (tpc + fpc)) if (tpc + fpc) > 0 else 0.0
    recall_true    = (tpc / (tpc + fnc)) if (tpc + fnc) > 0 else 0.0
    f1_true        = (2 * precision_true * recall_true / (precision_true + recall_true)) if (precision_true + recall_true) > 0 else 0.0

    class_stats = {}
    for i, cname in enumerate(class_names):
        tot = int(class_total[i].item())
        cor = int(tp[i].item())
        class_stats[cname] = {"idx": i, "acc": (cor/tot*100.0) if tot>0 else 0.0, "correct": cor, "total": tot}

    metrics = {"precision_true": precision_true, "recall_true": recall_true, "f1_true": f1_true, "pos_class_idx": pos_idx}
    return avg_loss, acc, class_stats, (lists if save_lists else {}), metrics

# --------------------------------- Main ---------------------------------- #

def main():
    ap = argparse.ArgumentParser(description='Unified fast validator (val_loader or scattered)')
    ap.add_argument('--mode', choices=['val_loader','scattered'], required=True,
                    help='val_loader: reuse your dataset_pansoma get_data_loader(); scattered: crawl roots')

    # Common
    ap.add_argument('--ckpt', required=True, type=str)
    ap.add_argument('--depths', type=int, nargs='+', default=[3,3,27,3])
    ap.add_argument('--dims', type=int, nargs='+', default=[192,384,768,1536])
    ap.add_argument('--batch_size', type=int, default=128)
    ap.add_argument('--num_workers', type=int, default=16)
    ap.add_argument('--device', choices=['auto','cpu','cuda'], default='auto')
    ap.add_argument('--fp16', action='store_true')
    ap.add_argument('--gpu_norm', action='store_true', help='Apply normalization on GPU before forward()')
    ap.add_argument('--save_val_json', type=str, default='', help='Optional JSON with metrics + per-class predicted filenames')

    # Mode A (val_loader)
    ap.add_argument('--data_path', type=str, default=None, help="Dataset root containing 'train/' and 'val/'")
    ap.add_argument('--assume_dataset_already_normalized', action='store_true',
                    help='Set if your NpyDataset already applies the 6ch Normalize in __getitem__')

    # Mode B (scattered)
    ap.add_argument('--roots', type=str, nargs='+', help='One or more roots to crawl (scattered mode)')
    ap.add_argument('--file_ext', type=str, default='.npy')
    ap.add_argument('--label_mode', choices=['parent','regex','csv'], default='parent')
    ap.add_argument('--label_regex', type=str, default=None)
    ap.add_argument('--labels_csv', type=str, default=None)
    ap.add_argument('--class_map_json', type=str, default=None)
    ap.add_argument('--mmap', action='store_true')
    ap.add_argument('--tile_single_channel', action='store_true')
    ap.add_argument('--preload', action='store_true', help='Scattered mode: preload into one pinned tensor')

    args = ap.parse_args()

    # Device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    elif args.device == 'cuda':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if device.type == 'cpu':
            print('CUDA requested but not available, falling back to CPU')
    else:
        device = torch.device('cpu')

    if device.type == 'cuda':
        torch.backends.cudnn.benchmark = True
        try:
            torch.set_float32_matmul_precision('high')
        except Exception:
            pass

    # Build dataset/loader
    if args.mode == 'val_loader':
        if not args.data_path:
            raise SystemExit('--data_path is required when --mode val_loader')
        # Use your helper to get the val dataset; we will rebuild the DataLoader with faster params.
        base_loader, genotype_map = get_data_loader(data_dir=args.data_path, dataset_type='val',
                                                    batch_size=args.batch_size, num_workers=args.num_workers,
                                                    shuffle=False, return_paths=True)
        if not genotype_map:
            raise SystemExit('genotype_map is empty from get_data_loader')
        class_names = [None] * len(genotype_map)
        for k,v in genotype_map.items():
            class_names[v] = k
        ds = base_loader.dataset  # reuse the same dataset object
        # Rebuild loader to ensure fast settings (pin, persistent, prefetch)
        dl = DataLoader(ds,
                        batch_size=args.batch_size,
                        num_workers=args.num_workers,
                        pin_memory=(device.type=='cuda'),
                        persistent_workers=(args.num_workers>0),
                        prefetch_factor=(4 if args.num_workers>0 else None),
                        shuffle=False,
                        drop_last=False)
        assume_norm = bool(args.assume_dataset_already_normalized)
    else:
        # Scattered mode
        if not args.roots:
            raise SystemExit('--roots is required when --mode scattered')
        files = list_files_parallel(args.roots, ext=args.file_ext.lower(), workers=args.num_workers)
        if len(files) == 0:
            raise SystemExit('No files found under given roots')
        if args.label_mode == 'parent':
            files, labels = labels_from_parent(files)
        elif args.label_mode == 'regex':
            if not args.label_regex:
                raise SystemExit('--label_regex required for label_mode=regex')
            files, labels = labels_from_regex(files, args.label_regex)
        else:
            if not args.labels_csv:
                raise SystemExit('--labels_csv required for label_mode=csv')
            files, labels = labels_from_csv(args.labels_csv, args.roots)
        if args.class_map_json:
            with open(args.class_map_json, 'r', encoding='utf-8') as f:
                class_to_idx = {str(k): int(v) for k,v in json.load(f).items()}
            unseen = sorted({l for l in labels if l not in class_to_idx})
            if unseen:
                raise SystemExit(f'Labels not in class_map_json (first 10): {unseen[:10]}')
        else:
            uniq = sorted(set(labels))
            class_to_idx = {c:i for i,c in enumerate(uniq)}
        class_names = [None]*len(class_to_idx)
        for k,v in class_to_idx.items():
            class_names[v] = k
        if args.preload:
            ds, paths = preload_all(files, labels, class_to_idx, args.tile_single_channel, args.mmap)
            # Collate to map index back to filename
            dl = DataLoader(ds, batch_size=args.batch_size, num_workers=0, pin_memory=True, shuffle=False,
                            drop_last=False,
                            collate_fn=lambda batch: (
                                torch.stack([b[0] for b in batch], dim=0),
                                torch.stack([b[1] for b in batch], dim=0),
                                [paths[int(b[2])] for b in batch]
                            ))
        else:
            ds = ScatteredNpyDataset(files, labels, class_to_idx,
                                     channels=6, tile_single_channel=args.tile_single_channel, mmap=args.mmap)
            dl = DataLoader(ds, batch_size=args.batch_size, num_workers=args.num_workers,
                            pin_memory=(device.type=='cuda'), persistent_workers=(args.num_workers>0),
                            prefetch_factor=(4 if args.num_workers>0 else None), shuffle=False, drop_last=False)
        assume_norm = False  # we normalize on GPU in scattered mode

    # Model
    model = load_model(args.ckpt, device=device, num_classes=len(class_names), depths=args.depths, dims=args.dims)
    if args.fp16 and device.type == 'cuda':
        model = model.half()

    # Validate
    avg_loss, acc, class_stats, lists, metrics = validate_loop(
        model, dl, device, class_names,
        gpu_norm=bool(args.gpu_norm), fp16=bool(args.fp16),
        assume_already_normalized=bool(assume_norm),
        save_lists=bool(args.save_val_json))

    print("\n=== Validation Summary ===")
    print(f"Batches: {len(dl)} | Avg loss: {avg_loss:.4f} | Acc: {acc:.2f}%")
    print(f"Prec(true): {metrics['precision_true']*100:.2f}% | Rec(true): {metrics['recall_true']*100:.2f}% | F1(true): {metrics['f1_true']:.4f} | pos_idx={metrics['pos_class_idx']}")
    print("Class-wise accuracy:")
    for cname, s in class_stats.items():
        print(f"  {cname:>12s} | idx={s['idx']:2d} | acc={s['acc']:6.2f}% | {s['correct']}/{s['total']}")

    if args.save_val_json:
        out_json = os.path.abspath(args.save_val_json)
        with open(out_json, 'w', encoding='utf-8') as f:
            json.dump({'metrics': metrics, 'class_stats': class_stats, 'predicted_lists': lists}, f, indent=2)
        print(f"Wrote validation JSON: {out_json}")


if __name__ == '__main__':
    main()
