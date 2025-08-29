#!/usr/bin/env python3
"""
Validation over a large, SCATTERED directory tree of .npy tiles (thousands of subfolders).

Highlights
---------
• Recursively discovers files from one or many ROOTS (fast os.scandir).
• Flexible label sourcing:
   - --label_mode parent    → label = immediate parent folder name
   - --label_mode regex     → label extracted by --label_regex with a named group (?P<label>...)
   - --label_mode csv       → read a CSV with columns: path,label (relative or absolute)
• Stable class mapping: optional --class_map_json to lock name→index mapping; otherwise inferred.
• 6‑channel normalization values (from your earlier setup) applied on GPU for speed.
• Works on single GPU/CPU. Supports FP16 (inference) if CUDA available.
• Outputs summary metrics + optional per-class prediction lists JSON + optional predictions CSV.

Example (labels by parent folder):
  python validate_scattered_npy.py \
    --roots /data/big_tree \
    --label_mode parent \
    --ckpt /path/to/model_best.pth \
    --batch_size 256 --num_workers 16 --mmap --fp16

Example (labels by regex over full path):
  python validate_scattered_npy.py \
    --roots /data/a /data/b \
    --label_mode regex --label_regex ".*/(?P<label>true|false)/.*" \
    --ckpt /path/to/model_best.pth

Example (labels from CSV):
  python validate_scattered_npy.py \
    --roots /data/big_tree \
    --label_mode csv --labels_csv /data/labels.csv \
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
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# ---- Your model import (same as training) ----
THIS_DIR = os.path.abspath(os.path.dirname(__file__))
import sys
sys.path.append(os.path.abspath(os.path.join(THIS_DIR, "..")))
from mynet import ConvNeXtCBAMClassifier  # noqa: E402

# --- 6-channel normalization (Testing set numbers from your message) ---
VAL_MEAN = torch.tensor([
    18.417816162109375, 12.649129867553711, -0.5452527403831482,
    24.723854064941406, 4.690611362457275, 0.2813551473402196
], dtype=torch.float32)
VAL_STD = torch.tensor([
    25.028322219848633, 14.809632301330566, 0.6181337833404541,
    29.972835540771484, 7.9231791496276855, 0.7659083659074717
], dtype=torch.float32)

# --------------------- Fast recursive file discovery --------------------- #

def _scan_tree(start_dir: str, ext: str) -> List[str]:
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

# --------------------- Label extraction strategies ---------------------- #

def labels_from_parent(files: List[str]) -> Tuple[List[str], List[str]]:
    labels = [os.path.basename(os.path.dirname(p)) for p in files]
    return files, labels


def labels_from_regex(files: List[str], pattern: str) -> Tuple[List[str], List[str]]:
    rx = re.compile(pattern)
    labels: List[str] = []
    for p in files:
        m = rx.match(p)
        if not m or ("label" not in m.groupdict()):
            raise ValueError(f"Regex did not capture a 'label' group for path: {p}")
        labels.append(str(m.group("label")))
    return files, labels


def labels_from_csv(csv_path: str, roots: List[str]) -> Tuple[List[str], List[str]]:
    # Accept absolute or relative paths; if relative, resolve against the first root.
    base = os.path.abspath(os.path.expanduser(roots[0])) if roots else os.getcwd()
    files: List[str] = []
    labels: List[str] = []
    with open(csv_path, "r", newline="", encoding="utf-8") as f:
        rdr = csv.DictReader(f)
        if not {"path", "label"}.issubset(set(rdr.fieldnames or [])):
            raise ValueError("CSV must have columns: path,label")
        for row in rdr:
            rp = row["path"].strip()
            ap = os.path.abspath(os.path.join(base, rp)) if not os.path.isabs(rp) else rp
            files.append(ap)
            labels.append(str(row["label"]).strip())
    return files, labels

# ---------------------------- Dataset ---------------------------------- #

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


class ScatteredNpyDataset(Dataset):
    def __init__(self,
                 files: List[str],
                 labels: List[str],
                 class_to_idx: Dict[str, int],
                 channels: int = 6,
                 tile_single_channel: bool = False,
                 mmap: bool = True):
        assert len(files) == len(labels), "files and labels must align"
        self.files = files
        self.labels = labels
        self.class_to_idx = class_to_idx
        self.channels = channels
        self.tile_single_channel = tile_single_channel
        self.mmap = mmap
        if self.channels != 6:
            raise ValueError(f"This script expects 6 channels; got {self.channels}")

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int):
        path = self.files[idx]
        y_name = self.labels[idx]
        y = self.class_to_idx[y_name]
        arr = np.load(path, allow_pickle=False, mmap_mode=('r' if self.mmap else None))
        chw = ensure_chw(arr, self.channels, tile_single_channel=self.tile_single_channel)
        t = torch.from_numpy(chw).float()  # CHW, float32 (no CPU normalization)
        return t, torch.tensor(y, dtype=torch.long), path

# --------------------------- Model loading ------------------------------ #

def load_model(ckpt_path: str, device: torch.device, in_channels: int, depths: List[int], dims: List[int], num_classes: int) -> torch.nn.Module:
    checkpoint = torch.load(ckpt_path, map_location=device)
    # Try to honor saved in_channels / genotype_map if present
    ckpt_in_ch = int(checkpoint.get('in_channels', in_channels))
    if ckpt_in_ch != in_channels:
        raise ValueError(f"Checkpoint was trained with in_channels={ckpt_in_ch}, but script is set to {in_channels}")

    ckpt_map = checkpoint.get('genotype_map', None)
    if ckpt_map is not None:
        num_classes = max(int(v) for v in ckpt_map.values()) + 1

    model = ConvNeXtCBAMClassifier(in_channels=in_channels, class_num=num_classes,
                                   depths=depths, dims=dims).to(device)

    state = checkpoint.get('model_state_dict', checkpoint)
    model.load_state_dict(state, strict=True)
    model.eval()
    return model

# ----------------------------- Metrics --------------------------------- #

def compute_metrics(tp: torch.Tensor, fp: torch.Tensor, fn: torch.Tensor, class_total: torch.Tensor,
                    class_names: List[str], pos_name: Optional[str] = "true") -> Tuple[Dict[str, Any], Dict[str, Any]]:
    # per-class accuracy
    class_stats = {}
    for i, cname in enumerate(class_names):
        total = int(class_total[i].item())
        tp_i = int(tp[i].item())
        acc = (tp_i / total * 100.0) if total > 0 else 0.0
        class_stats[cname] = {"idx": i, "acc": acc, "correct": tp_i, "total": total}

    # choose positive index
    if pos_name is not None and pos_name in class_names:
        pos_idx = class_names.index(pos_name)
    else:
        pos_idx = 1 if len(class_names) > 1 else 0

    tpc = float(tp[pos_idx].item())
    fpc = float(fp[pos_idx].item())
    fnc = float(fn[pos_idx].item())

    precision_true = (tpc / (tpc + fpc)) if (tpc + fpc) > 0 else 0.0
    recall_true = (tpc / (tpc + fnc)) if (tpc + fnc) > 0 else 0.0
    f1_true = (2 * precision_true * recall_true / (precision_true + recall_true)) if (precision_true + recall_true) > 0 else 0.0

    metrics = {
        'precision_true': precision_true,
        'recall_true': recall_true,
        'f1_true': f1_true,
        'pos_class_idx': pos_idx,
    }
    return class_stats, metrics

# --------------------------- Validation core --------------------------- #

@torch.inference_mode()
def validate(model: torch.nn.Module,
             dl: DataLoader,
             device: torch.device,
             class_names: List[str],
             save_lists: bool = False) -> Tuple[float, float, Dict[str, Any], Dict[str, List[str]], Dict[str, Any]]:
    model.eval()

    mean_dev = VAL_MEAN.to(device).view(-1, 1, 1)
    std_dev = VAL_STD.to(device).view(-1, 1, 1)

    num_classes = len(class_names)
    total_loss = 0.0
    batches = 0
    correct = 0
    total = 0

    # On-device counters
    tp = torch.zeros(num_classes, device=device, dtype=torch.long)
    fp = torch.zeros(num_classes, device=device, dtype=torch.long)
    fn = torch.zeros(num_classes, device=device, dtype=torch.long)
    class_total = torch.zeros(num_classes, device=device, dtype=torch.long)

    ce = torch.nn.CrossEntropyLoss(reduction='mean')  # use CE for reporting

    infer_lists: Dict[str, List[str]] = {c: [] for c in class_names}

    with tqdm(total=len(dl), desc="Validate", unit="batch", dynamic_ncols=True, leave=True) as bar:
        for images, labels, paths in dl:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            images = (images - mean_dev) / std_dev
            logits = model(images)
            if isinstance(logits, tuple):
                logits = logits[0]

            loss = ce(logits, labels)
            total_loss += float(loss.item())
            batches += 1

            preds = torch.argmax(logits, dim=1)
            correct += int((preds == labels).sum().item())
            total += int(labels.numel())

            # per-class accounting
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
                if save_lists:
                    infer_lists[class_names[p]].append(os.path.basename(paths[i]))

            bar.update(1)

    avg_loss = total_loss / max(1, batches)
    acc = (correct / max(1, total)) * 100.0
    class_stats, metrics = compute_metrics(tp, fp, fn, class_total, class_names, pos_name="true")
    return avg_loss, acc, class_stats, (infer_lists if save_lists else {}), metrics

# ------------------------------ Main ----------------------------------- #

def main():
    ap = argparse.ArgumentParser(description="Validate a ConvNeXtCBAMClassifier on scattered .npy trees")
    ap.add_argument('--roots', type=str, nargs='+', required=True,
                    help='One or more root directories to recursively scan for files')
    ap.add_argument('--file_ext', type=str, default='.npy', help='File extension to include (default: .npy)')

    ap.add_argument('--label_mode', choices=['parent', 'regex', 'csv'], required=True,
                    help='How to get labels: immediate parent folder, regex on path, or CSV mapping')
    ap.add_argument('--label_regex', type=str, default=None,
                    help="Regex applied to FULL PATH; must define a named group (?P<label>...) when --label_mode=regex")
    ap.add_argument('--labels_csv', type=str, default=None,
                    help='CSV with columns path,label when --label_mode=csv')

    ap.add_argument('--class_map_json', type=str, default=None,
                    help='Optional JSON with {class_name: index}. Locks mapping and class order.')

    ap.add_argument('--ckpt', type=str, required=True, help='Checkpoint .pth path')
    ap.add_argument('--depths', type=int, nargs='+', default=[3, 3, 27, 3])
    ap.add_argument('--dims', type=int, nargs='+', default=[192, 384, 768, 1536])

    ap.add_argument('--batch_size', type=int, default=128)
    ap.add_argument('--num_workers', type=int, default=16)
    ap.add_argument('--mmap', action='store_true', help='Use numpy memmap mode')
    ap.add_argument('--tile_single_channel', action='store_true', help='Tile 1-channel arrays to 6')
    ap.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto')
    ap.add_argument('--fp16', action='store_true')

    ap.add_argument('--save_pred_csv', type=str, default='', help='Write per-sample predictions CSV here (optional)')
    ap.add_argument('--save_val_json', type=str, default='', help='Write per-class predicted filenames JSON here (optional)')

    args = ap.parse_args()

    # Device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    elif args.device == 'cuda':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if device.type == 'cpu':
            print('CUDA requested but not available; using CPU')
    else:
        device = torch.device('cpu')

    if device.type == 'cuda':
        torch.backends.cudnn.benchmark = True
        try:
            torch.set_float32_matmul_precision('high')
        except Exception:
            pass

    # Discover files
    files = list_files_parallel(args.roots, ext=args.file_ext.lower(), workers=args.num_workers)
    if len(files) == 0:
        raise SystemExit('No files found.')

    # Labels
    if args.label_mode == 'parent':
        files, labels = labels_from_parent(files)
    elif args.label_mode == 'regex':
        if not args.label_regex:
            raise SystemExit('--label_regex is required when --label_mode=regex')
        files, labels = labels_from_regex(files, args.label_regex)
    else:  # csv
        if not args.labels_csv:
            raise SystemExit('--labels_csv is required when --label_mode=csv')
        files, labels = labels_from_csv(args.labels_csv, args.roots)

    # Class mapping
    if args.class_map_json:
        with open(args.class_map_json, 'r', encoding='utf-8') as f:
            class_to_idx = {str(k): int(v) for k, v in json.load(f).items()}
        # sanity: unseen labels?
        unseen = sorted({l for l in labels if l not in class_to_idx})
        if unseen:
            raise SystemExit(f"Labels not in class_map_json: {unseen[:10]} ... (total {len(unseen)})")
    else:
        # infer mapping from sorted unique labels
        uniq = sorted(set(labels))
        class_to_idx = {c: i for i, c in enumerate(uniq)}

    class_names = [None] * len(class_to_idx)
    for k, v in class_to_idx.items():
        class_names[v] = k

    # Dataset / DataLoader
    ds = ScatteredNpyDataset(files=files, labels=labels, class_to_idx=class_to_idx,
                             channels=6, tile_single_channel=args.tile_single_channel, mmap=args.mmap)
    dl = DataLoader(ds,
                    batch_size=args.batch_size,
                    num_workers=args.num_workers,
                    pin_memory=(device.type == 'cuda'),
                    persistent_workers=(args.num_workers > 0),
                    prefetch_factor=(4 if args.num_workers > 0 else None),
                    shuffle=False,
                    drop_last=False)

    # Model
    model = load_model(args.ckpt, device=device, in_channels=6,
                       depths=args.depths, dims=args.dims, num_classes=len(class_names))
    if args.fp16 and device.type == 'cuda':
        model = model.half()

    # Run validation
    avg_loss, acc, class_stats, infer_lists, metrics = validate(model, dl, device, class_names, save_lists=bool(args.save_val_json))

    print("\n=== Validation Summary ===")
    print(f"Files: {len(files)} | Batches: {max(1, len(dl))}")
    print(f"Avg loss: {avg_loss:.4f} | Acc: {acc:.2f}%")
    print(f"Precision(true): {metrics['precision_true']*100:.2f}% | Recall(true): {metrics['recall_true']*100:.2f}% | F1(true): {metrics['f1_true']:.4f} | pos_idx={metrics['pos_class_idx']}")
    print("Class-wise accuracy:")
    for cname in class_names:
        s = class_stats[cname]
        print(f"  {cname:>12s} | idx={s['idx']:2d} | acc={s['acc']:6.2f}% | {s['correct']}/{s['total']}")

    # Optional outputs
    if args.save_val_json:
        out_json = os.path.abspath(args.save_val_json)
        with open(out_json, 'w', encoding='utf-8') as f:
            json.dump({
                'metrics': metrics,
                'class_stats': class_stats,
                'predicted_lists': infer_lists,
            }, f, indent=2)
        print(f"Wrote validation JSON: {out_json}")

    if args.save_pred_csv:
        out_csv = os.path.abspath(args.save_pred_csv)
        with open(out_csv, 'w', newline='', encoding='utf-8') as f:
            w = csv.writer(f)
            w.writerow(["path", "true_label", "pred_label", "pred_idx"])
            # Re-run lightly to dump preds without storing all in RAM
            mean_dev = VAL_MEAN.to(device).view(-1,1,1)
            std_dev = VAL_STD.to(device).view(-1,1,1)
            model.eval()
            with torch.inference_mode():
                for images, labels, paths in tqdm(dl, desc="Dump preds", leave=False):
                    images = (images.to(device, non_blocking=True) - mean_dev) / std_dev
                    logits = model(images)
                    if isinstance(logits, tuple):
                        logits = logits[0]
                    preds = torch.argmax(logits, dim=1).cpu().tolist()
                    labels = labels.cpu().tolist()
                    for pth, y, p in zip(paths, labels, preds):
                        w.writerow([pth, class_names[y], class_names[p], p])
        print(f"Wrote predictions CSV: {out_csv}")


if __name__ == '__main__':
    main()
