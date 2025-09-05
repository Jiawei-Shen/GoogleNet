#!/usr/bin/env python3
"""
Fast batch inference over a directory of .npy files (recursively).

This simplified version:
  • Normalization is done in the Dataset (transforms.Normalize, CHW)
  • Inference loop is a simple for-loop over the DataLoader
  • Removed fp16 and channels_last options
  • Keeps mmap loading, torch.compile, and clean outputs
"""

import argparse
import os
import sys
import json
import time
from typing import List, Tuple, Dict, Any
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# Import your model the same way as in training
THIS_DIR = os.path.abspath(os.path.dirname(__file__))
sys.path.append(os.path.abspath(os.path.join(THIS_DIR, "..")))
from mynet import ConvNeXtCBAMClassifier  # noqa: E402

# --- EXACT SAME NORMALIZATION AS VAL (COLO829T Testing) ---
VAL_MEAN = torch.tensor([
    18.417816162109375, 12.649129867553711, -0.5452527403831482,
    24.723854064941406, 4.690611362457275, 0.2813551473402196
], dtype=torch.float32)
VAL_STD = torch.tensor([
    25.028322219848633, 14.809632301330566, 0.6181337833404541,
    29.972835540771484, 7.9231791496276855, 0.7659083659074717
], dtype=torch.float32)

# ----------------------------- Utilities ------------------------------------- #

def _scan_tree(start_dir: str) -> List[str]:
    """Iteratively scan one subtree with os.scandir (fast)."""
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
                            if e.name.lower().endswith(".npy"):
                                out.append(e.path)
                    except OSError:
                        continue
        except (PermissionError, FileNotFoundError):
            continue
    return out


def _format_eta(seconds: float) -> str:
    seconds = int(max(0, seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h > 0:
        return f"{h:d}h{m:02d}m{s:02d}s"
    elif m > 0:
        return f"{m:d}m{s:02d}s"
    else:
        return f"{s:d}s"


def list_npy_files_parallel(root: str, workers: int) -> List[str]:
    """Parallel file discovery with tqdm bar."""
    root = os.path.abspath(os.path.expanduser(root))
    top_dirs, files = [], []

    try:
        with os.scandir(root) as it:
            for e in it:
                try:
                    if e.is_dir(follow_symlinks=False):
                        top_dirs.append(e.path)
                    elif e.name.lower().endswith(".npy"):
                        files.append(e.path)
                except OSError:
                    continue
    except FileNotFoundError:
        return []

    if workers <= 1 or not top_dirs:
        for d in top_dirs:
            files.extend(_scan_tree(d))
        return sorted(files)

    start = time.time()
    with ThreadPoolExecutor(max_workers=workers) as ex, \
         tqdm(total=len(top_dirs), desc="Scan", unit="dir", leave=False, dynamic_ncols=True) as bar:
        futures = [ex.submit(_scan_tree, d) for d in top_dirs]
        for fut in as_completed(futures):
            files.extend(fut.result())
            bar.update(1)
            done = bar.n
            elapsed = time.time() - start
            speed = done / max(1e-9, elapsed)
            eta = (len(top_dirs) - done) / max(1e-9, speed)
            bar.set_postfix_str(f"speed={speed:.1f} dir/s ETA={_format_eta(eta)}")

    return sorted(files)


def ensure_chw(x: np.ndarray, channels: int) -> np.ndarray:
    """Ensure numpy array is (C,H,W) float32."""
    if x.ndim == 3:
        if x.shape[0] == channels:
            chw = x
        elif x.shape[-1] == channels:
            chw = np.transpose(x, (2, 0, 1))
        else:
            raise ValueError(f"Unexpected shape {x.shape}, expected {channels} channels")
    elif x.ndim == 2 and channels == 1:
        chw = x[None, ...]
    else:
        raise ValueError(f"Unsupported shape {x.shape} for channels={channels}")
    return chw.astype(np.float32, copy=False)


def print_data_info(files: List[str], in_channels: int, mmap: bool, k: int) -> None:
    print("\n=== Data Information ===")
    print(f"Total .npy files: {len(files)} | in_channels={in_channels} | mmap={mmap}")
    for i, p in enumerate(files[:k]):
        try:
            arr = np.load(p, mmap_mode="r" if mmap else None)
            print(f"  [{i+1}] {p} | shape={arr.shape}, dtype={arr.dtype}")
        except Exception as e:
            print(f"  [{i+1}] {p} | <error: {e}>")
    print("=" * 26)


# ---------------------------- Dataset ---------------------------------------- #

class NpyDirDataset(Dataset):
    def __init__(self, files: List[str], channels: int, mmap: bool = False, transform=None):
        self.files = files
        self.channels = channels
        self.mmap = mmap
        self.transform = transform
        if self.channels != 6:
            raise ValueError(f"Expected 6 channels, got {channels}")

    def __len__(self): return len(self.files)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, str]:
        path = self.files[idx]
        arr = np.load(path)
        t = torch.from_numpy(arr.copy()).float()
        # arr = np.load(path, mmap_mode="r" if self.mmap else None)
        # chw = ensure_chw(arr, self.channels)
        # t = torch.from_numpy(chw).float()
        if self.transform:
            t = self.transform(t)
        return t, path


# --------------------------- Model Loading ---------------------------------- #

def build_and_load_model(ckpt_path: str, device: torch.device,
                         depths: List[int], dims: List[int]):
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location=device)

    genotype_map = checkpoint.get("genotype_map", {})
    in_channels = int(checkpoint.get("in_channels", 6))

    model = ConvNeXtCBAMClassifier(
        in_channels=in_channels, class_num=len(genotype_map) or 2,
        depths=depths, dims=dims).to(device)

    state = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state, strict=True)
    model.eval()
    return model, genotype_map, in_channels


# --------------------------- Inference Core --------------------------------- #

# @torch.inference_mode()
def run_inference(model, dl, device, class_names: List[str],
                  total_files: int, no_probs: bool = False):
    results = []
    processed, t0 = 0, time.time()

    with tqdm(total=total_files, desc="Infer", unit="file", dynamic_ncols=True, leave=True) as bar:
        with torch.no_grad():
            for images, paths in dl:
                images = images.to(device)
                outputs = model(images)
                if isinstance(outputs, tuple):
                    outputs = outputs[0]

                # probs = torch.softmax(outputs, dim=1)
                # top_prob, top_idx = probs.max(dim=1)
                #
                # if no_probs:
                #     for i, pth in enumerate(paths):
                #         pred_idx = int(top_idx[i])
                #         pred_name = class_names[pred_idx] if pred_idx < len(class_names) else str(pred_idx)
                #         results.append({
                #             "path": pth,
                #             "pred_idx": pred_idx,
                #             "pred_class": pred_name,
                #             "pred_prob": float(top_prob[i]),
                #         })
                # else:
                #     probs_cpu = probs.cpu().numpy()
                #     for i, pth in enumerate(paths):
                #         pred_idx = int(top_idx[i])
                #         pred_name = class_names[pred_idx] if pred_idx < len(class_names) else str(pred_idx)
                #         prob_dict = {class_names[j]: float(probs_cpu[i, j]) for j in range(len(class_names))}
                #         results.append({
                #             "path": pth,
                #             "pred_idx": pred_idx,
                #             "pred_class": pred_name,
                #             "pred_prob": float(top_prob[i]),
                #             "probs": prob_dict
                #         })

                processed += len(paths)
                bar.update(len(paths))
                elapsed = time.time() - t0
                speed = processed / max(1e-9, elapsed)
                eta = (total_files - processed) / max(1e-9, speed)
                bar.set_postfix_str(f"speed={speed:.1f} file/s ETA={_format_eta(eta)}")

    return results


def write_outputs(results, csv_path: str, json_path: str):
    if json_path:
        with open(json_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Wrote JSON predictions: {json_path}")

    if csv_path:
        import csv
        any_probs = any("probs" in r for r in results)
        header = ["path", "pred_idx", "pred_class", "pred_prob"]
        class_cols = []
        if any_probs:
            all_classes = set()
            for r in results:
                if "probs" in r:
                    all_classes.update(r["probs"].keys())
            class_cols = sorted(all_classes)
            header += class_cols

        with open(csv_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(header)
            for r in results:
                row = [r["path"], r["pred_idx"], r["pred_class"], f"{r['pred_prob']:.6f}"]
                if any_probs:
                    row.extend([f"{r['probs'].get(c, 0.0):.6f}" for c in class_cols])
                w.writerow(row)
        print(f"Wrote CSV predictions: {csv_path}")


# ----------------------------- Main ----------------------------------------- #

def main():
    p = argparse.ArgumentParser(description="Infer over .npy files (Dataset-normalized, simplified).")
    p.add_argument("--input_dir", required=True)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--output_csv", default="predictions.csv")
    p.add_argument("--output_json", default="predictions.json")
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    p.add_argument("--compile", action="store_true")
    p.add_argument("--no_probs", action="store_true")
    p.add_argument("--mmap", action="store_true")
    p.add_argument("--depths", type=int, nargs="+", default=[3, 3, 27, 3])
    p.add_argument("--dims", type=int, nargs="+", default=[192, 384, 768, 1536])
    p.add_argument("--sample_info_k", type=int, default=3)
    args = p.parse_args()

    # Device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True

    # Build model
    print(f"Loading Model from {args.ckpt}")
    model, genotype_map, in_channels = build_and_load_model(args.ckpt, device, args.depths, args.dims)
    if in_channels != 6:
        raise ValueError(f"Expected 6-channel input, got {in_channels}")

    if args.compile:
        try: model = torch.compile(model)
        except Exception as e: print(f"torch.compile not enabled: {e}")

    # Class names
    if len(genotype_map) == 0:
        num_classes = next((m.out_features for _, m in model.named_modules() if isinstance(m, torch.nn.Linear)), 2)
        class_names = [str(i) for i in range(num_classes)]
    else:
        class_names = [None] * len(genotype_map)
        for cname, cidx in genotype_map.items():
            if 0 <= cidx < len(class_names):
                class_names[cidx] = str(cname)
        class_names = [c if c else str(i) for i, c in enumerate(class_names)]

    # Files
    files = list_npy_files_parallel(args.input_dir, workers=args.num_workers)
    if not files: raise SystemExit(f"No .npy files found in {args.input_dir}")

    print_data_info(files, in_channels, args.mmap, args.sample_info_k)

    # Dataset & Loader
    print(f"Building Data Loader...")
    from torchvision import transforms
    norm_transform = transforms.Normalize(mean=VAL_MEAN.tolist(), std=VAL_STD.tolist())
    ds = NpyDirDataset(files, in_channels, mmap=args.mmap, transform=norm_transform)
    dl = DataLoader(ds, batch_size=args.batch_size, num_workers=args.num_workers,
                    pin_memory=True, shuffle=False)

    # Run
    t0 = time.time()
    results = run_inference(model, dl, device, class_names, len(files), args.no_probs)
    elapsed = time.time() - t0
    print(f"\nFinished {len(files)} files in {elapsed:.2f}s ({len(files)/elapsed:.2f} file/s)")

    # Save
    write_outputs(results,
                  csv_path=args.output_csv if args.output_csv else None,
                  json_path=args.output_json if args.output_json else None)


if __name__ == "__main__":
    main()