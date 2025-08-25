#!/usr/bin/env python3
"""
Batch inference over a directory of .npy files (recursively) with print-only logs.

Example:
    python infer_npy_dir.py \
      --input_dir /path/to/npy_root \
      --ckpt /path/to/model_best.pth \
      --output_csv preds.csv \
      --output_json preds.json \
      --batch_size 128 --num_workers 8 --fp16 --mmap \
      --print_every_n 1000 --sample_info_k 3
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
                            name = e.name
                            if len(name) >= 4 and name[-4:].lower() == ".npy":
                                out.append(e.path)
                    except OSError:
                        continue
        except (PermissionError, FileNotFoundError):
            continue
    return out

def list_npy_files_parallel(root: str, workers: int, progress_every: int = 500) -> List[str]:
    """
    Parallel file discovery using exactly `workers` threads.
    """
    root = os.path.abspath(os.path.expanduser(root))

    # Collect top-level dirs + any .npy directly under root
    top_dirs: List[str] = []
    files: List[str] = []
    try:
        with os.scandir(root) as it:
            for e in it:
                try:
                    if e.is_dir(follow_symlinks=False):
                        top_dirs.append(e.path)
                    else:
                        name = e.name
                        if len(name) >= 4 and name[-4:].lower() == ".npy":
                            files.append(e.path)
                except OSError:
                    continue
    except FileNotFoundError:
        return []

    if workers <= 1 or not top_dirs:
        for d in top_dirs:
            files.extend(_scan_tree(d))
        files.sort()
        return files

    scanned = 0
    with ThreadPoolExecutor(max_workers=max(1, workers)) as ex:
        futures = [ex.submit(_scan_tree, d) for d in top_dirs]
        for fut in as_completed(futures):
            files.extend(fut.result())
            scanned += 1
            if progress_every and (scanned % progress_every == 0 or scanned == len(top_dirs)):
                print(f"[scan] {scanned}/{len(top_dirs)} top-level dirs done")

    files.sort()
    return files



def ensure_chw(x: np.ndarray, channels: int, tile_single_channel: bool = False) -> np.ndarray:
    """
    Convert numpy array to float32 CHW.
    Accepts [C,H,W], [H,W,C], [H,W]; can tile single-channel if requested.
    """
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
    return chw


def softmax_np(logits: np.ndarray, axis: int = -1) -> np.ndarray:
    m = logits.max(axis=axis, keepdims=True)
    y = np.exp(logits - m)
    return y / y.sum(axis=axis, keepdims=True)


def print_data_info(
    files: List[str],
    in_channels: int,
    normalize: bool,
    tile_single_channel: bool,
    mmap: bool,
    sample_info_k: int,
) -> None:
    print("\n=== Data Information ===")
    print(f"Total .npy files: {len(files)}")
    print(f"Expected input channels: {in_channels}")
    print(f"normalize={normalize} | tile_single_channel={tile_single_channel} | mmap={mmap}")
    k = min(sample_info_k, len(files))
    if k == 0:
        print("No files to sample.")
        print("=" * 26)
        return

    print(f"Sampling {k} file(s) for shape/dtype (using mmap to avoid loading data):")
    for i in range(k):
        p = files[i]
        try:
            arr = np.load(p, allow_pickle=False, mmap_mode="r" if mmap else None)
            print(f"  [{i+1}] {p}")
            print(f"      shape={tuple(arr.shape)}, dtype={arr.dtype}, ndim={arr.ndim}")
        except Exception as e:
            print(f"  [{i+1}] {p}")
            print(f"      <error reading header> {e}")
    print("=" * 26)


# ---------------------------- Dataset ---------------------------------------- #

class NpyDirDataset(Dataset):
    def __init__(
        self,
        files: List[str],
        channels: int,
        normalize: bool = False,
        tile_single_channel: bool = False,
        mmap: bool = False,
    ):
        self.files = files
        self.channels = channels
        self.normalize = normalize
        self.tile_single_channel = tile_single_channel
        self.mmap = mmap

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, str]:
        path = self.files[idx]
        arr = np.load(path, allow_pickle=False, mmap_mode=('r' if self.mmap else None))

        if arr.dtype == np.uint8 and self.normalize:
            arr = arr.astype(np.float32) / 255.0

        chw = ensure_chw(arr, self.channels, tile_single_channel=self.tile_single_channel)
        tensor = torch.from_numpy(chw)  # float32 CHW
        return tensor, path


# --------------------------- Model Loading ----------------------------------- #

def build_and_load_model(
    ckpt_path: str,
    device: torch.device,
    depths: List[int],
    dims: List[int]
) -> Tuple[torch.nn.Module, Dict[str, int], int]:
    """
    Returns: (model.eval() on device, genotype_map, in_channels)
    """
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    checkpoint = torch.load(ckpt_path, map_location=device)

    genotype_map = checkpoint.get("genotype_map", {})
    in_channels = int(checkpoint.get("in_channels", 6))

    model = ConvNeXtCBAMClassifier(in_channels=in_channels, class_num=len(genotype_map) or 2,
                                   depths=depths, dims=dims).to(device)

    state = checkpoint.get("model_state_dict", checkpoint)
    try:
        model.load_state_dict(state, strict=True)
    except RuntimeError as e:
        raise RuntimeError(
            "Failed to load state dict. If you trained with non-default depths/dims, "
            "re-run with matching --depths and --dims. Original error:\n" + str(e)
        )

    model.eval()
    return model, genotype_map, in_channels


# --------------------------- Inference Core ---------------------------------- #

@torch.inference_mode()
def run_inference(
    model: torch.nn.Module,
    dl: DataLoader,
    device: torch.device,
    fp16: bool,
    class_names: List[str],
    total_files: int,
    print_every_n: int,
) -> List[Dict[str, Any]]:
    results: List[Dict[str, Any]] = []
    use_amp = fp16 and device.type == "cuda"

    processed = 0
    next_milestone = max(1, print_every_n)
    t0 = time.time()

    for images, paths in tqdm(dl, desc="Infer", leave=True):
        images = images.to(device, non_blocking=True)
        if use_amp:
            images = images.half()

        with (torch.cuda.amp.autocast() if use_amp else torch.no_grad()):
            outputs = model(images)
            if isinstance(outputs, tuple):
                outputs = outputs[0]

        logits = outputs.detach().float().cpu().numpy()
        probs = softmax_np(logits, axis=1)
        top_idx = probs.argmax(axis=1)
        top_prob = probs.max(axis=1)

        for i in range(len(paths)):
            pred_idx = int(top_idx[i])
            pred_name = class_names[pred_idx] if 0 <= pred_idx < len(class_names) else str(pred_idx)
            prob_dict = {class_names[j]: float(probs[i, j]) for j in range(len(class_names))}
            results.append({
                "path": paths[i],
                "pred_idx": pred_idx,
                "pred_class": pred_name,
                "pred_prob": float(top_prob[i]),
                "probs": prob_dict
            })

        # ---- Progress prints ----
        processed += len(paths)
        if processed >= next_milestone or processed == total_files:
            elapsed = time.time() - t0
            speed = processed / max(1e-6, elapsed)
            pct = 100.0 * processed / max(1, total_files)
            remaining = total_files - processed
            eta = remaining / max(1e-6, speed)
            print(f"[{processed}/{total_files} | {pct:.1f}%] elapsed={elapsed:.1f}s "
                  f"| speed={speed:.1f} files/s | ETA={eta:.1f}s")
            next_milestone += print_every_n

    return results


def write_outputs(results: List[Dict[str, Any]], csv_path: str, json_path: str) -> None:
    if json_path:
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"Wrote JSON predictions: {os.path.abspath(json_path)}")

    if csv_path:
        import csv
        all_classes = set()
        for r in results:
            all_classes.update(r["probs"].keys())
        header = ["path", "pred_idx", "pred_class", "pred_prob"] + sorted(all_classes)

        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(header)
            for r in results:
                row = [r["path"], r["pred_idx"], r["pred_class"], f"{r['pred_prob']:.6f}"]
                for cname in sorted(all_classes):
                    row.append(f"{r['probs'].get(cname, 0.0):.6f}")
                w.writerow(row)
        print(f"Wrote CSV predictions:  {os.path.abspath(csv_path)}")


# ----------------------------- Main ------------------------------------------ #

def main():
    parser = argparse.ArgumentParser(description="Infer over .npy files (recursively) with ConvNeXtCBAMClassifier")
    parser.add_argument("--input_dir", required=True, type=str, help="Root dir to scan for .npy files")
    parser.add_argument("--ckpt", required=True, type=str, help="Path to checkpoint .pth")
    parser.add_argument("--output_csv", default="predictions.csv", type=str, help="CSV output path (or '' to skip)")
    parser.add_argument("--output_json", default="predictions.json", type=str, help="JSON output path (or '' to skip)")
    parser.add_argument("--batch_size", default=64, type=int)
    parser.add_argument("--num_workers", default=8, type=int)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"], help="Device choice")
    parser.add_argument("--fp16", action="store_true", help="Enable half-precision inference (CUDA only)")
    parser.add_argument("--normalize", action="store_true",
                        help="If input dtype is uint8, scale to [0,1] by /255.0")
    parser.add_argument("--tile_single_channel", action="store_true",
                        help="If an array is single-channel, tile it to match required channels")
    parser.add_argument("--mmap", action="store_true", help="Load .npy via mmap_mode='r' (lower RAM peak)")
    parser.add_argument("--depths", type=int, nargs="+", default=[3, 3, 27, 3],
                        help="ConvNeXt stage depths used in training (match if you changed it)")
    parser.add_argument("--dims", type=int, nargs="+", default=[192, 384, 768, 1536],
                        help="ConvNeXt dims used in training (match if you changed it)")
    # Print config
    parser.add_argument("--print_every_n", type=int, default=500,
                        help="Print a progress line every N files")
    parser.add_argument("--sample_info_k", type=int, default=3,
                        help="Print header info (shape/dtype) for the first K files")
    args = parser.parse_args()

    # Resolve device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif args.device == "cuda":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if device.type == "cpu":
            print("CUDA not available; falling back to CPU.")
    else:
        device = torch.device("cpu")

    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass

    # Build model & load weights
    model, genotype_map, in_channels = build_and_load_model(
        ckpt_path=args.ckpt, device=device, depths=args.depths, dims=args.dims
    )

    # Class names
    if len(genotype_map) == 0:
        print("WARNING: 'genotype_map' missing in checkpoint; using numeric class names.")
        num_classes = None
        for _, m in model.named_modules():
            if isinstance(m, torch.nn.Linear):
                num_classes = m.out_features
                break
        if num_classes is None:
            num_classes = 2
        class_names = [str(i) for i in range(num_classes)]
    else:
        class_names = [None] * len(genotype_map)
        for cname, cidx in genotype_map.items():
            if 0 <= int(cidx) < len(class_names):
                class_names[cidx] = str(cname)
        for i in range(len(class_names)):
            if class_names[i] is None:
                class_names[i] = str(i)

    # Gather files
    # replace previous file listing
    files = list_npy_files_parallel(args.input_dir, workers=args.num_workers, progress_every=10000)

    if len(files) == 0:
        raise SystemExit(f"No .npy files found under: {args.input_dir}")

    # Startup prints
    print("=== Inference Configuration ===")
    print(f"Device: {device} | FP16: {bool(args.fp16 and device.type=='cuda')}")
    print(f"Input dir: {os.path.abspath(args.input_dir)}")
    print(f"Found files: {len(files)}")
    print(f"In-channels: {in_channels}")
    print(f"Classes ({len(class_names)}): {class_names}")
    print(f"Batch size: {args.batch_size} | Workers: {args.num_workers} | mmap={args.mmap}")
    print("=" * 30)

    # Data info peek
    print_data_info(
        files=files,
        in_channels=in_channels,
        normalize=args.normalize,
        tile_single_channel=args.tile_single_channel,
        mmap=args.mmap,
        sample_info_k=max(0, args.sample_info_k),
    )

    # Dataset & DataLoader
    ds = NpyDirDataset(
        files=files,
        channels=in_channels,
        normalize=args.normalize,
        tile_single_channel=args.tile_single_channel,
        mmap=args.mmap,
    )
    dl = DataLoader(
        ds,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        shuffle=False,
        drop_last=False,
        collate_fn=lambda batch: (
            torch.stack([b[0] for b in batch], dim=0),
            [b[1] for b in batch],
        ),
    )

    if args.fp16 and device.type == "cuda":
        model = model.half()

    # Run
    t0 = time.time()
    results = run_inference(
        model=model,
        dl=dl,
        device=device,
        fp16=args.fp16,
        class_names=class_names,
        total_files=len(files),
        print_every_n=max(1, args.print_every_n),
    )
    elapsed = time.time() - t0
    speed = len(files) / max(1e-6, elapsed)
    print(f"\nFinished inference: {len(files)} files | elapsed={elapsed:.2f}s | avg speed={speed:.2f} files/s")

    # Save
    csv_path = args.output_csv if args.output_csv.strip() else None
    json_path = args.output_json if args.output_json.strip() else None
    write_outputs(results, csv_path=csv_path, json_path=json_path)

    print(f"\nDone. Files scanned: {len(files)}")
    if csv_path:
        print(f"CSV:  {os.path.abspath(csv_path)}")
    if json_path:
        print(f"JSON: {os.path.abspath(json_path)}")


if __name__ == "__main__":
    main()
