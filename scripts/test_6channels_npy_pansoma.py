#!/usr/bin/env python3
"""
Batch inference over a directory of .npy files (recursively) with single-line flashing progress,
using the SAME preprocessing as the validation loader (channel-wise Normalize).

Now with producer→consumer:
  • GPU producer enqueues logits
  • Multiple CPU consumers do softmax/argmax and build result dicts in parallel

Example:
    python infer_npy_dir.py \
      --input_dir /path/to/npy_root \
      --ckpt /path/to/model_best.pth \
      --output_csv preds.csv \
      --output_json preds.json \
      --batch_size 128 --num_workers 8 --mmap --postproc_workers 8
"""

import argparse
import os
import sys
import json
import time
from typing import List, Tuple, Dict, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
import multiprocessing as mp

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# Import your model the same way as in training
THIS_DIR = os.path.abspath(os.path.dirname(__file__))
sys.path.append(os.path.abspath(os.path.join(THIS_DIR, "..")))
from mynet import ConvNeXtCBAMClassifier  # noqa: E402

# --- EXACT SAME NORMALIZATION AS VAL ---

# COLO829T Testing
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
                            name = e.name
                            if len(name) >= 4 and name[-4:].lower() == ".npy":
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
    """
    Parallel file discovery using exactly `workers` threads, with a single-line tqdm bar.
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

    # If no subdirs, just return what we found at root
    if workers <= 1 or not top_dirs:
        for d in top_dirs:
            files.extend(_scan_tree(d))
        files.sort()
        return files

    start = time.time()
    with ThreadPoolExecutor(max_workers=max(1, workers)) as ex, \
         tqdm(total=len(top_dirs), desc="Scan", unit="dir", leave=False, dynamic_ncols=True) as bar:
        futures = [ex.submit(_scan_tree, d) for d in top_dirs]
        for fut in as_completed(futures):
            files.extend(fut.result())
            done = bar.n + 1
            elapsed = time.time() - start
            speed = done / max(1e-9, elapsed)
            eta = (len(top_dirs) - done) / max(1e-9, speed)
            bar.set_postfix_str(f"speed={speed:.1f} dir/s ETA={_format_eta(eta)}")
            bar.update(1)

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
    tile_single_channel: bool,
    mmap: bool,
    sample_info_k: int,
) -> None:
    print("\n=== Data Information ===")
    print(f"Total .npy files: {len(files)}")
    print(f"Expected input channels: {in_channels}")
    print(f"tile_single_channel={tile_single_channel} | mmap={mmap}")
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
        tile_single_channel: bool = False,
        mmap: bool = False,
    ):
        self.files = files
        self.channels = channels
        self.tile_single_channel = tile_single_channel
        self.mmap = mmap

        if self.channels != 6:
            raise ValueError(f"inference expects 6 channels to match training/val; got {self.channels}")

        # Precompute mean/std on CPU for CHW broadcasting
        self.mean = VAL_MEAN.view(-1, 1, 1)  # (6,1,1)
        self.std = VAL_STD.view(-1, 1, 1)    # (6,1,1)

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, str]:
        path = self.files[idx]
        arr = np.load(path, allow_pickle=False, mmap_mode=('r' if self.mmap else None))

        chw = ensure_chw(arr, self.channels, tile_single_channel=self.tile_single_channel)

        t = torch.from_numpy(chw).float()  # CHW float32
        # --- EXACT SAME NORMALIZATION AS VAL ---
        t = (t - self.mean) / self.std

        return t, path


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


# --------------------------- Producer→Consumer Inference ---------------------- #

def _postproc_worker(in_q, out_q, class_names):
    """
    CPU worker: receives (paths: List[str], logits: np.ndarray [B, C])
    and emits list[dict] with 'probs' included (to match write_outputs()).
    """
    try:
        while True:
            item = in_q.get()
            if item is None:
                break
            paths, logits = item
            probs = softmax_np(logits, axis=1)
            top_idx = probs.argmax(axis=1)
            top_prob = probs.max(axis=1)

            out_batch = []
            for i, pth in enumerate(paths):
                pred_idx = int(top_idx[i])
                pred_name = class_names[pred_idx] if 0 <= pred_idx < len(class_names) else str(pred_idx)
                rec = {
                    "path": pth,
                    "pred_idx": pred_idx,
                    "pred_class": pred_name,
                    "pred_prob": float(top_prob[i]),
                    "probs": {class_names[j]: float(probs[i, j]) for j in range(len(class_names))}
                }
                out_batch.append(rec)

            out_q.put(out_batch)
    except KeyboardInterrupt:
        pass
    finally:
        out_q.put(None)


@torch.inference_mode()
def run_inference_pc(
    model: torch.nn.Module,
    dl: DataLoader,
    device: torch.device,
    fp16: bool,
    class_names: List[str],
    total_files: int,
    *,
    postproc_workers: int = 4,
) -> List[Dict[str, Any]]:
    """
    Producer (GPU): model→logits
    Consumers (CPU mp processes): softmax/argmax/build result dicts (with 'probs')
    """
    results: List[Dict[str, Any]] = []
    use_amp = (fp16 and device.type == "cuda")

    # 'spawn' plays nicest with CUDA
    ctx = mp.get_context("spawn")
    in_q = ctx.Queue(maxsize=max(2, postproc_workers * 2))
    out_q = ctx.Queue(maxsize=max(2, postproc_workers * 2))

    # Start workers
    workers = []
    for _ in range(max(1, postproc_workers)):
        p = ctx.Process(target=_postproc_worker, args=(in_q, out_q, class_names), daemon=True)
        p.start()
        workers.append(p)

    finished_workers = 0
    processed = 0
    t0 = time.time()

    with tqdm(total=total_files, desc="Infer", unit="file", dynamic_ncols=True, leave=True) as bar:
        for images, paths in dl:
            images = images.to(device, non_blocking=True)
            if use_amp:
                with torch.cuda.amp.autocast():
                    outputs = model(images)
            else:
                outputs = model(images)

            if isinstance(outputs, tuple):
                outputs = outputs[0]

            # Move logits to CPU as numpy for workers
            logits = outputs.detach().float().cpu().numpy()
            in_q.put((paths, logits))

            processed += len(paths)
            bar.update(len(paths))
            elapsed = time.time() - t0
            speed = processed / max(1e-9, elapsed)
            eta = (total_files - processed) / max(1e-9, speed)
            bar.set_postfix_str(f"speed={speed:.1f} file/s ETA={_format_eta(eta)}")

        # Tell workers we're done
        for _ in workers:
            in_q.put(None)

        # Drain results until all workers send sentinel
        while finished_workers < len(workers):
            out = out_q.get()
            if out is None:
                finished_workers += 1
            else:
                results.extend(out)

    # Join (best-effort)
    for p in workers:
        p.join(timeout=1.0)

    return results


# --------------------------- Inference Core (legacy) -------------------------- #
# (Keeping your original for reference; the script now calls run_inference_pc)
@torch.inference_mode()
def run_inference(
    model: torch.nn.Module,
    dl: DataLoader,
    device: torch.device,
    fp16: bool,
    class_names: List[str],
    total_files: int,
) -> List[Dict[str, Any]]:
    results: List[Dict[str, Any]] = []
    use_amp = (fp16 and device.type == "cuda")

    processed = 0
    t0 = time.time()

    with tqdm(total=total_files, desc="Infer", unit="file", dynamic_ncols=True, leave=True) as bar:
        for images, paths in dl:
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

            processed += len(paths)
            bar.update(len(paths))
            elapsed = time.time() - t0
            speed = processed / max(1e-9, elapsed)
            eta = (total_files - processed) / max(1e-9, speed)
            bar.set_postfix_str(f"speed={speed:.1f} file/s ETA={_format_eta(eta)}")

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
    parser = argparse.ArgumentParser(description="Infer over .npy files (recursively) with ConvNeXtCBAMClassifier, matching val preprocessing, with producer→consumer post-processing.")
    parser.add_argument("--input_dir", required=True, type=str, help="Root dir to scan for .npy files")
    parser.add_argument("--ckpt", required=True, type=str, help="Path to checkpoint .pth")
    parser.add_argument("--output_csv", default="predictions.csv", type=str, help="CSV output path (or '' to skip)")
    parser.add_argument("--output_json", default="predictions.json", type=str, help="JSON output path (or '' to skip)")
    parser.add_argument("--batch_size", default=64, type=int)
    parser.add_argument("--num_workers", default=8, type=int)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"], help="Device choice")
    parser.add_argument("--fp16", action="store_true", help="Enable half-precision inference (CUDA only). For val parity, leave off.")
    parser.add_argument("--tile_single_channel", action="store_true", help="If an array is single-channel, tile it to 6 channels")
    parser.add_argument("--mmap", action="store_true", help="Load .npy via mmap_mode='r' (lower RAM peak)")
    parser.add_argument("--depths", type=int, nargs="+", default=[3, 3, 27, 3], help="ConvNeXt stage depths used in training")
    parser.add_argument("--dims", type=int, nargs="+", default=[192, 384, 768, 1536], help="ConvNeXt dims used in training")
    parser.add_argument("--sample_info_k", type=int, default=3, help="Print header info (shape/dtype) for the first K files")
    parser.add_argument("--postproc_workers", type=int, default=4, help="CPU processes for post-processing logits")
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
    if in_channels != 6:
        raise ValueError(f"Checkpoint expects in_channels={in_channels}; this script assumes 6-channel normalization.")

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

    # Gather files (single-line progress)
    files = list_npy_files_parallel(args.input_dir, workers=args.num_workers)
    if len(files) == 0:
        raise SystemExit(f"No .npy files found under: {args.input_dir}")

    # Startup prints
    print("=== Inference Configuration ===")
    print(f"Device: {device} | FP16: {bool(args.fp16 and device.type=='cuda')} (val uses FP32)")
    print(f"Input dir: {os.path.abspath(args.input_dir)}")
    print(f"Found files: {len(files)}")
    print(f"In-channels: {in_channels}")
    print(f"Classes ({len(class_names)}): {class_names}")
    print(f"Batch size: {args.batch_size} | Workers: {args.num_workers} | mmap={args.mmap} | postproc_workers={args.postproc_workers}")
    print("=" * 30)

    # Data info peek
    print_data_info(
        files=files,
        in_channels=in_channels,
        tile_single_channel=args.tile_single_channel,
        mmap=args.mmap,
        sample_info_k=max(0, args.sample_info_k),
    )

    # Dataset & DataLoader
    ds = NpyDirDataset(
        files=files,
        channels=in_channels,
        tile_single_channel=args.tile_single_channel,
        mmap=args.mmap,
    )
    dl = DataLoader(
        ds,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(args.num_workers > 0),
        prefetch_factor=(8 if args.num_workers > 0 else None),
        shuffle=False,
        drop_last=False,
        collate_fn=lambda batch: (
            torch.stack([b[0] for b in batch], dim=0),
            [b[1] for b in batch],
        ),
        multiprocessing_context=("forkserver" if args.num_workers > 0 else None),
    )

    # Keep model in FP32 unless explicitly asked otherwise
    if args.fp16 and device.type == "cuda":
        model = model.half()

    # Run (single-line flashing bar) with producer→consumer
    t0 = time.time()
    results = run_inference_pc(
        model=model,
        dl=dl,
        device=device,
        fp16=args.fp16,
        class_names=class_names,
        total_files=len(files),
        postproc_workers=args.postproc_workers,
    )
    elapsed = time.time() - t0
    speed = len(files) / max(1e-9, elapsed)
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
