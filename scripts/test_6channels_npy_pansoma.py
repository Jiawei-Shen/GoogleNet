#!/usr/bin/env python3
import argparse, os, sys, json, time, csv
from typing import List
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

THIS_DIR = os.path.abspath(os.path.dirname(__file__))
sys.path.append(os.path.abspath(os.path.join(THIS_DIR, "..")))
from mynet import ConvNeXtCBAMClassifier  # noqa: E402

# Same stats as before
VAL_MEAN = torch.tensor([18.417816162109375, 12.649129867553711, -0.5452527403831482,
                         24.723854064941406, 4.690611362457275, 0.2813551473402196], dtype=torch.float32)
VAL_STD  = torch.tensor([25.028322219848633, 14.809632301330566, 0.6181337833404541,
                         29.972835540771484, 7.9231791496276855, 0.7659083659074717], dtype=torch.float32)

# ----------------------------- Utilities ------------------------------------- #
def _scan_tree(start_dir: str) -> List[str]:
    stack = [start_dir]; out: List[str] = []
    while stack:
        d = stack.pop()
        try:
            with os.scandir(d) as it:
                for e in it:
                    try:
                        if e.is_dir(follow_symlinks=False): stack.append(e.path)
                        elif e.name.lower().endswith(".npy"): out.append(e.path)
                    except OSError: continue
        except (PermissionError, FileNotFoundError):
            continue
    return out

def _format_eta(seconds: float) -> str:
    s = int(max(0, seconds)); h, rem = divmod(s, 3600); m, s = divmod(rem, 60)
    return f"{h:d}h{m:02d}m{s:02d}s" if h else (f"{m:d}m{s:02d}s" if m else f"{s:d}s")

def list_npy_files_parallel(root: str, workers: int) -> List[str]:
    root = os.path.abspath(os.path.expanduser(root)); top_dirs, files = [], []
    try:
        with os.scandir(root) as it:
            for e in it:
                try:
                    if e.is_dir(follow_symlinks=False): top_dirs.append(e.path)
                    elif e.name.lower().endswith(".npy"): files.append(e.path)
                except OSError: continue
    except FileNotFoundError:
        return []
    if workers <= 1 or not top_dirs:
        for d in top_dirs: files.extend(_scan_tree(d))
        return sorted(files)
    start = time.time()
    with ThreadPoolExecutor(max_workers=workers) as ex, \
         tqdm(total=len(top_dirs), desc="Scan", unit="dir", leave=False, dynamic_ncols=True) as bar:
        futures = [ex.submit(_scan_tree, d) for d in top_dirs]
        for fut in as_completed(futures):
            files.extend(fut.result()); bar.update(1)
            done = bar.n; elapsed = time.time() - start
            speed = done / max(1e-9, elapsed); eta = (len(top_dirs) - done) / max(1e-9, speed)
            bar.set_postfix_str(f"speed={speed:.1f} dir/s ETA={_format_eta(eta)}")
    return sorted(files)

def ensure_chw(x: np.ndarray, channels: int) -> np.ndarray:
    if x.ndim == 3:
        if x.shape[0] == channels: chw = x
        elif x.shape[-1] == channels: chw = np.transpose(x, (2, 0, 1))
        else: raise ValueError(f"Unexpected shape {x.shape}, expected {channels} channels")
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
    def __init__(self, files: List[str], channels: int, mmap: bool = False):
        self.files = files; self.channels = channels; self.mmap = mmap
        if self.channels != 6:
            raise ValueError(f"Expected 6 channels, got {channels}")
    def __len__(self): return len(self.files)
    def __getitem__(self, idx: int):
        path = self.files[idx]
        arr = np.load(path, mmap_mode="r" if self.mmap else None)
        chw = ensure_chw(arr, self.channels)
        return torch.from_numpy(chw).float(), path

# --------------------------- Model Loading ---------------------------------- #
def build_and_load_model(ckpt_path: str, device: torch.device,
                         depths: List[int], dims: List[int]):
    if not os.path.isfile(ckpt_path): raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    genotype_map = checkpoint.get("genotype_map", {})
    in_channels = int(checkpoint.get("in_channels", 6))
    model = ConvNeXtCBAMClassifier(in_channels=in_channels,
                                   class_num=len(genotype_map) or 2,
                                   depths=depths, dims=dims).to(device)
    state = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state, strict=True)
    model.eval()
    return model, genotype_map, in_channels

# --------------------------- Inference Core --------------------------------- #
@torch.inference_mode()
def run_inference(model, dl, device, class_names: List[str],
                  total_files: int, no_probs: bool,
                  gpu_norm: bool, mean: torch.Tensor, std: torch.Tensor,
                  amp_mode: str):
    """Return a list[dict] of results; NO writing inside."""
    results = []
    any_probs = (not no_probs)

    # Prepare GPU mean/std if needed
    mean_d = std_d = None
    if gpu_norm:
        mean_d = mean.to(device, non_blocking=True).view(1, -1, 1, 1)
        std_d  = std.to(device, non_blocking=True).view(1, -1, 1, 1)

    scaler_dtype = None
    if amp_mode == "bf16" and torch.cuda.is_available() and device.type == "cuda":
        scaler_dtype = torch.bfloat16
    elif amp_mode == "fp16" and torch.cuda.is_available() and device.type == "cuda":
        scaler_dtype = torch.float16

    processed, t0 = 0, time.time()
    with tqdm(total=total_files, desc="Infer", unit="file", dynamic_ncols=True, leave=True) as bar:
        for images, paths in dl:
            # Normalize then move OR move then normalize (to avoid device mismatch)
            if gpu_norm:
                images = images.to(device, non_blocking=True)
                images = (images - mean_d) / std_d
            else:
                images.sub_(VAL_MEAN.view(-1, 1, 1)).div_(VAL_STD.view(-1, 1, 1))  # CPU-side
                images = images.to(device, non_blocking=True)

            # Forward (optional AMP)
            if scaler_dtype is not None:
                with torch.autocast(device_type="cuda", dtype=scaler_dtype):
                    outputs = model(images)
            else:
                outputs = model(images)
            if isinstance(outputs, tuple): outputs = outputs[0]

            if any_probs:
                probs = torch.softmax(outputs, dim=1)
                top_prob, top_idx = probs.max(dim=1)
                probs_cpu = probs.cpu().numpy()
                top_prob_cpu = top_prob.cpu().numpy()
            else:
                top_idx = outputs.argmax(dim=1)

            # Collect results in memory
            for i, pth in enumerate(paths):
                pred_idx = int(top_idx[i])
                pred_name = class_names[pred_idx] if pred_idx < len(class_names) else str(pred_idx)
                rec = {"path": pth, "pred_idx": pred_idx, "pred_class": pred_name}
                if any_probs:
                    rec["pred_prob"] = float(top_prob_cpu[i])
                    rec["probs"] = {class_names[j]: float(probs_cpu[i, j]) for j in range(len(class_names))}
                results.append(rec)

            processed += len(paths)
            elapsed = time.time() - t0; speed = processed / max(1e-9, elapsed)
            eta = (total_files - processed) / max(1e-9, speed)
            bar.update(len(paths))
            bar.set_postfix_str(f"speed={speed:.1f} file/s ETA={_format_eta(eta)}")

    return results

def write_outputs(results: List[dict], class_names: List[str],
                  csv_path: str | None, json_path: str | None, no_probs: bool):
    if not results:
        return
    # JSON (one shot)
    if json_path:
        with open(json_path, "w") as jf:
            json.dump(results, jf, indent=2)
        print(f"Wrote JSON predictions: {json_path}")

    # CSV (one shot)
    if csv_path:
        any_probs = (not no_probs) and any(("probs" in r) for r in results)
        header = ["path", "pred_idx", "pred_class", "pred_prob"]
        if any_probs:
            header += class_names
        with open(csv_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(header)
            for r in results:
                row = [r["path"], r["pred_idx"], r["pred_class"]]
                row.append(f"{r.get('pred_prob', 0.0):.6f}" if "pred_prob" in r else "")
                if any_probs:
                    row.extend([f"{r['probs'].get(c, 0.0):.6f}" for c in class_names])
                w.writerow(row)
        print(f"Wrote CSV predictions: {csv_path}")

# ----------------------------- Main ----------------------------------------- #
def main():
    p = argparse.ArgumentParser(description="Fast inference over .npy files (collect first, write after).")
    p.add_argument("--input_dir", required=True)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--output_csv", default="predictions.csv")
    p.add_argument("--output_json", default="predictions.json")
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--prefetch_factor", type=int, default=4)
    p.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    p.add_argument("--compile", action="store_true")
    p.add_argument("--no_probs", action="store_true")
    p.add_argument("--mmap", action="store_true")
    p.add_argument("--depths", type=int, nargs="+", default=[3, 3, 27, 3])
    p.add_argument("--dims", type=int, nargs="+", default=[192, 384, 768, 1536])
    p.add_argument("--sample_info_k", type=int, default=3)
    try:
        from argparse import BooleanOptionalAction
        p.add_argument("--gpu_norm", action=BooleanOptionalAction, default=True,
                       help="Normalize on GPU (default: True). Use --no-gpu-norm to disable.")
    except Exception:
        p.add_argument("--gpu_norm", action="store_true", default=True,
                       help="Normalize on GPU (default: True).")
    p.add_argument("--amp", choices=["off", "bf16", "fp16"], default="off", help="Mixed precision mode (CUDA only)")
    args = p.parse_args()

    # Device + perf knobs
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        try: torch.set_float32_matmul_precision("high")
        except Exception: pass

    if device.type != "cuda" and args.gpu_norm:
        print("NOTE: CUDA not available; falling back to CPU-side normalization.")
        args.gpu_norm = False

    # Build model
    print(f"Loading Model from {args.ckpt}")
    model, genotype_map, in_channels = build_and_load_model(args.ckpt, device, args.depths, args.dims)
    if in_channels != 6:
        raise ValueError(f"Expected 6-channel input, got {in_channels}")

    if args.compile:
        try:
            model = torch.compile(model, mode="reduce-overhead")
        except Exception as e:
            print(f"torch.compile not enabled: {e}")

    # Class names
    if len(genotype_map) == 0:
        num_classes = next((m.out_features for _, m in model.named_modules() if isinstance(m, torch.nn.Linear)), 2)
        class_names = [str(i) for i in range(num_classes)]
    else:
        class_names = [None] * len(genotype_map)
        for cname, cidx in genotype_map.items():
            if 0 <= cidx < len(class_names): class_names[cidx] = str(cname)
        class_names = [c if c else str(i) for i, c in enumerate(class_names)]

    # Files
    files = list_npy_files_parallel(args.input_dir, workers=args.num_workers)
    if not files: raise SystemExit(f"No .npy files found in {args.input_dir}")
    print_data_info(files, in_channels, args.mmap, args.sample_info_k)

    # Dataset & Loader
    ds = NpyDirDataset(files, in_channels, mmap=args.mmap)
    dl = DataLoader(
        ds,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(args.num_workers > 0),
        prefetch_factor=args.prefetch_factor if args.num_workers > 0 else None,
        shuffle=False
    )

    # Run (collect) -> Write (after)
    t0 = time.time()
    results = run_inference(
        model, dl, device, class_names, len(files), args.no_probs,
        gpu_norm=args.gpu_norm, mean=VAL_MEAN, std=VAL_STD, amp_mode=args.amp
    )
    elapsed = time.time() - t0
    print(f"\nFinished {len(files)} files in {elapsed:.2f}s ({len(files)/elapsed:.2f} file/s)")

    write_outputs(results, class_names,
                  csv_path=(args.output_csv or None),
                  json_path=(args.output_json or None),
                  no_probs=args.no_probs)

if __name__ == "__main__":
    main()
