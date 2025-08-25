#!/usr/bin/env python3
"""
Batch inference over a directory of .npy files (recursively).

Example:
    python infer_npy_dir.py \
        --input_dir /path/to/npy_root \
        --ckpt /path/to/model_best.pth \
        --output_csv preds.csv \
        --output_json preds.json \
        --batch_size 64 --num_workers 8 --fp16

Notes:
- Assumes your checkpoint (from the provided training script) includes:
    * 'model_state_dict'
    * 'genotype_map' (class_name -> idx)
    * 'in_channels'
- Model architecture depths/dims default to [3,3,27,3] / [192,384,768,1536].
  If you trained with different values, pass --depths / --dims to match.
"""

import argparse
import os
import sys
import json
import math
from typing import List, Tuple, Dict, Any

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# Make sure we can import your model class the same way as in training
THIS_DIR = os.path.abspath(os.path.dirname(__file__))
sys.path.append(os.path.abspath(os.path.join(THIS_DIR, "..")))
from mynet import ConvNeXtCBAMClassifier  # noqa: E402


# ----------------------------- Utilities ------------------------------------- #

def list_npy_files(root: str) -> List[str]:
    files = []
    for dp, _, fns in os.walk(root):
        for fn in fns:
            if fn.lower().endswith(".npy"):
                files.append(os.path.join(dp, fn))
    files.sort()
    return files


def ensure_chw(x: np.ndarray, channels: int, tile_single_channel: bool = False) -> np.ndarray:
    """
    Convert numpy array to float32 CHW.
    Accepts [C,H,W], [H,W,C], [H,W]; can tile single-channel if requested.
    """
    if x.ndim == 3:
        # Either CHW or HWC
        if x.shape[0] == channels:
            chw = x
        elif x.shape[-1] == channels:
            chw = np.transpose(x, (2, 0, 1))
        elif x.shape[0] == 1 and tile_single_channel and channels > 1:
            # [1,H,W] -> tile
            chw = np.tile(x, (channels, 1, 1))
        elif x.shape[-1] == 1 and tile_single_channel and channels > 1:
            # [H,W,1] -> [1,H,W] -> tile
            chw = np.transpose(x, (2, 0, 1))
            chw = np.tile(chw, (channels, 1, 1))
        else:
            raise ValueError(f"Cannot infer channel axis for shape {x.shape} with channels={channels}")
    elif x.ndim == 2:
        # [H,W]
        if tile_single_channel and channels > 1:
            chw = np.tile(x[None, ...], (channels, 1, 1))
        else:
            if channels != 1:
                raise ValueError(f"Input is 2D but channels={channels}. "
                                 f"Use --tile_single_channel if you want to tile it.")
            chw = x[None, ...]  # [1,H,W]
    else:
        raise ValueError(f"Unsupported tensor ndim={x.ndim}, shape={x.shape}")

    # ensure dtype float32
    if chw.dtype != np.float32:
        chw = chw.astype(np.float32, copy=False)
    return chw


def softmax_np(logits: np.ndarray, axis: int = -1) -> np.ndarray:
    m = logits.max(axis=axis, keepdims=True)
    y = np.exp(logits - m)
    return y / y.sum(axis=axis, keepdims=True)


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
        # Safe load
        arr = np.load(path, allow_pickle=False, mmap_mode=('r' if self.mmap else None))

        # Common case: uint8 images or features; allow optional normalization
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

    # Try to read the training-time metadata saved by your script
    genotype_map = checkpoint.get("genotype_map", {})
    in_channels = int(checkpoint.get("in_channels", 6))

    # Construct the model with provided (or default) depths/dims
    model = ConvNeXtCBAMClassifier(in_channels=in_channels, class_num=len(genotype_map) or 2,
                                   depths=depths, dims=dims).to(device)

    # Load weights (handles both wrapped and unwrapped state dicts)
    state = checkpoint.get("model_state_dict", checkpoint)
    try:
        model.load_state_dict(state, strict=True)
    except RuntimeError as e:
        # Helpful message if depths/dims mismatch
        raise RuntimeError(
            f"Failed to load state dict. If you trained with non-default depths/dims, "
            f"re-run with matching --depths and --dims. Original error:\n{e}"
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
    class_names: List[str]
) -> List[Dict[str, Any]]:
    results: List[Dict[str, Any]] = []
    use_amp = fp16 and device.type == "cuda"

    for batch in tqdm(dl, desc="Infer", leave=True):
        images, paths = batch
        images = images.to(device, non_blocking=True)
        if use_amp:
            images = images.half()

        with (torch.cuda.amp.autocast() if use_amp else torch.no_grad()):
            outputs = model(images)
            if isinstance(outputs, tuple):
                outputs = outputs[0]  # use main head

        # logits -> probs (cpu numpy)
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

    return results


def write_outputs(results: List[Dict[str, Any]], csv_path: str, json_path: str) -> None:
    if json_path:
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

    if csv_path:
        import csv
        # Collect all class names to produce consistent columns
        all_classes = set()
        for r in results:
            all_classes.update(r["probs"].keys())
        header = ["path", "pred_idx", "pred_class", "pred_prob"] + sorted(all_classes)

        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(header)
            for r in results:
                row = [r["path"], r["pred_idx"], r["pred_class"], f"{r['pred_prob']:.6f}"]
                # Append probs in the same sorted order
                for cname in sorted(all_classes):
                    row.append(f"{r['probs'].get(cname, 0.0):.6f}")
                w.writerow(row)


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
    args = parser.parse_args()

    # Resolve device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif args.device == "cuda":
        if not torch.cuda.is_available():
            print("CUDA not available, falling back to CPU.")
            device = torch.device("cpu")
        else:
            device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    # For faster inference on varying shapes
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass

    # Build model & load weights
    model, genotype_map, in_channels = build_and_load_model(
        ckpt_path=args.ckpt,
        device=device,
        depths=args.depths,
        dims=args.dims,
    )

    if len(genotype_map) == 0:
        # Fallback if checkpoint lacks class names; we try to infer count from the head
        # But your training script stores genotype_map, so this is unlikely.
        print("WARNING: 'genotype_map' missing in checkpoint; using numeric class names.")
        # Try to peek classifier out-features
        num_classes = None
        for n, m in model.named_modules():
            if isinstance(m, torch.nn.Linear) and (num_classes is None or m.out_features < num_classes):
                num_classes = m.out_features
        if num_classes is None:
            num_classes = 2
        class_names = [str(i) for i in range(num_classes)]
    else:
        # Map indices -> names in the correct order
        class_names = [None] * len(genotype_map)
        for cname, cidx in genotype_map.items():
            if 0 <= int(cidx) < len(class_names):
                class_names[cidx] = str(cname)
        # fill any None (just in case)
        for i in range(len(class_names)):
            if class_names[i] is None:
                class_names[i] = str(i)

    # Gather files
    files = list_npy_files(os.path.abspath(os.path.expanduser(args.input_dir)))
    if len(files) == 0:
        raise SystemExit(f"No .npy files found under: {args.input_dir}")

    # Build dataset/dataloader
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

    # Optionally convert model to half
    if args.fp16 and device.type == "cuda":
        model = model.half()

    # Run
    results = run_inference(
        model=model,
        dl=dl,
        device=device,
        fp16=args.fp16,
        class_names=class_names
    )

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
