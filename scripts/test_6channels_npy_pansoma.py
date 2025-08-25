#!/usr/bin/env python3
import argparse
import os
import sys
import json
import time
from collections import defaultdict

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel
from tqdm import tqdm
import torch.nn.functional as F

# Local imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from mynet import ConvNeXtCBAMClassifier
from dataset_pansoma_npy_6ch import get_inference_data_loader  # <-- unlabeled loader

# Globals updated in __main__ when using DDP
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IS_MAIN_PROCESS = True  # rank-0 only printing

def print_and_log(msg, log_path):
    if IS_MAIN_PROCESS:
        print(msg, flush=True)
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(msg + "\n")

def _fmt_int(n):
    try: return f"{int(n):,}"
    except Exception: return str(n)

@torch.no_grad()
def run_inference(model, data_loader, class_names, log_file, topk=1, save_probs=False, save_logits=False):
    """
    Returns list of dicts:
      {"path": <str>, "top1_idx": int, "top1_name": str, "top1_prob": float,
       "topk_idxs": [...], "topk_names": [...], "topk_probs": [...],
       "probs": [...], "logits": [...]}
    """
    model.eval()
    results = []
    total = 0
    start = time.time()

    pbar = tqdm(data_loader, desc="Infer", disable=not IS_MAIN_PROCESS)
    for batch in pbar:
        # loader returns (x, -1, path)
        if isinstance(batch, (list, tuple)) and len(batch) == 3:
            images, _, paths = batch
        else:
            images, _ = batch
            paths = [""] * images.size(0)

        images = images.to(device, non_blocking=True)
        outputs = model(images)
        if isinstance(outputs, tuple):
            outputs = outputs[0]

        probs = F.softmax(outputs, dim=1)
        k = min(topk, probs.shape[1])
        top_p, top_i = torch.topk(probs, k=k, dim=1)

        for i in range(images.size(0)):
            top_idxs = top_i[i].tolist()
            top_probs = top_p[i].tolist()
            rec = {
                "path": os.path.basename(paths[i]) if paths[i] else f"sample_{total+i:06d}",
                "top1_idx": int(top_idxs[0]),
                "top1_name": class_names[top_idxs[0]] if 0 <= top_idxs[0] < len(class_names) else str(top_idxs[0]),
                "top1_prob": float(top_probs[0]),
            }
            if k > 1:
                rec["topk_idxs"] = [int(x) for x in top_idxs]
                rec["topk_names"] = [class_names[j] if 0 <= j < len(class_names) else str(j) for j in top_idxs]
                rec["topk_probs"] = [float(x) for x in top_probs]
            if save_probs:
                rec["probs"] = [float(x) for x in probs[i].tolist()]
            if save_logits:
                rec["logits"] = [float(x) for x in outputs[i].tolist()]
            results.append(rec)

        total += images.size(0)
        if IS_MAIN_PROCESS:
            pbar.set_postfix(seen=_fmt_int(total))

    elapsed = time.time() - start
    if IS_MAIN_PROCESS and elapsed > 0:
        print_and_log(f"[Infer] Processed {_fmt_int(total)} samples in {elapsed:.2f}s "
                      f"({int(total/max(1e-9, elapsed))} samples/s).", log_file)
    return results

def _ensure_model(depths, dims, in_channels, class_num):
    return ConvNeXtCBAMClassifier(
        in_channels=in_channels,
        class_num=class_num,
        depths=depths, dims=dims
    )

def main():
    parser = argparse.ArgumentParser(description="Inference on 6-channel .npy folder (no labels)")
    parser.add_argument("data_path", type=str, help="Directory containing .npy files.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint .pth (model_best.pth).")
    parser.add_argument("-o", "--output_path", default="./inference_results_6channel", type=str,
                        help="Directory to save logs/predictions.")
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size")
    parser.add_argument("--num_workers", type=int, default=8, help="DataLoader workers")
    parser.add_argument("--depths", type=int, nargs="+", default=[3, 3, 27, 3], help="ConvNeXt stage depths")
    parser.add_argument("--dims", type=int, nargs="+", default=[192, 384, 768, 1536], help="ConvNeXt dims")

    # Multi-GPU (optional)
    parser.add_argument("--data_parallel", action="store_true", help="Use nn.DataParallel (single-process).")

    # Outputs
    parser.add_argument("--topk", type=int, default=1, help="Also save top-k predictions (k>=1).")
    parser.add_argument("--save_probs", action="store_true", help="Include full per-class probabilities.")
    parser.add_argument("--save_logits", action="store_true", help="Include raw logits.")
    args = parser.parse_args()

    # Safer start method for multiprocess loading
    try:
        import multiprocessing as mp
        mp.set_start_method("spawn", force=True)
    except Exception:
        pass

    data_dir = os.path.abspath(os.path.expanduser(args.data_path))
    os.makedirs(args.output_path, exist_ok=True)
    log_file = os.path.join(args.output_path, "inference_log_6ch.txt")
    if os.path.exists(log_file):
        os.remove(log_file)

    # Summary
    print_and_log("======== Inference Args ========", log_file)
    print_and_log(f"Output path : {args.output_path}", log_file)
    print_and_log(f"Checkpoint  : {args.checkpoint}", log_file)
    print_and_log(f"Batch size  : {args.batch_size}", log_file)
    print_and_log(f"Workers     : {args.num_workers}", log_file)
    print_and_log(f"Device      : {'cuda' if torch.cuda.is_available() else 'cpu'}", log_file)
    print_and_log(f"Data dir    : {data_dir}", log_file)
    print_and_log("================================", log_file)

    # Build loader (unlabeled)
    print_and_log("[Data] Building inference DataLoader...", log_file)
    infer_loader, genotype_map = get_inference_data_loader(
        data_dir, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False, return_paths=True
    )
    print_and_log("[Data] Inference DataLoader built.", log_file)
    try:
        ds_len = len(infer_loader.dataset)
        print_and_log(f"[Data] Files: {_fmt_int(ds_len)}", log_file)
    except Exception:
        pass

    # Load checkpoint
    if not os.path.isfile(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location=device)
    ckpt_classes = ckpt.get("genotype_map", None)
    if not ckpt_classes:
        raise RuntimeError("Checkpoint must contain 'genotype_map' to name classes.")
    num_classes = len(ckpt_classes)
    class_names = [None] * num_classes
    for name, idx in ckpt_classes.items():
        if 0 <= idx < num_classes:
            class_names[idx] = str(name)
    # Fill any gaps with the index to be safe
    for i in range(num_classes):
        if class_names[i] is None:
            class_names[i] = str(i)

    in_channels = int(ckpt.get("in_channels", 6))
    model = _ensure_model(args.depths, args.dims, in_channels=in_channels, class_num=num_classes).to(device)

    # Load weights
    missing, unexpected = model.load_state_dict(ckpt["model_state_dict"], strict=False)
    if missing: print_and_log(f"[Model] Missing keys (ok if only classifier head): {missing}", log_file)
    if unexpected: print_and_log(f"[Model] Unexpected keys: {unexpected}", log_file)

    # Optional DataParallel
    if args.data_parallel and torch.cuda.is_available() and torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
        print_and_log(f"[Runtime] Using DataParallel across {torch.cuda.device_count()} GPUs.", log_file)
    else:
        print_and_log(f"[Runtime] Using device: {device}", log_file)

    # Run inference
    print_and_log("[Infer] Starting...", log_file)
    preds = run_inference(
        model, infer_loader, class_names, log_file,
        topk=max(1, args.topk), save_probs=args.save_probs, save_logits=args.save_logits
    )

    # Save predictions
    import csv
    csv_path = os.path.join(args.output_path, "predictions.csv")
    json_path = os.path.join(args.output_path, "predictions.json")
    with open(csv_path, "w", newline="") as f:
        base_fields = ["path", "top1_idx", "top1_name", "top1_prob"]
        extra = []
        if args.topk > 1:
            extra += ["topk_idxs", "topk_names", "topk_probs"]
        if args.save_probs:
            extra += ["probs"]
        if args.save_logits:
            extra += ["logits"]
        writer = csv.DictWriter(f, fieldnames=base_fields + extra)
        writer.writeheader()
        for r in preds:
            row = {k: r[k] for k in base_fields}
            for k in extra:
                # store lists as JSON strings to keep CSV tidy
                row[k] = json.dumps(r.get(k, []))
            writer.writerow(row)
    with open(json_path, "w") as f:
        json.dump(preds, f, indent=2)

    print_and_log(f"[Save] Wrote {len(preds)} predictions to:", log_file)
    print_and_log(f"       {csv_path}", log_file)
    print_and_log(f"       {json_path}", log_file)
    print_and_log("[Infer] Finished.", log_file)

if __name__ == "__main__":
    main()
