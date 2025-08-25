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
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel
from tqdm import tqdm
import torch.nn.functional as F

# Local imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from mynet import ConvNeXtCBAMClassifier
from dataset_pansoma_npy_6ch import get_testing_data_loader  # <-- using testing loader

# Globals updated in __main__ when using DDP
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IS_MAIN_PROCESS = True  # rank-0 only printing


# ---------------- Utilities ----------------
def print_and_log(message, log_path):
    if not IS_MAIN_PROCESS:
        return
    print(message, flush=True)
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(message + "\n")


def _state_dict(m):
    return m.module.state_dict() if hasattr(m, "module") else m.state_dict()


def _load_state_dict(m, state):
    if hasattr(m, "module"):
        m.module.load_state_dict(state)
    else:
        m.load_state_dict(state)


def _read_paths_file(file_path):
    paths = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            paths.append(os.path.abspath(os.path.expanduser(s)))
    return paths


def _fmt_int(n):
    try:
        return f"{int(n):,}"
    except Exception:
        return str(n)


# ---------------- Evaluation ----------------
@torch.no_grad()
def evaluate(model, data_loader, genotype_map, log_file, ddp=False, world_size=1,
             save_predictions=False, predictions_limit=None):
    model.eval()
    num_classes = len(genotype_map) if genotype_map else 0
    idx_to_class = {v: k for k, v in genotype_map.items()} if genotype_map else {}

    correct = torch.zeros(1, device=device, dtype=torch.long)
    total   = torch.zeros(1, device=device, dtype=torch.long)

    class_correct = torch.zeros(num_classes, device=device, dtype=torch.long)
    class_total   = torch.zeros(num_classes, device=device, dtype=torch.long)

    confusion = torch.zeros((num_classes, num_classes), device=device, dtype=torch.long)

    tp = torch.zeros(num_classes, device=device, dtype=torch.long)
    fp = torch.zeros(num_classes, device=device, dtype=torch.long)
    fn = torch.zeros(num_classes, device=device, dtype=torch.long)

    local_predictions = []
    total_batches = len(data_loader) if hasattr(data_loader, "__len__") else None
    log_every = max(1, (total_batches or 1) // 20)  # ~5%

    start_time = time.time()
    pbar = tqdm(data_loader, desc="Testing", disable=not IS_MAIN_PROCESS)

    for b_idx, batch in enumerate(pbar):
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

        probs = F.softmax(outputs, dim=1)
        conf, pred = torch.max(probs, dim=1)

        correct += (pred == labels).sum()
        total += labels.size(0)

        for i in range(labels.size(0)):
            ti = int(labels[i]); pi = int(pred[i])
            class_total[ti] += 1
            if ti == pi:
                class_correct[ti] += 1
                tp[ti] += 1
            else:
                if pi < num_classes:
                    fp[pi] += 1
                fn[ti] += 1
            if ti < num_classes and pi < num_classes:
                confusion[ti, pi] += 1

            if save_predictions and (predictions_limit is None or len(local_predictions) < predictions_limit):
                local_predictions.append({
                    "path": os.path.basename(paths[i]) if paths[i] else "",
                    "true_idx": ti,
                    "true_name": idx_to_class.get(ti, str(ti)),
                    "pred_idx": pi,
                    "pred_name": idx_to_class.get(pi, str(pi)),
                    "pred_prob": float(conf[i].item())
                })

        if IS_MAIN_PROCESS and ((b_idx == 0) or ((b_idx + 1) % log_every == 0)):
            seen = total.item()
            acc_so_far = (correct.item() / max(1, seen)) * 100.0
            if total_batches:
                print_and_log(
                    f"[Eval] Batch {b_idx + 1}/{total_batches} | Seen { _fmt_int(seen) } | Acc-so-far {acc_so_far:.2f}%",
                    log_file
                )
            else:
                print_and_log(
                    f"[Eval] Batch {b_idx + 1} | Seen { _fmt_int(seen) } | Acc-so-far {acc_so_far:.2f}%",
                    log_file
                )

    if ddp and world_size > 1 and dist.is_initialized():
        dist.all_reduce(correct, op=dist.ReduceOp.SUM)
        dist.all_reduce(total,   op=dist.ReduceOp.SUM)
        if num_classes > 0:
            dist.all_reduce(class_correct, op=dist.ReduceOp.SUM)
            dist.all_reduce(class_total,   op=dist.ReduceOp.SUM)
            dist.all_reduce(tp, op=dist.ReduceOp.SUM)
            dist.all_reduce(fp, op=dist.ReduceOp.SUM)
            dist.all_reduce(fn, op=dist.ReduceOp.SUM)
            dist.all_reduce(confusion, op=dist.ReduceOp.SUM)

        if save_predictions:
            gathered = [None for _ in range(world_size)]
            dist.all_gather_object(gathered, local_predictions)
            if IS_MAIN_PROCESS:
                merged = []
                for chunk in gathered:
                    if chunk:
                        merged.extend(chunk)
                local_predictions = merged
            else:
                local_predictions = None

    elapsed = time.time() - start_time
    if IS_MAIN_PROCESS and elapsed > 0:
        print_and_log(f"[Eval] Done in {elapsed:.2f}s | ~{_fmt_int(int(total.item() / max(1e-9, elapsed)))} samples/s",
                      log_file)

    confusion_cpu = confusion.detach().cpu()
    class_stats = {}
    for cname, cidx in (genotype_map.items() if genotype_map else []):
        c_correct = int(class_correct[cidx].item())
        c_total   = int(class_total[cidx].item())
        acc = (c_correct / c_total * 100.0) if c_total > 0 else 0.0
        tp_i = int(confusion_cpu[cidx, cidx].item())
        fp_i = int(confusion_cpu[:, cidx].sum().item()) - tp_i
        fn_i = int(confusion_cpu[cidx, :].sum().item()) - tp_i
        prec = (tp_i / (tp_i + fp_i)) if (tp_i + fp_i) > 0 else 0.0
        rec  = (tp_i / (tp_i + fn_i)) if (tp_i + fn_i) > 0 else 0.0
        f1   = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
        class_stats[cname] = {"idx": cidx, "acc": acc, "precision": prec, "recall": rec, "f1": f1, "support": c_total}

    # pick positive class
    pos_idx = None
    if genotype_map:
        for name, idx in genotype_map.items():
            if str(name).lower() == "true":
                pos_idx = idx; break
    if pos_idx is None:
        if len(genotype_map) > 1:
            pos_idx = 1
        elif len(genotype_map) > 0:
            supports = class_total.clone()
            if supports.sum() > 0:
                pos_idx = int(torch.nonzero(supports == supports[supports > 0].min(), as_tuple=False)[0].item())
            else:
                pos_idx = 0
        else:
            pos_idx = 0

    tpc = float(tp[pos_idx].item() if pos_idx < len(genotype_map) else 0.0)
    fpc = float(fp[pos_idx].item() if pos_idx < len(genotype_map) else 0.0)
    fnc = float(fn[pos_idx].item() if pos_idx < len(genotype_map) else 0.0)
    precision_true = (tpc / (tpc + fpc)) if (tpc + fpc) > 0 else 0.0
    recall_true    = (tpc / (tpc + fnc)) if (tpc + fnc) > 0 else 0.0
    f1_true        = (2 * precision_true * recall_true / (precision_true + recall_true)) if (precision_true + recall_true) > 0 else 0.0

    metrics = {
        "overall_accuracy": (correct.item() / max(1, total.item())) * 100.0,
        "precision_true": precision_true,
        "recall_true": recall_true,
        "f1_true": f1_true,
        "pos_class_idx": pos_idx
    }
    return metrics, class_stats, confusion_cpu, (local_predictions if save_predictions else None)


def _build_loader_for_test(data_spec, batch_size, num_workers, ddp=False):
    """
    Build a test DataLoader using get_testing_data_loader (no subfolders).
    Re-wrap the dataset to force persistent_workers=False and small prefetch.
    """
    # Initial loader from your utility (may set persistent_workers=True internally)
    loader, genotype_map = get_testing_data_loader(
        data_spec, batch_size=batch_size, num_workers=num_workers,
        shuffle=False, return_paths=True
    )
    ds = loader.dataset

    # Rebuild DataLoader ourselves to avoid hangs with persistent_workers
    common_kwargs = dict(
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=False,
    )
    if ddp:
        sampler = DistributedSampler(ds, shuffle=False, drop_last=False)
        common_kwargs.update(dict(sampler=sampler))
    else:
        common_kwargs.update(dict(shuffle=False))

    # prefetch_factor only valid when num_workers > 0
    if num_workers and num_workers > 0:
        common_kwargs["prefetch_factor"] = 2

    loader = DataLoader(ds, **common_kwargs)
    return loader, genotype_map


def _ensure_model(depths, dims, in_channels):
    model = ConvNeXtCBAMClassifier(
        in_channels=in_channels,
        class_num=1,  # temp; fixed after we know num_classes
        depths=depths, dims=dims
    )
    return model


# ---------------- Main ----------------
def main():
    parser = argparse.ArgumentParser(description="Test/Evaluate a 6-channel classifier on .npy dataset")
    parser.add_argument("data_path", nargs="?", type=str, help="Directory containing .npy files (Mode A).")
    parser.add_argument("--test_data_paths_file", type=str, default=None,
                        help="Text file: one or more directories with .npy files (Mode B).")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint .pth (e.g., model_best.pth).")

    parser.add_argument("-o", "--output_path", default="./test_results_6channel", type=str,
                        help="Directory to save logs/metrics (created if missing).")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--num_workers", type=int, default=8, help="DataLoader workers")

    parser.add_argument("--depths", type=int, nargs="+", default=[3, 3, 27, 3], help="ConvNeXt stage depths")
    parser.add_argument("--dims", type=int, nargs="+", default=[192, 384, 768, 1536], help="ConvNeXt dims")

    parser.add_argument("--ddp", action="store_true", help="Use DistributedDataParallel (torchrun).")
    parser.add_argument("--data_parallel", action="store_true", help="Use nn.DataParallel (ignored if --ddp).")
    parser.add_argument("--save_predictions", action="store_true", help="Save per-sample predictions CSV/JSON.")
    parser.add_argument("--predictions_limit", type=int, default=None, help="Cap saved predictions.")

    args = parser.parse_args()

    # Safer start method with CUDA + workers
    try:
        import multiprocessing as mp
        mp.set_start_method("spawn", force=True)
    except Exception:
        pass

    # Build data spec (no subfolders)
    has_base = args.data_path is not None
    has_file = args.test_data_paths_file is not None
    if has_base and has_file:
        parser.error("Provide either positional data_path (Mode A) OR --test_data_paths_file (Mode B), not both.")
    if not has_base and not has_file:
        parser.error("You must provide exactly one input mode: data_path OR --test_data_paths_file")
    if has_base:
        data_spec = os.path.abspath(os.path.expanduser(args.data_path))
        roots_for_log = [data_spec]
    else:
        roots_for_log = _read_paths_file(args.test_data_paths_file)
        if not roots_for_log:
            parser.error(f"--test_data_paths_file is empty or unreadable: {args.test_data_paths_file}")
        data_spec = roots_for_log

    os.makedirs(args.output_path, exist_ok=True)
    log_file = os.path.join(args.output_path, "test_log_6ch.txt")
    if os.path.exists(log_file):
        os.remove(log_file)

    # Args summary
    print_and_log("======== Inference Args ========", log_file)
    print_and_log(f"Output path: {args.output_path}", log_file)
    print_and_log(f"Checkpoint : {args.checkpoint}", log_file)
    print_and_log(f"Batch size : {args.batch_size}", log_file)
    print_and_log(f"Workers    : {args.num_workers}", log_file)
    print_and_log(f"DDP        : {args.ddp}", log_file)
    print_and_log(f"DataParallel: {args.data_parallel}", log_file)
    if len(roots_for_log) == 1:
        print_and_log(f"Roots      : {roots_for_log[0]}", log_file)
    else:
        show = ", ".join(roots_for_log[:3]) + (" ..." if len(roots_for_log) > 3 else "")
        print_and_log(f"Roots (n={len(roots_for_log)}): {show}", log_file)
    print_and_log("================================", log_file)

    # DDP init (takes precedence)
    if args.ddp:
        if not torch.cuda.is_available():
            raise RuntimeError("DDP requires CUDA.")
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        torch.cuda.set_device(local_rank)
        global device, IS_MAIN_PROCESS
        device = torch.device(f"cuda:{local_rank}")
        dist.init_process_group(backend="nccl", init_method="env://")
        IS_MAIN_PROCESS = (dist.get_rank() == 0)
        if IS_MAIN_PROCESS:
            print_and_log(f"[DDP] World size={dist.get_world_size()} | Local rank={local_rank} | Global rank={dist.get_rank()}",
                          log_file)
    else:
        local_rank = 0
        IS_MAIN_PROCESS = True
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print_and_log(f"[Runtime] Device={device} | CUDA available={torch.cuda.is_available()}", log_file)

    # Build loader (test)
    print_and_log("[Data] Building test DataLoader...", log_file)
    test_loader, genotype_map = _build_loader_for_test(
        data_spec, batch_size=args.batch_size, num_workers=args.num_workers, ddp=args.ddp
    )
    print_and_log("[Data] Test DataLoader built.", log_file)

    # (Optional) smoke test to surface worker issues early (non-DDP only)
    if IS_MAIN_PROCESS and not args.ddp:
        try:
            print_and_log("[Data] Running one-batch smoke test...", log_file)
            first_batch = next(iter(test_loader))
            if isinstance(first_batch, (list, tuple)) and len(first_batch) >= 2:
                bx = first_batch[0].shape[0]
                print_and_log(f"[Data] Smoke test OK. First batch size: {bx}", log_file)
            else:
                print_and_log("[Data] Smoke test OK.", log_file)
        except Exception as e:
            print_and_log(f"[Data] Smoke test FAILED: {e}", log_file)
            print_and_log("[Hint] Retry with --num_workers 0 to disable multiprocessing.", log_file)
            return

    try:
        ds_len = len(test_loader.dataset)
    except Exception:
        ds_len = None
    if IS_MAIN_PROCESS:
        if ds_len is not None:
            steps = len(test_loader) if hasattr(test_loader, "__len__") else "?"
            print_and_log(f"[Data] Test samples: {_fmt_int(ds_len)} | Batches per rank: {steps}", log_file)
        else:
            print_and_log(f"[Data] Built test DataLoader.", log_file)

    if not genotype_map:
        print_and_log("FATAL: genotype_map is empty from dataloader.", log_file)
        return
    num_classes = len(genotype_map)
    sorted_class_names = sorted(genotype_map.keys(), key=lambda k: genotype_map[k])
    if IS_MAIN_PROCESS:
        print_and_log(f"[Data] num_classes={num_classes}", log_file)
        for cname in sorted_class_names:
            print_and_log(f"  - class '{cname}' -> idx {genotype_map[cname]}", log_file)

    # Prepare model
    in_channels = 6
    model = _ensure_model(args.depths, args.dims, in_channels=in_channels).to(device)

    if (not args.ddp) and args.data_parallel and torch.cuda.is_available():
        n = torch.cuda.device_count()
        if n > 1:
            print_and_log(f"[Runtime] Wrapping model in DataParallel across {n} GPUs.", log_file)
            model = nn.DataParallel(model)
        else:
            print_and_log("[Runtime] DataParallel requested but single CUDA device found; running on one GPU.", log_file)

    # Load checkpoint
    if not os.path.isfile(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location=device)
    if IS_MAIN_PROCESS:
        ep = ckpt.get("epoch", None); best_f1 = ckpt.get("best_f1_true", None)
        print_and_log(f"[Checkpoint] Loaded: {args.checkpoint}", log_file)
        if ep is not None:
            print_and_log(f"[Checkpoint] epoch={ep}", log_file)
        if best_f1 is not None:
            print_and_log(f"[Checkpoint] best_f1_true={best_f1:.4f}", log_file)

    if "in_channels" in ckpt:
        in_ckpt = int(ckpt["in_channels"])
        if in_ckpt != in_channels:
            model = _ensure_model(args.depths, args.dims, in_channels=in_ckpt).to(device)
            if IS_MAIN_PROCESS:
                print_and_log(f"[Model] Rebuilt with in_channels={in_ckpt} (from checkpoint).", log_file)

    class_num_target = num_classes
    fresh_model = ConvNeXtCBAMClassifier(
        in_channels=getattr(model, "in_channels", 6) if hasattr(model, "in_channels") else 6,
        class_num=class_num_target,
        depths=args.depths,
        dims=args.dims
    ).to(device)

    missing, unexpected = fresh_model.load_state_dict(ckpt["model_state_dict"], strict=False)
    if IS_MAIN_PROCESS:
        if missing:
            print_and_log(f"[Model] Missing keys (ok if only classifier head): {missing}", log_file)
        if unexpected:
            print_and_log(f"[Model] Unexpected keys: {unexpected}", log_file)
    model = fresh_model

    if args.ddp:
        model = DistributedDataParallel(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            gradient_as_bucket_view=True,
            broadcast_buffers=False,
        )
        if IS_MAIN_PROCESS:
            print_and_log("[Runtime] Model wrapped with DistributedDataParallel.", log_file)

    if IS_MAIN_PROCESS:
        print_and_log("[Eval] Starting evaluation...", log_file)
    world_size = dist.get_world_size() if args.ddp and dist.is_initialized() else 1
    metrics, class_stats, confusion, predictions = evaluate(
        model, test_loader, genotype_map, log_file,
        ddp=args.ddp, world_size=world_size,
        save_predictions=args.save_predictions,
        predictions_limit=args.predictions_limit
    )

    if IS_MAIN_PROCESS:
        print_and_log("\nOverall Test Results:", log_file)
        print_and_log(f"  Accuracy: {metrics['overall_accuracy']:.2f}%", log_file)
        print_and_log(f"  True-class Precision: {metrics['precision_true']*100:.2f}%", log_file)
        print_and_log(f"  True-class Recall:    {metrics['recall_true']*100:.2f}%", log_file)
        print_and_log(f"  True-class F1:        {metrics['f1_true']*100:.2f}%", log_file)
        if metrics.get("pos_class_idx") is not None:
            pos_idx = metrics["pos_class_idx"]
            cname = next((n for n, i in genotype_map.items() if i == pos_idx), str(pos_idx))
            print_and_log(f"  Positive class index: {pos_idx} (\"{cname}\")", log_file)

        print_and_log("\nClass-wise Metrics:", log_file)
        for cname in sorted_class_names:
            s = class_stats[cname]
            print_and_log(
                f"  {cname} (idx {s['idx']}): Acc {s['acc']:.2f}% | "
                f"Prec {s['precision']*100:.2f}% | Rec {s['recall']*100:.2f}% | "
                f"F1 {s['f1']*100:.2f}% | n={_fmt_int(s['support'])}",
                log_file
            )

        out_metrics = {"overall": metrics, "class_stats": class_stats, "genotype_map": genotype_map}
        with open(os.path.join(args.output_path, "metrics.json"), "w") as f:
            json.dump(out_metrics, f, indent=2)
        print_and_log(f"[Save] Wrote metrics.json", log_file)

        import csv
        cm_path = os.path.join(args.output_path, "confusion_matrix.csv")
        with open(cm_path, "w", newline="") as f:
            writer = csv.writer(f)
            header = ["true\\pred"] + [str(c) for c in sorted_class_names]
            writer.writerow(header)
            for cname in sorted_class_names:
                row = [cname]
                ti = genotype_map[cname]
                for pj_name in sorted_class_names:
                    pj = genotype_map[pj_name]
                    row.append(int(confusion[ti, pj]))
                writer.writerow(row)
        print_and_log(f"[Save] Wrote confusion_matrix.csv", log_file)

        if args.save_predictions and predictions is not None:
            pred_csv = os.path.join(args.output_path, "predictions.csv")
            with open(pred_csv, "w", newline="") as f:
                import csv
                writer = csv.DictWriter(f, fieldnames=["path", "true_idx", "true_name", "pred_idx", "pred_name", "pred_prob"])
                writer.writeheader()
                for r in predictions:
                    writer.writerow(r)
            pred_json = os.path.join(args.output_path, "predictions.json")
            with open(pred_json, "w") as f:
                json.dump(predictions, f, indent=2)
            print_and_log(f"[Save] Wrote predictions ({_fmt_int(len(predictions))} rows) to predictions.csv/json", log_file)

        print_and_log("[Eval] Finished. See test_log_6ch.txt for details.", log_file)

    if args.ddp and dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
