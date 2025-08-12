#!/usr/bin/env python3
import argparse
import json
import os
import sys
from collections import defaultdict

import torch
import torch.nn.functional as F
from tqdm import tqdm

# Make repo imports work when called as a script
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from mynet import ConvNeXtCBAMClassifier
from dataset_pansoma_npy_6ch import get_data_loader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_model(in_channels, num_classes, depths, dims, ckpt_state_dict):
    """
    Build the model and load weights.
    depths/dims should match training; pass the same values you used then.
    """
    model = ConvNeXtCBAMClassifier(
        in_channels=in_channels,
        class_num=num_classes,
        depths=depths,
        dims=dims
    ).to(device)

    # Load weights
    missing, unexpected = model.load_state_dict(ckpt_state_dict, strict=False)
    if missing:
        print(f"[WARN] Missing keys when loading state_dict ({len(missing)}): {sorted(missing)[:10]}{' ...' if len(missing)>10 else ''}")
    if unexpected:
        print(f"[WARN] Unexpected keys when loading state_dict ({len(unexpected)}): {sorted(unexpected)[:10]}{' ...' if len(unexpected)>10 else ''}")
    return model


def softmax_probs(logits):
    if isinstance(logits, tuple):
        logits = logits[0]  # use main head
    return F.softmax(logits, dim=1)


def try_make_loader(data_dir, batch_size, num_workers):
    """
    Prefer 'test' split with return_paths=True.
    Fall back to 'val' with return_paths=True.
    As a last resort, try without return_paths (no file paths in outputs).
    """
    # 1) test with paths
    try:
        dl, genotype_map = get_data_loader(
            data_dir=data_dir, dataset_type="test",
            batch_size=batch_size, num_workers=num_workers,
            shuffle=False, return_paths=True
        )
        return dl, genotype_map, True
    except Exception as e:
        print(f"[INFO] 'test' with return_paths=True unavailable: {e}")

    # 2) val with paths
    try:
        dl, genotype_map = get_data_loader(
            data_dir=data_dir, dataset_type="val",
            batch_size=batch_size, num_workers=num_workers,
            shuffle=False, return_paths=True
        )
        return dl, genotype_map, True
    except Exception as e:
        print(f"[INFO] 'val' with return_paths=True unavailable: {e}")

    # 3) test without paths
    try:
        dl, genotype_map = get_data_loader(
            data_dir=data_dir, dataset_type="test",
            batch_size=batch_size, num_workers=num_workers,
            shuffle=False
        )
        return dl, genotype_map, False
    except Exception as e:
        print(f"[INFO] 'test' without return_paths unavailable: {e}")

    # 4) val without paths
    dl, genotype_map = get_data_loader(
        data_dir=data_dir, dataset_type="val",
        batch_size=batch_size, num_workers=num_workers,
        shuffle=False
    )
    return dl, genotype_map, False


def run_inference(
    data_path,
    ckpt_path,
    output_path,
    depths,
    dims,
    batch_size=64,
    num_workers=8,
    topk=1
):
    os.makedirs(output_path, exist_ok=True)

    # --- Load checkpoint ---
    print(f"[INFO] Loading checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state = ckpt.get("model_state_dict", ckpt)  # allow raw state_dict as well
    ckpt_genotype_map = ckpt.get("genotype_map", None)
    ckpt_in_channels = ckpt.get("in_channels", 6)

    # --- Build dataloader (to know num_classes and maybe get paths) ---
    loader, dl_genotype_map, have_paths = try_make_loader(
        data_dir=data_path, batch_size=batch_size, num_workers=num_workers
    )

    if not dl_genotype_map and not ckpt_genotype_map:
        raise RuntimeError("Cannot determine genotype_map from dataloader or checkpoint.")

    # Prefer genotype_map from checkpoint (this is what the model was trained on)
    genotype_map = ckpt_genotype_map or dl_genotype_map
    num_classes = len(genotype_map)
    if num_classes <= 0:
        raise RuntimeError("Invalid number of classes (<=0). Check your dataset or checkpoint.")

    idx_to_class = {v: k for k, v in genotype_map.items()}

    # --- Build model, load weights ---
    model = build_model(
        in_channels=ckpt_in_channels,
        num_classes=num_classes,
        depths=depths,
        dims=dims,
        ckpt_state_dict=state
    )
    model.eval()

    # --- Prepare outputs ---
    pred_jsonl = os.path.join(output_path, "predictions.jsonl")
    pred_csv = os.path.join(output_path, "predictions.csv")
    metrics_json = os.path.join(output_path, "metrics.json")

    jsonl_f = open(pred_jsonl, "w", encoding="utf-8")
    csv_f = open(pred_csv, "w", encoding="utf-8")
    # CSV header: path(optional), pred_idx, pred_class, [topk probs/classes], true(optional)
    if have_paths:
        csv_f.write("path,pred_idx,pred_class,prob")
    else:
        csv_f.write("pred_idx,pred_class,prob")
    # add columns for top-k>1
    if topk and topk > 1:
        for i in range(topk):
            csv_f.write(f",top{i+1}_idx,top{i+1}_class,top{i+1}_prob")
    # label (if present)
    csv_f.write(",true_idx,true_class\n")

    # --- Inference loop ---
    total = 0
    correct = 0
    class_correct = defaultdict(int)
    class_total = defaultdict(int)
    labels_available = None  # determine on first batch

    with torch.no_grad():
        for batch in tqdm(loader, desc="Inferring", leave=True):
            # Batch shape can be:
            #   (images, labels, paths)  or
            #   (images, labels)         or
            #   (images, paths)
            images = batch[0].to(device)

            # detect batch contents
            b_labels = None
            b_paths = None
            if len(batch) == 3:
                # assume (images, labels, paths)
                if isinstance(batch[1], torch.Tensor):
                    b_labels = batch[1].to(device)
                    b_paths = batch[2]
                else:
                    # (images, paths, ???) unlikely, but handle
                    # try to find tensor as labels and list as paths
                    for item in batch[1:]:
                        if isinstance(item, torch.Tensor):
                            b_labels = item.to(device)
                        elif isinstance(item, (list, tuple)):
                            b_paths = item
            elif len(batch) == 2:
                # could be (images, labels) OR (images, paths)
                if isinstance(batch[1], torch.Tensor):
                    b_labels = batch[1].to(device)
                else:
                    b_paths = batch[1]
            # decide labels availability
            if labels_available is None:
                labels_available = b_labels is not None

            logits = model(images)
            probs = softmax_probs(logits)  # (B, C)

            # top-1 prediction
            confs, pred_idx = torch.max(probs, dim=1)

            # metrics if labels exist
            if labels_available:
                correct += (pred_idx == b_labels).sum().item()
                total += b_labels.size(0)
                for i in range(b_labels.size(0)):
                    class_total[b_labels[i].item()] += 1
                    if pred_idx[i].item() == b_labels[i].item():
                        class_correct[b_labels[i].item()] += 1

            # write outputs
            if topk and topk > 1:
                topk_probs, topk_idx = torch.topk(probs, k=min(topk, probs.shape[1]), dim=1)

            B = probs.size(0)
            for i in range(B):
                pred_i = int(pred_idx[i].item())
                prob_i = float(confs[i].item())
                pred_name = idx_to_class.get(pred_i, str(pred_i))

                rec = {
                    "pred_idx": pred_i,
                    "pred_class": pred_name,
                    "prob": prob_i
                }
                if topk and topk > 1:
                    rec["topk"] = [
                        {
                            "idx": int(topk_idx[i, j].item()),
                            "class": idx_to_class.get(int(topk_idx[i, j].item()), str(int(topk_idx[i, j].item()))),
                            "prob": float(topk_probs[i, j].item())
                        }
                        for j in range(topk_probs.size(1))
                    ]
                if have_paths and b_paths is not None:
                    rec["path"] = str(b_paths[i])
                if labels_available:
                    true_idx = int(b_labels[i].item())
                    rec["true_idx"] = true_idx
                    rec["true_class"] = idx_to_class.get(true_idx, str(true_idx))

                jsonl_f.write(json.dumps(rec) + "\n")

                # CSV row
                csv_parts = []
                if have_paths and b_paths is not None:
                    csv_parts.append(str(b_paths[i]))
                csv_parts.extend([str(pred_i), pred_name, f"{prob_i:.6f}"])
                if topk and topk > 1:
                    for j in range(topk_probs.size(1)):
                        t_idx = int(topk_idx[i, j].item())
                        t_name = idx_to_class.get(t_idx, str(t_idx))
                        t_prob = float(topk_probs[i, j].item())
                        csv_parts.extend([str(t_idx), t_name, f"{t_prob:.6f}"])
                if labels_available:
                    t_idx = int(b_labels[i].item())
                    t_name = idx_to_class.get(t_idx, str(t_idx))
                    csv_parts.extend([str(t_idx), t_name])
                else:
                    csv_parts.extend(["", ""])  # true_idx,true_class empty
                csv_f.write(",".join([str(x) for x in csv_parts]) + "\n")

    jsonl_f.close()
    csv_f.close()
    print(f"[INFO] Wrote predictions to:\n  - {pred_jsonl}\n  - {pred_csv}")

    # Write metrics if labels available
    if labels_available and total > 0:
        overall_acc = correct / total
        class_stats = {}
        for cls_idx, tot in class_total.items():
            corr = class_correct.get(cls_idx, 0)
            acc = corr / tot if tot > 0 else 0.0
            class_stats[idx_to_class.get(cls_idx, str(cls_idx))] = {
                "idx": int(cls_idx),
                "correct": int(corr),
                "total": int(tot),
                "acc": acc
            }
        metrics = {
            "overall": {
                "correct": int(correct),
                "total": int(total),
                "accuracy": overall_acc
            },
            "per_class": class_stats
        }
        with open(metrics_json, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)
        print(f"[INFO] Wrote metrics to: {metrics_json}")
    else:
        print("[INFO] No labels detected in loader; metrics were not computed.")


def parse_args():
    p = argparse.ArgumentParser(description="Inference script for ConvNeXt-CBAM classifier on 6-channel .npy dataset")
    p.add_argument("data_path", type=str, help="Path to the dataset root (expects a 'test' or 'val' split).")
    p.add_argument("ckpt_path", type=str, help="Path to saved model checkpoint (.pth).")
    p.add_argument("-o", "--output_path", type=str, default="./inference_outputs", help="Directory to save predictions.")
    p.add_argument("--batch_size", type=int, default=64, help="Batch size for inference.")
    p.add_argument("--num_workers", type=int, default=8, help="Number of dataloader workers.")
    p.add_argument("--topk", type=int, default=1, help="Also report top-k predictions (k>=1).")

    # Must match training architecture
    p.add_argument("--depths", type=int, nargs="+", default=[3, 3, 27, 3],
                   help="ConvNeXt stage depths; MUST match the training model.")
    p.add_argument("--dims", type=int, nargs="+", default=[192, 384, 768, 1536],
                   help="ConvNeXt stage dims; MUST match the training model.")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_inference(
        data_path=args.data_path,
        ckpt_path=args.ckpt_path,
        output_path=args.output_path,
        depths=args.depths,
        dims=args.dims,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        topk=args.topk
    )
