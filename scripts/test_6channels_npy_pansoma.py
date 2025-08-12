#!/usr/bin/env python3
import argparse
import json
import os
import re
import sys
from collections import defaultdict

import torch
import torch.nn.functional as F
from tqdm import tqdm

# repo-local imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from mynet import ConvNeXtCBAMClassifier
from dataset_pansoma_npy_6ch import get_data_loader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

FNAME_PAT = re.compile(
    r"^(?P<node>\d+?)_(?P<start>\d+?)_(?P<etype>[XID])_(?P<ref>[ACGTN\-]+)_(?P<alt>[ACGTN\-]+)\.npy$",
    re.IGNORECASE
)

def parse_graph_fname(basename: str):
    """
    Parse file names like:
      1623868_162_X_A_C.npy
      nodeID_start_event_ref_alt.npy
    event: X=substitution, I=insertion, D=deletion
    Returns (node_id:str, start:int, ref:str, alt:str, etype:str)
    """
    m = FNAME_PAT.match(basename)
    if not m:
        raise ValueError(f"Filename not in expected pattern: {basename}")
    node = m.group("node")
    start = int(m.group("start"))
    etype = m.group("etype").upper()
    ref = m.group("ref").upper()
    alt = m.group("alt").upper()
    return node, start, ref, alt, etype

def softmax_probs(logits):
    if isinstance(logits, tuple):
        logits = logits[0]
    return F.softmax(logits, dim=1)

def build_model(in_channels, num_classes, depths, dims, state_dict):
    model = ConvNeXtCBAMClassifier(
        in_channels=in_channels,
        class_num=num_classes,
        depths=depths,
        dims=dims
    ).to(device)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"[WARN] Missing keys: {len(missing)} (showing up to 10): {missing[:10]}")
    if unexpected:
        print(f"[WARN] Unexpected keys: {len(unexpected)} (showing up to 10): {unexpected[:10]}")
    model.eval()
    return model

def try_make_loader(data_dir, batch_size, num_workers):
    # Try test/val with return_paths=True first
    for split in ("test", "val"):
        try:
            dl, genotype_map = get_data_loader(
                data_dir=data_dir, dataset_type=split,
                batch_size=batch_size, num_workers=num_workers,
                shuffle=False, return_paths=True
            )
            return dl, genotype_map, True
        except Exception as e:
            print(f"[INFO] {split} with return_paths=True unavailable: {e}")
    # Fallbacks without paths
    for split in ("test", "val"):
        try:
            dl, genotype_map = get_data_loader(
                data_dir=data_dir, dataset_type=split,
                batch_size=batch_size, num_workers=num_workers,
                shuffle=False
            )
            return dl, genotype_map, False
        except Exception as e:
            print(f"[INFO] {split} without return_paths unavailable: {e}")
    raise RuntimeError("Could not create a data loader from test/val.")

def infer_positive_index(genotype_map, positive_label_cli):
    """
    Determine which class index counts as 'variant'.
    Priority:
      1) exact key match to --positive_label
      2) heuristic: a key containing 'var' or 'alt' (case-insensitive)
      3) if 2 classes: choose the higher index
      4) otherwise raise
    """
    if positive_label_cli is not None:
        if positive_label_cli in genotype_map:
            return genotype_map[positive_label_cli]
        # try numeric-as-string
        for k, v in genotype_map.items():
            if str(k) == str(positive_label_cli):
                return v
        raise ValueError(f"--positive_label '{positive_label_cli}' not found in genotype_map keys: {list(genotype_map.keys())}")

    # heuristic
    for k, v in genotype_map.items():
        if isinstance(k, str) and (("var" in k.lower()) or ("alt" in k.lower()) or ("pos" in k.lower())):
            return v

    if len(genotype_map) == 2:
        # likely 0=ref, 1=variant; pick max index
        return max(genotype_map.values())

    raise ValueError("Cannot infer positive/variant class. Specify --positive_label.")

def open_vcf(out_path, model_name):
    f = open(out_path, "w", encoding="utf-8")
    f.write("##fileformat=VCFv4.2\n")
    f.write(f"##source={model_name}\n")
    f.write("##INFO=<ID=PROB,Number=1,Type=Float,Description=\"Model probability for the emitted class\">\n")
    f.write("##INFO=<ID=PRED,Number=1,Type=String,Description=\"Predicted class label\">\n")
    f.write("##INFO=<ID=ETYPE,Number=1,Type=String,Description=\"Event type from file name: X=substitution, I=insertion, D=deletion\">\n")
    f.write("##INFO=<ID=FILE,Number=1,Type=String,Description=\"Source window file basename\">\n")
    f.write("#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\n")
    return f

def write_record(vcf_f, node, pos, ref, alt, qual, filt, info_dict):
    info = ";".join([f"{k}={info_dict[k]}" for k in info_dict])
    row = [str(node), str(pos), ".", ref, alt, f"{qual:.4f}", filt, info]
    vcf_f.write("\t".join(row) + "\n")

def run(
    data_path,
    ckpt_path,
    output_vcf,
    depths,
    dims,
    batch_size=64,
    num_workers=8,
    prob_threshold=0.5,
    positive_label=None,
    emit_all=False
):
    os.makedirs(os.path.dirname(os.path.abspath(output_vcf)) or ".", exist_ok=True)

    # Load checkpoint
    print(f"[INFO] Loading checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state = ckpt.get("model_state_dict", ckpt)
    genotype_map_ckpt = ckpt.get("genotype_map")
    in_channels = ckpt.get("in_channels", 6)
    model_name = os.path.basename(ckpt_path)

    # Build loader
    loader, genotype_map_dl, have_paths = try_make_loader(data_path, batch_size, num_workers)
    genotype_map = genotype_map_ckpt or genotype_map_dl
    if not genotype_map:
        raise RuntimeError("genotype_map not available (neither in checkpoint nor dataloader).")

    idx_to_class = {v: k for k, v in genotype_map.items()}
    pos_idx = infer_positive_index(genotype_map, positive_label)

    # Build model
    model = build_model(in_channels, num_classes=len(genotype_map), depths=depths, dims=dims, state_dict=state)

    # Open VCF
    vcf_f = open_vcf(output_vcf, model_name)
    skipped_no_paths = 0
    emitted = 0

    with torch.no_grad():
        for batch in tqdm(loader, desc="Inferring & writing VCF", leave=True):
            images = batch[0].to(device)

            # get paths if present
            paths = None
            if len(batch) >= 2 and not isinstance(batch[1], torch.Tensor):
                paths = batch[1]
            if len(batch) >= 3 and not isinstance(batch[2], torch.Tensor):
                # safeguard if order differs
                paths = batch[2]

            logits = model(images)
            probs = softmax_probs(logits)  # (B, C)
            confs, pred_idx = torch.max(probs, dim=1)

            B = probs.size(0)
            for i in range(B):
                # Need a file path to parse node/pos/ref/alt
                if not have_paths or paths is None:
                    skipped_no_paths += 1
                    continue

                bname = os.path.basename(str(paths[i]))
                try:
                    node, start, ref, alt, etype = parse_graph_fname(bname)
                except Exception as e:
                    print(f"[WARN] Skipping file with unparsable name '{bname}': {e}")
                    continue

                p_idx = int(pred_idx[i].item())
                p_name = str(idx_to_class.get(p_idx, p_idx))
                prob = float(confs[i].item())

                # Decide whether to emit
                is_variant_call = (p_idx == pos_idx) and (prob >= prob_threshold)
                if not emit_all and not is_variant_call:
                    continue

                filt = "PASS" if prob >= prob_threshold else "LowQual"
                info = {
                    "PROB": f"{prob:.6f}",
                    "PRED": p_name,
                    "ETYPE": etype,
                    "FILE": bname
                }
                # VCF record: CHROM=nodeID, POS=starting offset
                write_record(vcf_f, node=node, pos=start, ref=ref, alt=alt, qual=prob, filt=filt, info_dict=info)
                emitted += 1

    vcf_f.close()
    print(f"[INFO] Wrote VCF: {output_vcf}")
    if skipped_no_paths:
        print(f"[INFO] Skipped {skipped_no_paths} items without paths (cannot parse node/pos/ref/alt).")
    print(f"[INFO] Emitted {emitted} VCF records.")

def parse_args():
    p = argparse.ArgumentParser(description="Inference-to-VCF for pangenome graph (CHROM=nodeID, POS=starting offset)")
    p.add_argument("data_path", type=str, help="Dataset root. Expect test/val split supporting return_paths=True.")
    p.add_argument("ckpt_path", type=str, help="Model checkpoint (.pth).")
    p.add_argument("-o", "--output_vcf", type=str, default="./calls.graph.vcf", help="Output VCF path.")
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--prob_threshold", type=float, default=0.5, help="Min probability to mark PASS / emit (unless --emit_all).")
    p.add_argument("--positive_label", type=str, default="variant",
                   help="Class name in genotype_map to treat as 'variant'. If not found, we try heuristics; specify explicitly if needed.")
    p.add_argument("--emit_all", action="store_true", help="Emit every site to VCF (even predicted non-variants).")

    # architecture (must match training)
    p.add_argument("--depths", type=int, nargs="+", default=[3, 3, 27, 3])
    p.add_argument("--dims", type=int, nargs="+", default=[192, 384, 768, 1536])
    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()
    run(
        data_path=args.data_path,
        ckpt_path=args.ckpt_path,
        output_vcf=args.output_vcf,
        depths=args.depths,
        dims=args.dims,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        prob_threshold=args.prob_threshold,
        positive_label=args.positive_label,
        emit_all=args.emit_all
    )
