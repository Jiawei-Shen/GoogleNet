#!/usr/bin/env python3
import argparse
import os
import re
import sys
import json
from typing import List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# repo-local imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from mynet import ConvNeXtCBAMClassifier

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Example: 1623868_162_X_A_C.npy
FNAME_PAT = re.compile(
    r"^(?P<node>\d+?)_(?P<start>\d+?)_(?P<etype>[XID])_(?P<ref>[ACGTN\-]+)_(?P<alt>[ACGTN\-]+)\.npy$",
    re.IGNORECASE
)

def parse_graph_fname(basename: str) -> Tuple[str, int, str, str, str]:
    m = FNAME_PAT.match(basename)
    if not m:
        raise ValueError(f"Filename not in expected pattern: {basename}")
    return (
        m.group("node"),
        int(m.group("start")),
        m.group("ref").upper(),
        m.group("alt").upper(),
        m.group("etype").upper(),
    )

def to_chw(arr: np.ndarray, in_channels: int) -> np.ndarray:
    """
    Convert array to shape [C,H,W]. Supports [C,H,W] and [H,W,C].
    """
    if arr.ndim == 3:
        if arr.shape[0] == in_channels:
            return arr
        if arr.shape[-1] == in_channels:
            return np.transpose(arr, (2, 0, 1))
    if arr.ndim == 2 and in_channels == 1:
        return arr[None, ...]
    raise ValueError(f"Unsupported npy shape {arr.shape}; expected CxHxW or HxWxC with C={in_channels}")

class NpyFolderDataset(Dataset):
    def __init__(self, root: str, in_channels: int):
        self.in_channels = in_channels
        self.files: List[str] = []
        for dirpath, _, filenames in os.walk(root):
            for fn in filenames:
                if fn.endswith(".npy"):
                    self.files.append(os.path.join(dirpath, fn))
        if not self.files:
            raise RuntimeError(f"No .npy files found under: {root}")
        self.files.sort()

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path = self.files[idx]
        arr = np.load(path, allow_pickle=False)
        arr = to_chw(arr, self.in_channels).astype(np.float32, copy=False)
        tensor = torch.from_numpy(arr)  # [C,H,W]
        return tensor, path

def softmax_probs(logits: torch.Tensor) -> torch.Tensor:
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

def infer_positive_index(genotype_map, positive_label_cli):
    if positive_label_cli is not None:
        if positive_label_cli in genotype_map:
            return genotype_map[positive_label_cli]
        for k, v in genotype_map.items():
            if str(k) == str(positive_label_cli):
                return v
        raise ValueError(f"--positive_label '{positive_label_cli}' not found in genotype_map keys: {list(genotype_map.keys())}")
    # heuristic
    for k, v in genotype_map.items():
        if isinstance(k, str) and (("var" in k.lower()) or ("alt" in k.lower()) or ("pos" in k.lower())):
            return v
    if len(genotype_map) == 2:
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
    folder_path,
    ckpt_path,
    output_vcf,
    depths,
    dims,
    batch_size=64,
    num_workers=8,
    prob_threshold=0.5,
    positive_label="variant",
    emit_all=False
):
    os.makedirs(os.path.dirname(os.path.abspath(output_vcf)) or ".", exist_ok=True)

    print(f"[INFO] Loading checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state = ckpt.get("model_state_dict", ckpt)
    genotype_map = ckpt.get("genotype_map")
    if not genotype_map:
        raise RuntimeError("Checkpoint must contain 'genotype_map' for class indexing.")
    in_channels = ckpt.get("in_channels", 6)
    model_name = os.path.basename(ckpt_path)

    pos_idx = infer_positive_index(genotype_map, positive_label)
    idx_to_class = {v: k for k, v in genotype_map.items()}

    model = build_model(in_channels, num_classes=len(genotype_map), depths=depths, dims=dims, state_dict=state)

    # Dataset & DataLoader over the folder (recursive)
    ds = NpyFolderDataset(folder_path, in_channels=in_channels)
    dl = DataLoader(ds, batch_size=batch_size, num_workers=num_workers, shuffle=False, pin_memory=True)

    vcf_f = open_vcf(output_vcf, model_name)
    emitted = 0
    skipped_unparsable = 0

    with torch.no_grad():
        for batch in tqdm(dl, desc="Inferring & writing VCF", leave=True):
            images, paths = batch
            images = images.to(device, non_blocking=True)
            logits = model(images)
            probs = softmax_probs(logits)
            confs, pred_idx = torch.max(probs, dim=1)

            for i in range(images.size(0)):
                bname = os.path.basename(str(paths[i]))
                try:
                    node, start, ref, alt, etype = parse_graph_fname(bname)
                except Exception as e:
                    skipped_unparsable += 1
                    continue

                p_idx = int(pred_idx[i].item())
                p_name = str(idx_to_class.get(p_idx, p_idx))
                prob = float(confs[i].item())
                is_variant_call = (p_idx == pos_idx) and (prob >= prob_threshold)

                if not emit_all and not is_variant_call:
                    continue

                filt = "PASS" if prob >= prob_threshold else "LowQual"
                info = {"PROB": f"{prob:.6f}", "PRED": p_name, "ETYPE": etype, "FILE": bname}
                write_record(vcf_f, node=node, pos=start, ref=ref, alt=alt, qual=prob, filt=filt, info_dict=info)
                emitted += 1

    vcf_f.close()
    print(f"[INFO] Wrote VCF: {output_vcf}")
    print(f"[INFO] Emitted {emitted} VCF records.")
    if skipped_unparsable:
        print(f"[INFO] Skipped {skipped_unparsable} files with unparsable names (pattern node_start_[X|I|D]_ref_alt.npy).")

def parse_args():
    p = argparse.ArgumentParser(description="Folder-recursive inference-to-VCF (CHROM=nodeID, POS=starting offset)")
    p.add_argument("folder_path", type=str, help="Root folder to scan recursively for .npy files.")
    p.add_argument("ckpt_path", type=str, help="Model checkpoint (.pth) containing model_state_dict and genotype_map.")
    p.add_argument("-o", "--output_vcf", type=str, default="./calls.graph.vcf", help="Output VCF path.")
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--prob_threshold", type=float, default=0.5, help="Min probability to emit (unless --emit_all).")
    p.add_argument("--positive_label", type=str, default="variant", help="Label in genotype_map treated as 'variant'.")
    p.add_argument("--emit_all", action="store_true", help="Emit every site, including predicted non-variants.")
    p.add_argument("--depths", type=int, nargs="+", default=[3, 3, 27, 3], help="ConvNeXt depths; must match training.")
    p.add_argument("--dims", type=int, nargs="+", default=[192, 384, 768, 1536], help="ConvNeXt dims; must match training.")
    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()
    run(
        folder_path=args.folder_path,
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
