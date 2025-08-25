#!/usr/bin/env python3
"""
Create a node/offset VCF from predictions + per-node variant_summary.json.

Custom semantics:
  - CHROM = node_id (directory containing the .npy tile)
  - POS   = starting offset (first token in the tile filename)
  - REF/ALT parsed from "<offset>_<X>_<REF>_<ALT>.npy"
  - Only include rows where pred_class == "true" (case-insensitive)

INFO fields written (when available from inputs):
  PROB   : probability for the 'true' class (from predictions JSON)
  SRC    : source .npy path
  AF     : alt_allele_frequency (variant_summary.json)
  DP     : coverage_at_locus (variant_summary.json)
  AD     : "ref,alt" counts (ref_allele_count_at_locus, alt_allele_count)
  OTHER  : other_allele_count_at_locus
  BQ     : mean_alt_allele_base_quality

Examples:
  python predictions_to_node_vcf.py \
    --predictions predictions.json \
    --output_vcf out/combined.vcf --sort

  python predictions_to_node_vcf.py \
    --predictions predictions.jsonl --jsonl \
    --output_vcf out/combined.vcf --min_true_prob 0.9
"""

import argparse
import json
import os
import re
import sys
from typing import Dict, Iterator, List, Optional, Tuple

from tqdm import tqdm

# Parse filenames like: 11_X_A_G.npy
NAME_RE = re.compile(
    r"""^(?P<offset>-?\d+)
        _(?P<chrom_hint>[^_]+)
        _(?P<ref>[ACGTN]+)
        _(?P<alt>[ACGTN]+)
        \.[Nn][Pp][Yy]$
    """,
    re.VERBOSE,
)


def read_json_array(path: str) -> List[dict]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("Predictions JSON must be a list when not using --jsonl.")
    return [x for x in data if isinstance(x, dict)]


def iter_jsonl(path: str) -> Iterator[dict]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s or not s.startswith("{"):
                continue
            try:
                obj = json.loads(s)
                if isinstance(obj, dict):
                    yield obj
            except Exception:
                continue


def true_prob(rec: dict) -> Optional[float]:
    # Prefer probs['true'] if present
    p = rec.get("probs")
    if isinstance(p, dict):
        for k, v in p.items():
            if str(k).lower() == "true":
                try:
                    return float(v)
                except Exception:
                    pass
    # Fallback to pred_prob
    try:
        return float(rec.get("pred_prob"))
    except Exception:
        return None


def load_variant_summary_for_node(node_dir: str) -> Dict[str, dict]:
    """
    Expected schema per your example:
    {
      "node_id": 10000009,
      "node_length": 19,
      "variants_passing_af_filter": [
        {
          "variant_key": "12_X_T_G",
          "tensor_file": "12_X_T_G.npy",
          "alt_allele_count": 14,
          "ref_allele_count_at_locus": 54,
          "other_allele_count_at_locus": 4,
          "coverage_at_locus": 72,
          "alt_allele_frequency": 0.1944,
          "mean_alt_allele_base_quality": 22.71
        }, ...
      ]
    }
    Build an index: basename(.npy) -> row dict
    """
    idx: Dict[str, dict] = {}
    path = os.path.join(node_dir, "variant_summary.json")
    if not os.path.exists(path):
        return idx
    try:
        with open(path, "r", encoding="utf-8") as f:
            summary = json.load(f)
    except Exception:
        return idx

    rows = []
    if isinstance(summary, dict) and isinstance(summary.get("variants_passing_af_filter"), list):
        rows = [r for r in summary["variants_passing_af_filter"] if isinstance(r, dict)]
    elif isinstance(summary, list):
        rows = [r for r in summary if isinstance(r, dict)]

    for r in rows:
        tf = r.get("tensor_file") or r.get("variant_key")
        if isinstance(tf, str):
            bn = os.path.basename(tf)
            if not bn.lower().endswith(".npy"):
                bn = bn + ".npy"
            idx[bn] = r
    return idx


def escape_info(v) -> str:
    return (
        str(v)
        .replace(" ", "_")
        .replace(",", "%2C")
        .replace(";", "%3B")
        .replace("=", "%3D")
    )


def main():
    ap = argparse.ArgumentParser(description="Generate node/offset VCF from predictions + variant_summary.json")
    ap.add_argument("--predictions", required=True, help="predictions.json (array) or .jsonl")
    ap.add_argument("--jsonl", action="store_true", help="Interpret --predictions as JSONL")
    ap.add_argument("--output_vcf", required=True, help="Path to write combined VCF")
    ap.add_argument("--sort", action="store_true", help="Sort by node_id (CHROM) then POS")
    ap.add_argument("--min_true_prob", type=float, default=None, help="Keep only true records with prob>=this (if prob present)")
    args = ap.parse_args()

    print("=== VCF Build Configuration ===")
    print(f"Predictions: {os.path.abspath(args.predictions)}  (jsonl={bool(args.jsonl)})")
    print(f"Output VCF : {os.path.abspath(args.output_vcf)}")
    print(f"min_true_prob: {args.min_true_prob} | sort: {args.sort}")
    print("=" * 40)

    # Iterate predictions with a flashing progress bar
    records: List[Tuple[str, int, str, str, dict]] = []
    total_in = 0
    kept_true = 0
    missing_pattern = 0
    by_node_cache: Dict[str, Dict[str, dict]] = {}

    if args.jsonl:
        it = iter_jsonl(args.predictions)
        bar = tqdm(total=None, desc="Scan preds", unit="rec", dynamic_ncols=True, leave=True)
        for obj in it:
            total_in += 1
            bar.update(1)

            if str(obj.get("pred_class", "")).strip().lower() != "true":
                bar.set_postfix_str(f"kept={kept_true}")
                continue

            path = obj.get("path") or obj.get("file") or obj.get("src")
            if not path:
                bar.set_postfix_str(f"kept={kept_true}")
                continue

            base = os.path.basename(path)
            m = NAME_RE.match(base)
            if not m:
                missing_pattern += 1
                bar.set_postfix_str(f"kept={kept_true}")
                continue

            node_id = os.path.basename(os.path.dirname(path.rstrip("/")))
            try:
                offset = int(m.group("offset"))
            except Exception:
                bar.set_postfix_str(f"kept={kept_true}")
                continue
            ref = m.group("ref")
            alt = m.group("alt")

            prob = true_prob(obj)
            if args.min_true_prob is not None and prob is not None and prob < float(args.min_true_prob):
                bar.set_postfix_str(f"kept={kept_true}")
                continue

            node_dir = os.path.dirname(path)
            if node_id not in by_node_cache:
                by_node_cache[node_id] = load_variant_summary_for_node(node_dir)

            info = {
                "PROB": f"{prob:.6g}" if prob is not None else None,
                "SRC": path,
            }
            row = by_node_cache[node_id].get(base)
            if row:
                af = row.get("alt_allele_frequency")
                cov = row.get("coverage_at_locus")
                rc = row.get("ref_allele_count_at_locus")
                ac = row.get("alt_allele_count")
                oc = row.get("other_allele_count_at_locus")
                bq = row.get("mean_alt_allele_base_quality")

                if af is not None:
                    info["AF"] = f"{float(af):.6g}"
                if cov is not None:
                    info["DP"] = str(int(cov)) if isinstance(cov, (int, float)) else escape_info(cov)
                if rc is not None and ac is not None:
                    try:
                        info["AD"] = f"{int(rc)},{int(ac)}"
                    except Exception:
                        info["AD"] = f"{rc},{ac}"
                if oc is not None:
                    try:
                        info["OTHER"] = str(int(oc))
                    except Exception:
                        info["OTHER"] = escape_info(oc)
                if bq is not None:
                    try:
                        info["BQ"] = f"{float(bq):.3f}"
                    except Exception:
                        info["BQ"] = escape_info(bq)

            records.append((node_id, offset, ref, alt, info))
            kept_true += 1
            bar.set_postfix_str(f"kept={kept_true}")
        bar.close()

    else:
        data = read_json_array(args.predictions)
        bar = tqdm(total=len(data), desc="Scan preds", unit="rec", dynamic_ncols=True, leave=True)
        for obj in data:
            total_in += 1
            bar.update(1)

            if str(obj.get("pred_class", "")).strip().lower() != "true":
                bar.set_postfix_str(f"kept={kept_true}")
                continue

            path = obj.get("path") or obj.get("file") or obj.get("src")
            if not path:
                bar.set_postfix_str(f"kept={kept_true}")
                continue

            base = os.path.basename(path)
            m = NAME_RE.match(base)
            if not m:
                missing_pattern += 1
                bar.set_postfix_str(f"kept={kept_true}")
                continue

            node_id = os.path.basename(os.path.dirname(path.rstrip("/")))
            try:
                offset = int(m.group("offset"))
            except Exception:
                bar.set_postfix_str(f"kept={kept_true}")
                continue
            ref = m.group("ref")
            alt = m.group("alt")

            prob = true_prob(obj)
            if args.min_true_prob is not None and prob is not None and prob < float(args.min_true_prob):
                bar.set_postfix_str(f"kept={kept_true}")
                continue

            node_dir = os.path.dirname(path)
            if node_id not in by_node_cache:
                by_node_cache[node_id] = load_variant_summary_for_node(node_dir)

            info = {
                "PROB": f"{prob:.6g}" if prob is not None else None,
                "SRC": path,
            }
            row = by_node_cache[node_id].get(base)
            if row:
                af = row.get("alt_allele_frequency")
                cov = row.get("coverage_at_locus")
                rc = row.get("ref_allele_count_at_locus")
                ac = row.get("alt_allele_count")
                oc = row.get("other_allele_count_at_locus")
                bq = row.get("mean_alt_allele_base_quality")

                if af is not None:
                    info["AF"] = f"{float(af):.6g}"
                if cov is not None:
                    info["DP"] = str(int(cov)) if isinstance(cov, (int, float)) else escape_info(cov)
                if rc is not None and ac is not None:
                    try:
                        info["AD"] = f"{int(rc)},{int(ac)}"
                    except Exception:
                        info["AD"] = f"{rc},{ac}"
                if oc is not None:
                    try:
                        info["OTHER"] = str(int(oc))
                    except Exception:
                        info["OTHER"] = escape_info(oc)
                if bq is not None:
                    try:
                        info["BQ"] = f"{float(bq):.3f}"
                    except Exception:
                        info["BQ"] = escape_info(bq)

            records.append((node_id, offset, ref, alt, info))
            kept_true += 1
            bar.set_postfix_str(f"kept={kept_true}")
        bar.close()

    if missing_pattern:
        tqdm.write(f"WARNING: {missing_pattern} filenames did not match '<offset>_<X>_<REF>_<ALT>.npy' and were skipped.")

    if kept_true == 0:
        print("No qualifying 'true' predictions. Nothing to write.", file=sys.stderr)
        sys.exit(0)

    # Optional sort
    if args.sort:
        def chrom_key(c: str):
            try:
                return (0, int(c))
            except Exception:
                return (1, c)
        records.sort(key=lambda r: (chrom_key(r[0]), r[1]))

    # Write VCF with a flashing bar
    out_path = os.path.abspath(args.output_vcf)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    with open(out_path, "w", encoding="utf-8") as out:
        # Header (custom semantics doc)
        out.write("##fileformat=VCFv4.2\n")
        out.write("##META=<ID=SCHEMA,Description=\"Custom: CHROM=node_id, POS=starting_offset\">\n")
        out.write("##INFO=<ID=PROB,Number=1,Type=Float,Description=\"Model probability for class 'true'\">\n")
        out.write("##INFO=<ID=SRC,Number=1,Type=String,Description=\"Source .npy path\">\n")
        out.write("##INFO=<ID=AF,Number=1,Type=Float,Description=\"Alt allele frequency from variant_summary.json\">\n")
        out.write("##INFO=<ID=DP,Number=1,Type=Integer,Description=\"Coverage at locus from variant_summary.json\">\n")
        out.write("##INFO=<ID=AD,Number=R,Type=Integer,Description=\"Allele depths: ref,alt (from variant_summary.json)\">\n")
        out.write("##INFO=<ID=OTHER,Number=1,Type=Integer,Description=\"Other allele count at locus\">\n")
        out.write("##INFO=<ID=BQ,Number=1,Type=Float,Description=\"Mean alt allele base quality\">\n")

        # Declare contigs (node IDs)
        seen = set()
        for chrom, _, _, _, _ in records:
            if chrom not in seen:
                out.write(f"##contig=<ID={chrom}>\n")
                seen.add(chrom)

        out.write("#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\n")

        with tqdm(total=len(records), desc="Write VCF", unit="rec", dynamic_ncols=True, leave=True) as wbar:
            for chrom, pos, ref, alt, info in records:
                info_str = ";".join(f"{k}={escape_info(v)}" for k, v in info.items() if v is not None) or "."
                out.write(f"{chrom}\t{pos}\t.\t{ref}\t{alt}\t.\tPASS\t{info_str}\n")
                wbar.update(1)

    print(f"VCF written: {out_path} (records={len(records)}, scanned={total_in}, kept_true={kept_true})")


if __name__ == "__main__":
    main()
