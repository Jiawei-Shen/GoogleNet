#!/usr/bin/env python3
"""
Create a node/offset VCF from predictions + per-node variant_summary.json.

Semantics:
  - CHROM = node_id (directory containing the .npy tile)
  - POS   = starting offset (first token in the tile filename)
  - REF/ALT parsed from "<offset>_<X>_<REF>_<ALT>.npy"
  - Only include rows where pred_class == "true" (case-insensitive)

INFO fields written (when available):
  PROB   : probability for the 'true' class (from predictions JSON)
  SRC    : source .npy path
  AF     : alt_allele_frequency from variant_summary.json
  DP     : coverage_at_locus
  AD     : "ref,alt" counts (ref_allele_count_at_locus,alt_allele_count)
  OTHER  : other_allele_count_at_locus
  BQ     : mean_alt_allele_base_quality

Usage:
  python predictions_to_node_vcf.py \
    --predictions predictions.json \
    --output_vcf out/combined.vcf \
    --sort

  # JSONL input
  python predictions_to_node_vcf.py \
    --predictions predictions.jsonl --jsonl \
    --output_vcf out/combined.vcf
"""

import argparse
import json
import os
import re
import sys
from typing import Dict, Iterator, List, Optional, Tuple

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

def read_json_array(path: str) -> Iterator[dict]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        for obj in data:
            if isinstance(obj, dict):
                yield obj

def read_jsonl(path: str) -> Iterator[dict]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s or not s.startswith("{"):
                continue
            try:
                yield json.loads(s)
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
    # Fallback to pred_prob (may already be 'true' prob if true is predicted)
    try:
        return float(rec.get("pred_prob"))
    except Exception:
        return None

def load_variant_summary_for_node(node_dir: str) -> Dict[str, dict]:
    """
    Expects schema:
    {
      "node_id": 10000009,
      "node_length": 19,
      "variants_passing_af_filter": [ { ... per-variant ... }, ... ]
    }
    We build an index by 'tensor_file' (basename).
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
        # loose fallback (not expected with your schema)
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
    ap = argparse.ArgumentParser(description="Generate node/offset VCF from predictions JSON and variant_summary.json")
    ap.add_argument("--predictions", required=True, help="predictions.json (array) or .jsonl")
    ap.add_argument("--jsonl", action="store_true", help="Interpret --predictions as JSONL")
    ap.add_argument("--output_vcf", required=True, help="Path to write combined VCF")
    ap.add_argument("--sort", action="store_true", help="Sort by node_id (CHROM) then POS")
    ap.add_argument("--min_true_prob", type=float, default=None, help="Keep only true records with prob>=this (if prob present)")
    args = ap.parse_args()

    # Iterate predictions
    it = read_jsonl(args.predictions) if args.jsonl else read_json_array(args.predictions)

    records: List[Tuple[str,int,str,str,dict]] = []
    total_in = 0
    kept_true = 0
    by_node_cache: Dict[str, Dict[str, dict]] = {}

    for obj in it:
        total_in += 1
        if str(obj.get("pred_class", "")).strip().lower() != "true":
            continue

        # Path and filename
        path = obj.get("path") or obj.get("file") or obj.get("src")
        if not path:
            continue
        base = os.path.basename(path)
        m = NAME_RE.match(base)
        if not m:
            continue

        # Parse node_id and offset/ref/alt
        node_id = os.path.basename(os.path.dirname(path.rstrip("/")))
        try:
            offset = int(m.group("offset"))
        except Exception:
            continue
        ref = m.group("ref")
        alt = m.group("alt")

        prob = true_prob(obj)
        if args.min_true_prob is not None and prob is not None and prob < float(args.min_true_prob):
            continue

        # Load per-node summary once
        node_dir = os.path.dirname(path)
        if node_id not in by_node_cache:
            by_node_cache[node_id] = load_variant_summary_for_node(node_dir)
        summary_idx = by_node_cache[node_id]

        info = {
            "PROB": f"{prob:.6g}" if prob is not None else None,
            "SRC": path,
        }

        # Fill AF/DP/AD/OTHER/BQ when available
        row = summary_idx.get(base)
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

    # Write VCF
    out_path = os.path.abspath(args.output_vcf)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    with open(out_path, "w", encoding="utf-8") as out:
        # Header explains custom CHROM/POS semantics
        out.write("##fileformat=VCFv4.2\n")
        out.write("##META=<ID=SCHEMA,Description=\"Custom: CHROM=node_id, POS=starting_offset\">\n")
        out.write("##INFO=<ID=PROB,Number=1,Type=Float,Description=\"Model probability for class 'true'\">\n")
        out.write("##INFO=<ID=SRC,Number=1,Type=String,Description=\"Source .npy path\">\n")
        out.write("##INFO=<ID=AF,Number=1,Type=Float,Description=\"Alt allele frequency from variant_summary.json\">\n")
        out.write("##INFO=<ID=DP,Number=1,Type=Integer,Description=\"Coverage at locus from variant_summary.json\">\n")
        out.write("##INFO=<ID=AD,Number=R,Type=Integer,Description=\"Allele depths: ref,alt (from variant_summary.json)\">\n")
        out.write("##INFO=<ID=OTHER,Number=1,Type=Integer,Description=\"Other allele count at locus\">\n")
        out.write("##INFO=<ID=BQ,Number=1,Type=Float,Description=\"Mean alt allele base quality\">\n")

        # Declare contigs as node IDs we actually use
        seen = set()
        for chrom, _, _, _, _ in records:
            if chrom not in seen:
                out.write(f"##contig=<ID={chrom}>\n")
                seen.add(chrom)

        out.write("#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\n")
        for chrom, pos, ref, alt, info in records:
            info_str = ";".join(f"{k}={escape_info(v)}" for k, v in info.items() if v is not None) or "."
            out.write(f"{chrom}\t{pos}\t.\t{ref}\t{alt}\t.\tPASS\t{info_str}\n")

    print(f"Wrote VCF: {out_path} (records={len(records)}, input={total_in}, kept_true={kept_true})")

if __name__ == "__main__":
    main()
