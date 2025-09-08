#!/usr/bin/env python3
"""
Count variants by node class: reference nodes, ref-alt nodes, and alt-alt nodes.

Definitions (based on your JSON):
- Reference node: JSON record has population AF (or other extra metadata),
  OR in general has more than just node id + reference sequence.
- Ref-alt node: JSON record only includes "node id" and a reference/sequence field.
- Alt-alt node: node id not present in the JSON at all.

Input:
  - Node/metadata JSON (array/dict OR JSONL).
  - VCF where CHROM is a numeric node_id (node/offset format). Non-numeric CHROM lines are ignored.

Output:
  - Prints counts and percentages for each class.
  - Optional TSV with counts.

Usage:
  python count_variant_node_classes.py \
      --in_vcf node_coords.vcf.gz \
      --map_json nodes.json \
      [--map_jsonl] \
      [--out_tsv counts.tsv]
"""

import argparse
import gzip
import json
import sys
from typing import Dict, Iterable, Iterator, Optional, Tuple

# ---- JSON helpers ----

ID_KEYS = {"node_id", "node", "id", "nid"}
REFSEQ_KEYS = {"reference", "ref", "ref_seq", "refseq", "sequence", "seq"}

# AF-like keys we’ll recognize explicitly; extend as needed.
AF_KEYS = {
    "af", "population_af", "pop_af", "allele_frequency", "global_af",
    "gnomad_af", "af_popmax", "af_1000g", "af_gnomad", "af_graph"
}

def _get_first(d: dict, keys: Iterable[str]):
    for k in keys:
        if k in d:
            return d[k]
    return None

def _node_id_from_obj(obj: dict) -> Optional[int]:
    nid = _get_first(obj, ID_KEYS)
    if nid is None:
        return None
    try:
        return int(nid)
    except Exception:
        return None

def _has_af_like(obj: dict) -> bool:
    for k, v in obj.items():
        lk = str(k).lower()
        if lk in AF_KEYS:
            # Any non-empty / numeric value counts
            if v is None:
                continue
            try:
                float(v)
                return True
            except Exception:
                # If it's not directly numeric but present, still treat as AF-like
                return True
    return False

def _is_ref_alt_minimal(obj: dict) -> bool:
    """
    True if the record is essentially {id + reference/seq} and nothing else.
    """
    keys = set(obj.keys())
    # must have at least one id-key and one ref-seq key, and nothing beyond those
    has_id = any(k in keys for k in ID_KEYS)
    has_ref = any(k in keys for k in REFSEQ_KEYS)
    extra = keys - (ID_KEYS | REFSEQ_KEYS)
    return has_id and has_ref and len(extra) == 0

def iter_json_records(path: str, jsonl: bool) -> Iterator[dict]:
    if jsonl:
        # JSON Lines: one JSON object per line
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                s = line.strip()
                if not s or not s.startswith("{"):
                    continue
                try:
                    obj = json.loads(s)
                except Exception:
                    continue
                if isinstance(obj, dict):
                    yield obj
    else:
        # JSON: allow dict-of-dicts or list-of-dicts
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            for v in data.values():
                if isinstance(v, dict):
                    yield v
        elif isinstance(data, list):
            for v in data:
                if isinstance(v, dict):
                    yield v
        else:
            raise ValueError("Unsupported JSON structure (expect list/dict of dicts or JSONL).")

def build_node_class_map(json_path: str, jsonl: bool) -> Dict[int, str]:
    """
    Returns a mapping: node_id -> class {'reference', 'ref-alt'}.
    Nodes absent from this map are implicitly 'alt-alt'.
    """
    cls_map: Dict[int, str] = {}

    for obj in iter_json_records(json_path, jsonl):
        nid = _node_id_from_obj(obj)
        if nid is None:
            continue

        # Priority: explicit AF-like -> 'reference'
        if _has_af_like(obj):
            cls_map[nid] = "reference"
            continue

        # Minimal (id + reference only) -> 'ref-alt'
        if _is_ref_alt_minimal(obj):
            # Only set if not already marked as 'reference'
            cls_map.setdefault(nid, "ref-alt")
            continue

        # Otherwise, has "some other information" -> 'reference'
        cls_map[nid] = "reference"

    return cls_map

# ---- VCF helpers ----

def open_text_auto(path: str):
    if path.endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8")
    return open(path, "r", encoding="utf-8")

def count_variants_by_class(vcf_path: str, node_class: Dict[int, str]) -> Tuple[int, int, int, int]:
    """
    Returns tuple: (n_reference, n_ref_alt, n_alt_alt, n_non_numeric_chrom)
    Non-numeric CHROM lines (e.g., linear contigs) are counted separately and excluded from classes.
    """
    n_ref = n_refalt = n_altalt = n_nonnum = 0
    with open_text_auto(vcf_path) as f:
        for line in f:
            if not line or line[0] == "#":
                continue
            cols = line.rstrip("\n").split("\t")
            if len(cols) < 2:
                continue
            chrom = cols[0]
            try:
                nid = int(chrom)
            except Exception:
                n_nonnum += 1
                continue

            cls = node_class.get(nid)
            if cls == "reference":
                n_ref += 1
            elif cls == "ref-alt":
                n_refalt += 1
            else:
                n_altalt += 1
    return n_ref, n_refalt, n_altalt, n_nonnum

def pct(n: int, d: int) -> str:
    return f"{(100.0 * n / d):.2f}%" if d else "0.00%"

def main():
    ap = argparse.ArgumentParser(description="Count VCF records by node class (reference, ref-alt, alt-alt) using a node JSON.")
    ap.add_argument("--in_vcf", required=True, help="Input VCF (.vcf or .vcf.gz) where CHROM is a numeric node_id.")
    ap.add_argument("--map_json", required=True, help="Node map JSON (list/dict) or JSONL of node records.")
    ap.add_argument("--map_jsonl", action="store_true", help="Interpret --map_json as JSONL.")
    ap.add_argument("--out_tsv", default=None, help="Optional: write counts to this TSV.")
    args = ap.parse_args()

    print("=== Node class counting ===")
    print(f"VCF       : {args.in_vcf}")
    print(f"Node JSON : {args.map_json} (jsonl={bool(args.map_jsonl)})")

    node_class = build_node_class_map(args.map_json, args.map_jsonl)
    n_ref_nodes  = sum(1 for c in node_class.values() if c == "reference")
    n_ralt_nodes = sum(1 for c in node_class.values() if c == "ref-alt")
    print(f"Node classes from JSON -> reference:{n_ref_nodes}  ref-alt:{n_ralt_nodes}  (others default to alt-alt)")

    n_ref, n_refalt, n_altalt, n_nonnum = count_variants_by_class(args.in_vcf, node_class)
    total = n_ref + n_refalt + n_altalt
    grand = total + n_nonnum

    print("\n=== Variant counts (by VCF record) ===")
    print(f"reference : {n_ref:10d}  ({pct(n_ref, total)})")
    print(f"ref-alt   : {n_refalt:10d}  ({pct(n_refalt, total)})")
    print(f"alt-alt   : {n_altalt:10d}  ({pct(n_altalt, total)})")
    print(f"----------------------------------------")
    print(f"TOTAL (node-CHROM only): {total:10d}")
    if n_nonnum:
        print(f"Non-numeric CHROM (ignored): {n_nonnum}  (grand total lines incl. linear = {grand})")

    if args.out_tsv:
        try:
            with open(args.out_tsv, "w", encoding="utf-8") as w:
                w.write("class\tcount\tpercent_of_node_records\n")
                w.write(f"reference\t{n_ref}\t{pct(n_ref, total)}\n")
                w.write(f"ref-alt\t{n_refalt}\t{pct(n_refalt, total)}\n")
                w.write(f"alt-alt\t{n_altalt}\t{pct(n_altalt, total)}\n")
                w.write(f"TOTAL_NODE_CHROM\t{total}\t100.00%\n")
                if n_nonnum:
                    w.write(f"NON_NUMERIC_CHROM_IGNORED\t{n_nonnum}\t-\n")
            print(f"\nWrote TSV: {args.out_tsv}")
        except Exception as e:
            print(f"WARNING: failed to write --out_tsv: {e}", file=sys.stderr)

if __name__ == "__main__":
    main()
