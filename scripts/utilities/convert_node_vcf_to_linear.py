#!/usr/bin/env python3
"""
Convert a node/offset VCF to linear-reference coordinates, writing TWO outputs:
  1) --out_linear_vcf : records successfully converted to linear (CHROM=chr*, POS=linear)
  2) --out_nodes_vcf  : records that stayed in node form (no mapping found)

Inputs
------
- in_vcf : VCF where CHROM=node_id and POS=starting offset
- map_json : node map (JSON array or JSONL) with fields like:
    {
      "node_id": "39168232",
      "grch38_position_start": 111235483,   # 1-based
      "strand_in_path": "+",                # "+" or "-"
      "length": 28,
      "chrom": "chr4",
      ...
    }
  Some entries may include only {node_id, sequence}. Those cannot be used directly;
  for ALT nodes, we try to find their REF node via TSV and map using the REF node.

- tsv (optional): ALT→REF mapping, columns like
    CHROM POS ID TYPE REF_BASE REF_NODE ALT_STR ALT_NODE(S)
  We parse ALT_NODE(S) list and build a dict alt_node -> ref_node (ints).

Behavior
--------
- If a VCF record's node is listed as ALT in TSV, we map to its REF node for conversion.
- If a node (or its REF node) lacks a linear start in the JSON, the record is **not converted**.
- Strand-aware: for strand '-', linear position is computed from the node's start and length.

Output columns
--------------
Both outputs have trailing columns:
  NODE   : original node_id
  OFFSET : original starting offset (from input VCF POS)

Usage
-----
python convert_node_vcf_to_linear.py \
  --in_vcf node_coords.vcf \
  --map_json node_map.json \
  --out_linear_vcf out/linear.vcf \
  --out_nodes_vcf out/unconverted_nodes.vcf \
  --tsv alt_ref_map.tsv \
  --sort
"""

import argparse
import csv
import json
import os
import re
import sys
from typing import Dict, Iterable, List, Optional, Tuple

from tqdm import tqdm


# ------------------------ TSV (ALT->REF) mapping ------------------------ #

def load_alt_to_ref_tsv(tsv_path: Optional[str]) -> Dict[int, int]:
    """
    Load TSV with columns like:
    CHROM POS ID TYPE REF_BASE REF_NODE ALT_STR ALT_NODE(S)
    Return: dict alt_node -> ref_node (ints)
    """
    if not tsv_path:
        return {}
    mapping: Dict[int, int] = {}
    with open(tsv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        hdr = [c.strip() for c in (reader.fieldnames or [])]
        ref_node_key = None
        alt_nodes_key = None
        for k in hdr:
            lk = k.lower()
            if ref_node_key is None and lk in ("ref_node", "refnode", "ref"):
                ref_node_key = k
            if alt_nodes_key is None and lk in ("alt_node(s)", "alt_nodes", "altnodes", "alt"):
                alt_nodes_key = k
        if ref_node_key is None or alt_nodes_key is None:
            raise ValueError("TSV must have REF_NODE and ALT_NODE(S) columns (case-insensitive).")

        for row in reader:
            try:
                ref_node = int(str(row[ref_node_key]).strip())
            except Exception:
                continue
            alts_raw = str(row[alt_nodes_key]).strip()
            # Extract any integer tokens (accept comma/semicolon/space/etc. separators)
            for token in re.findall(r"\d+", alts_raw):
                try:
                    alt_node = int(token)
                except Exception:
                    continue
                mapping[alt_node] = ref_node
    return mapping


# ------------------------ Node map (JSON/JSONL) ------------------------ #

def _first_present(d: dict, keys: Iterable[str], default=None):
    for k in keys:
        if k in d:
            return d[k]
    return default

def _to_int(x) -> Optional[int]:
    try:
        return int(x)
    except Exception:
        return None

def iter_node_map_records(json_path: str, jsonl: bool) -> Iterable[dict]:
    """
    Yield dicts from a JSON array file or JSONL file.
    """
    if jsonl:
        with open(json_path, "r", encoding="utf-8") as f:
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
    else:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            for obj in data:
                if isinstance(obj, dict):
                    yield obj
        elif isinstance(data, dict):
            # tolerate dict-of-dicts keyed by node_id
            for _, v in data.items():
                if isinstance(v, dict):
                    yield v
        else:
            raise ValueError("Unsupported JSON structure for node map.")


def load_node_map(json_path: str,
                  jsonl: bool,
                  node_start_is_1_based: bool = True) -> Tuple[
                      Dict[int, Tuple[str, int, str, int]],  # anchors: node -> (chrom, start0, strand, length)
                      set                                       # nodes_seen (even if no anchor)
                  ]:
    """
    Build:
      anchors[node_id] = (chrom, start0, strand, length)
      nodes_seen       = set of node_ids present in JSON (even if no anchor fields)

    start0 is 0-based coordinate for the node start on the linear chromosome.
    """
    anchors: Dict[int, Tuple[str, int, str, int]] = {}
    nodes_seen: set = set()

    for rec in iter_node_map_records(json_path, jsonl):
        nid = _first_present(rec, ["node_id", "node", "id", "nid"])
        node_id = _to_int(nid)
        if node_id is None:
            continue
        nodes_seen.add(node_id)

        chrom = _first_present(rec, ["chrom", "chr", "chromosome", "contig"])
        start1 = _first_present(rec, ["grch38_position_start", "start_1based", "pos1"])
        start0 = _first_present(rec, ["start_0based", "pos0"])
        strand = str(_first_present(rec, ["strand_in_path", "strand"], "+") or "+")
        length = _to_int(_first_present(rec, ["length", "len", "node_length"]))

        # Decide start0
        if start0 is not None:
            s0 = _to_int(start0)
        elif start1 is not None:
            s1 = _to_int(start1)
            s0 = (s1 - 1) if s1 is not None else None
        else:
            # Maybe a generic 'start'
            generic = _first_present(rec, ["start", "pos", "start_pos"])
            if generic is not None:
                sg = _to_int(generic)
                if sg is not None:
                    s0 = (sg - 1) if node_start_is_1_based else sg
                else:
                    s0 = None
            else:
                s0 = None

        if chrom is not None and s0 is not None and length is not None:
            anchors[node_id] = (str(chrom), int(s0), strand if strand in ("+", "-") else "+", int(length))

    return anchors, nodes_seen


# ------------------------ VCF helpers ------------------------ #

def parse_vcf_header_and_count_records(vcf_path: str) -> Tuple[List[str], int]:
    """Return (header_lines, n_records)."""
    header: List[str] = []
    n = 0
    with open(vcf_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.startswith("#"):
                header.append(line.rstrip("\n"))
            else:
                n += 1
    return header, n


# ------------------------ Conversion core ------------------------ #

def convert_vcf(
    in_vcf: str,
    out_linear_vcf: str,
    out_nodes_vcf: str,
    anchors: Dict[int, Tuple[str, int, str, int]],
    nodes_seen_in_json: set,
    alt_to_ref: Dict[int, int],
    offset_is_0_based: bool,
    sort_records: bool,
) -> None:
    """
    Stream the input VCF and split into two outputs:
      - out_linear_vcf: converted to linear coords (CHROM=chr*, POS=linear)
      - out_nodes_vcf : stayed in node/offset coords
    Both outputs keep original INFO and append trailing NODE/OFFSET columns.
    """
    header_lines, total = parse_vcf_header_and_count_records(in_vcf)

    # We’ll keep most of the original meta lines except the #CHROM header and old contigs
    kept_header = [h for h in header_lines if not h.startswith("##contig=") and not h.startswith("#CHROM")]

    converted: List[Tuple[str, int, str, str, str, str, str, str, str]] = []
    unconverted: List[Tuple[str, int, str, str, str, str, str, str, str]] = []
    # tuples are: (CHROM, POS, ID, REF, ALT, QUAL, FILTER, INFO, NODE, OFFSET)

    with open(in_vcf, "r", encoding="utf-8") as fin, \
         tqdm(total=total if total > 0 else None, desc="Convert", unit="rec",
              dynamic_ncols=True, leave=True) as bar:

        for line in fin:
            if line.startswith("#"):
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 8:
                bar.update(1 if total else 0)
                continue

            chrom_node = parts[0]
            pos_offset = parts[1]
            vid = parts[2] if len(parts) > 2 else "."
            ref = parts[3] if len(parts) > 3 else "N"
            alt = parts[4] if len(parts) > 4 else "N"
            qual = parts[5] if len(parts) > 5 else "."
            filt = parts[6] if len(parts) > 6 else "PASS"
            info = parts[7] if len(parts) > 7 else "."

            # Parse numeric node/offset
            try:
                node_id = int(chrom_node)
                offset = int(pos_offset)
            except Exception:
                # Not a node-form record; keep in "unconverted" as-is but preserve NODE/OFFSET placeholders
                unconverted.append((chrom_node, int(pos_offset) if pos_offset.isdigit() else pos_offset,
                                    vid, ref, alt, qual, filt, info, chrom_node, pos_offset))
                bar.update(1 if total else 0)
                continue

            base_node = node_id

            # Prefer ref-node mapping if this node is ALT in TSV
            ref_node_candidate = alt_to_ref.get(base_node)
            anchor = None

            if ref_node_candidate is not None:
                anchor = anchors.get(ref_node_candidate)
            if anchor is None:
                anchor = anchors.get(base_node)

            # If still no anchor but node exists in JSON only with sequence,
            # and alt->ref map exists, try the ref of this node (if any).
            if anchor is None and base_node in nodes_seen_in_json:
                if ref_node_candidate is not None:
                    anchor = anchors.get(ref_node_candidate)

            if anchor is None:
                # Could not convert — keep in node coords
                unconverted.append((chrom_node, offset, vid, ref, alt, qual, filt, info, str(node_id), str(offset)))
                bar.update(1 if total else 0)
                continue

            chrom_lin, start0, strand, length = anchor

            # Compute linear 1-based POS with strand handling
            if offset_is_0_based:
                off0 = offset
            else:
                off0 = offset - 1

            if strand == "+":
                pos_lin = start0 + off0 + 1
            else:
                # On '-' strand: position counts from the "end" of the node
                # Example: node spans [start0 .. start0+length-1] on chrom (0-based)
                # offset 0 -> last base => pos = start0 + (length-1 - 0) + 1
                pos_lin = start0 + (length - 1 - off0) + 1

            converted.append((chrom_lin, int(pos_lin), vid, ref, alt, qual, filt, info, str(node_id), str(offset)))
            bar.update(1 if total else 0)

    # Optional sort
    if sort_records:
        def chrom_key(c: str):
            m = re.fullmatch(r"(?:chr)?(\d+)", c, flags=re.IGNORECASE)
            if m:
                return (0, int(m.group(1)))
            if c.lower() in ("chrx", "x"):
                return (1, 23)
            if c.lower() in ("chry", "y"):
                return (1, 24)
            if c.lower() in ("chrm", "chrmt", "m", "mt"):
                return (1, 25)
            return (2, c.lower())
        converted.sort(key=lambda r: (chrom_key(r[0]), int(r[1])))
        # Keep node-form in original order (debuggability), or sort similarly if you prefer:
        # unconverted.sort(key=lambda r: (int(r[8]) if str(r[8]).isdigit() else 10**12, int(r[9]) if str(r[9]).isdigit() else 10**12))

    # Gather contigs for each output
    lin_contigs = []
    seen_lin = set()
    for c, *_ in converted:
        if c not in seen_lin:
            seen_lin.add(c)
            lin_contigs.append(c)

    node_contigs = []
    seen_node = set()
    for c, *_ in unconverted:
        if isinstance(c, int):
            cc = str(c)
        else:
            cc = str(c)
        if cc not in seen_node:
            seen_node.add(cc)
            node_contigs.append(cc)

    # Write linear VCF
    os.makedirs(os.path.dirname(out_linear_vcf), exist_ok=True)
    with open(out_linear_vcf, "w", encoding="utf-8") as out:
        for h in kept_header:
            if h.startswith("#CHROM"):
                continue
            out.write(h + "\n")

        out.write('##META=<ID=CONVERTED,Description="CHROM/POS converted from node_id/offset using node map and optional ALT->REF TSV. '
                  'Trailing NODE/OFFSET preserve originals. Strand-aware.">\n')
        for c in lin_contigs:
            out.write(f"##contig=<ID={c}>\n")

        out.write("#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tNODE\tOFFSET\n")

        with tqdm(total=len(converted), desc="Write linear", unit="rec", dynamic_ncols=True, leave=True) as wbar:
            for chrom, pos, vid, ref, alt, qual, filt, info, node, off in converted:
                out.write(f"{chrom}\t{pos}\t{vid}\t{ref}\t{alt}\t{qual}\t{filt}\t{info}\t{node}\t{off}\n")
                wbar.update(1)

    # Write node-form VCF (unconverted)
    os.makedirs(os.path.dirname(out_nodes_vcf), exist_ok=True)
    with open(out_nodes_vcf, "w", encoding="utf-8") as out:
        for h in kept_header:
            if h.startswith("#CHROM"):
                continue
            out.write(h + "\n")

        out.write('##META=<ID=UNCONVERTED,Description="Records that remained in node/offset coordinates. '
                  'Trailing NODE/OFFSET preserve originals.">\n')
        for c in node_contigs:
            out.write(f"##contig=<ID={c}>\n")

        out.write("#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tNODE\tOFFSET\n")

        with tqdm(total=len(unconverted), desc="Write nodes", unit="rec", dynamic_ncols=True, leave=True) as wbar:
            for chrom, pos, vid, ref, alt, qual, filt, info, node, off in unconverted:
                out.write(f"{chrom}\t{pos}\t{vid}\t{ref}\t{alt}\t{qual}\t{filt}\t{info}\t{node}\t{off}\n")
                wbar.update(1)

    print(f"Done. Linear: {os.path.abspath(out_linear_vcf)} (records={len(converted)}), "
          f"Unconverted: {os.path.abspath(out_nodes_vcf)} (records={len(unconverted)})")


# ------------------------ CLI ------------------------ #

def main():
    ap = argparse.ArgumentParser(description="Convert node/offset VCF to linear-reference coordinates; split into linear + node outputs.")
    ap.add_argument("--in_vcf",        required=True, help="Input VCF (CHROM=node_id, POS=offset).")
    ap.add_argument("--map_json",      required=True, help="Node map JSON (array) or JSONL.")
    ap.add_argument("--map_jsonl",     action="store_true", help="Interpret --map_json as JSONL (one JSON object per line).")
    ap.add_argument("--tsv",           default=None, help="Optional TSV mapping ALT_NODE(S) -> REF_NODE.")
    ap.add_argument("--out_linear_vcf",required=True, help="Output VCF with linear CHROM/POS and trailing NODE/OFFSET.")
    ap.add_argument("--out_nodes_vcf", required=True, help="Output VCF for records that stayed in node form.")
    ap.add_argument("--offset_is_0_based", type=lambda s: str(s).lower() in ("1","true","t","yes","y"), default=True,
                    help="Offsets in input VCF are 0-based (default: true). Set false if they are 1-based.")
    ap.add_argument("--node_start_is_1_based", type=lambda s: str(s).lower() in ("1","true","t","yes","y"), default=True,
                    help="Node starts in JSON are 1-based (default: true).")
    ap.add_argument("--sort",          action="store_true", help="Sort the linear output by CHROM and POS.")
    args = ap.parse_args()

    print("=== Convert VCF Configuration ===")
    print(f"in_vcf                : {os.path.abspath(args.in_vcf)}")
    print(f"map_json              : {os.path.abspath(args.map_json)} (jsonl={bool(args.map_jsonl)})")
    print(f"tsv (ALT->REF mapping): {os.path.abspath(args.tsv) if args.tsv else '(none)'}")
    print(f"out_linear_vcf        : {os.path.abspath(args.out_linear_vcf)}")
    print(f"out_nodes_vcf         : {os.path.abspath(args.out_nodes_vcf)}")
    print(f"offset_is_0_based     : {args.offset_is_0_based}")
    print(f"node_start_is_1_based : {args.node_start_is_1_based}")
    print(f"sort                  : {args.sort}")
    print("=" * 70)

    # Load inputs
    alt_to_ref = load_alt_to_ref_tsv(args.tsv) if args.tsv else {}
    if alt_to_ref:
        print(f"ALT->REF mappings loaded: {len(alt_to_ref)}")

    anchors, nodes_seen = load_node_map(
        args.map_json,
        jsonl=args.map_jsonl,
        node_start_is_1_based=args.node_start_is_1_based
    )
    print(f"Node map: anchors={len(anchors)} (with chrom+start+strand+length), nodes_seen={len(nodes_seen)}")

    # Convert
    convert_vcf(
        in_vcf=args.in_vcf,
        out_linear_vcf=args.out_linear_vcf,
        out_nodes_vcf=args.out_nodes_vcf,
        anchors=anchors,
        nodes_seen_in_json=nodes_seen,
        alt_to_ref=alt_to_ref,
        offset_is_0_based=args.offset_is_0_based,
        sort_records=args.sort
    )


if __name__ == "__main__":
    main()
