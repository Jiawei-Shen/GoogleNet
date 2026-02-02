#!/usr/bin/env python3
"""
align_INDEL_VCF.py

Strictly anchor Pansoma-style INDEL VCF against a linear FASTA.

Requested behavior:
  1) TYPE=I: ALT must start with REF -> force ALT = REF + ALT if missing.
  2) TYPE=D: add exactly ONE left-flanking base to BOTH REF and ALT:
       NEW_POS = POS-1
       NEW_REF = left_base + deleted_seq
       NEW_ALT = left_base

Notes:
  - For deletions, deleted_seq is taken from FASTA at POS with length = len(original REF).
    (This is the safest interpretation if the VCF lost characters.)
  - Drops records with REF==ALT or missing ALT.
  - INFO/AD is single-valued in your VCF; pysam often expects Number=R. We preserve it as ADALT.

Output:
  - .vcf.gz + .tbi (unless --no-index)
  - TSV log

Example:
  python scripts/align_INDEL_VCF.py \
    --vcf  input.vcf.gz \
    --fasta GRCh38_no_alt_analysis_set.fasta \
    --out  output.realigned.vcf.gz
"""

import argparse
import sys
from typing import Dict, List, Any

import pysam


def build_contig_mapper(vcf_contigs: List[str], fasta_contigs: List[str]) -> Dict[str, str]:
    fasta_set = set(fasta_contigs)
    m: Dict[str, str] = {}
    for c in vcf_contigs:
        if c in fasta_set:
            m[c] = c
        elif c.startswith("chr") and c[3:] in fasta_set:
            m[c] = c[3:]
        elif ("chr" + c) in fasta_set:
            m[c] = "chr" + c
        else:
            m[c] = c
    return m


def fetch_ref(fa: pysam.FastaFile, contig: str, pos_1based: int, length: int) -> str:
    if pos_1based < 1:
        raise ValueError(f"Invalid position {pos_1based}")
    start0 = pos_1based - 1
    return fa.fetch(contig, start0, start0 + length).upper()


def make_header(in_header: pysam.VariantHeader) -> pysam.VariantHeader:
    out = in_header.copy()
    if "FIXED" not in out.info:
        out.add_line('##INFO=<ID=FIXED,Number=0,Type=Flag,Description="Record fixed against FASTA by align_INDEL_VCF.py">')
    if "FIXNOTE" not in out.info:
        out.add_line('##INFO=<ID=FIXNOTE,Number=1,Type=String,Description="Fix note">')
    if "ADALT" not in out.info:
        out.add_line('##INFO=<ID=ADALT,Number=1,Type=Integer,Description="Alt allele depth (from original INFO/AD when AD was single-valued)">')
    return out


def _as_list_of_ints(x: Any) -> List[int]:
    if x is None:
        return []
    if isinstance(x, (tuple, list)):
        out = []
        for v in x:
            try:
                out.append(int(v))
            except Exception:
                pass
        return out
    if isinstance(x, (int, float)):
        return [int(x)]
    try:
        return [int(str(x))]
    except Exception:
        return []


def sanitize_info(rec: pysam.VariantRecord) -> Dict[str, Any]:
    info = {}
    for k in rec.info.keys():
        try:
            info[k] = rec.info[k]
        except Exception:
            continue

    if "AD" in info:
        ad_list = _as_list_of_ints(info["AD"])
        if len(ad_list) == 1:
            info["ADALT"] = int(ad_list[0])
            info.pop("AD", None)

    return info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vcf", required=True)
    ap.add_argument("--fasta", required=True)
    ap.add_argument("--out", required=True)          # .vcf.gz
    ap.add_argument("--log", default=None)           # default: out + .fix_log.tsv
    ap.add_argument("--no-index", action="store_true")
    ap.add_argument("--force-ref-from-fasta", action="store_true",
                    help="If set, overwrite REF with FASTA at POS (len=original REF). Recommended.")
    args = ap.parse_args()

    log_path = args.log if args.log else (args.out + ".fix_log.tsv")

    invcf = pysam.VariantFile(args.vcf)
    fa = pysam.FastaFile(args.fasta)

    contig_map = build_contig_mapper(list(invcf.header.contigs), list(fa.references))
    outvcf = pysam.VariantFile(args.out, "wz", header=make_header(invcf.header))

    fixed = 0
    kept = 0
    dropped = 0

    with open(log_path, "w", encoding="utf-8") as logf:
        logf.write("#chrom\told_pos\told_ref\told_alt\tnew_pos\tnew_ref\tnew_alt\taction\tnote\n")

        for rec in invcf.fetch():
            chrom = rec.contig
            fasta_chrom = contig_map.get(chrom, chrom)
            if fasta_chrom not in fa.references:
                raise SystemExit(f"ERROR: contig '{chrom}' not in FASTA (tried '{fasta_chrom}')")

            if rec.alts is None or len(rec.alts) != 1:
                dropped += 1
                continue

            alt = rec.alts[0]
            ref = rec.ref

            # drop non-variant (your chr1:9575235 A A)
            if ref.upper() == alt.upper():
                dropped += 1
                logf.write(f"{chrom}\t{rec.pos}\t{ref}\t{alt}\t.\t.\t.\tDROP\tREF_eq_ALT\n")
                continue

            vtype = str(rec.info.get("TYPE", "")).upper()

            info = sanitize_info(rec)

            # ---------- TYPE=I (insertion): ALT must start with REF ----------
            if vtype == "I":
                # Optionally force REF from FASTA (safest)
                if args.force_ref_from_fasta:
                    ref_from_fa = fetch_ref(fa, fasta_chrom, rec.pos, len(ref))
                    ref = ref_from_fa

                new_ref = ref.upper()
                new_alt = alt.upper()
                if not new_alt.startswith(new_ref):
                    new_alt = new_ref + new_alt
                    note = "ins_prefix_ref_to_alt"
                    fixed_flag = True
                else:
                    note = "ins_ok"
                    fixed_flag = False

                new = outvcf.new_record(
                    contig=rec.contig,
                    start=rec.start,
                    stop=rec.start + len(new_ref),
                    id=rec.id,
                    qual=rec.qual,
                    alleles=(new_ref, new_alt),
                    filter=list(rec.filter.keys()),
                    info=info,
                )
                for s in rec.samples:
                    new.samples[s].update(rec.samples[s])

                if fixed_flag:
                    new.info["FIXED"] = True
                    new.info["FIXNOTE"] = note
                    fixed += 1
                    logf.write(f"{chrom}\t{rec.pos}\t{rec.ref}\t{alt}\t{rec.pos}\t{new_ref}\t{new_alt}\tFIX_INS\t{note}\n")
                kept += 1
                outvcf.write(new)
                continue

            # ---------- TYPE=D (deletion): add ONE left-flanking base to REF and ALT ----------
            if vtype == "D":
                # Your deletions are often ALT="*" or other placeholder; we ignore ALT and rebuild.
                pos = rec.pos
                if pos <= 1:
                    dropped += 1
                    logf.write(f"{chrom}\t{pos}\t{rec.ref}\t{alt}\t.\t.\t.\tDROP\tDEL_no_left_anchor\n")
                    continue

                left_base = fetch_ref(fa, fasta_chrom, pos - 1, 1)
                del_len = len(rec.ref)
                deleted_seq = fetch_ref(fa, fasta_chrom, pos, del_len)

                new_pos = pos - 1
                new_ref = (left_base + deleted_seq).upper()
                new_alt = left_base.upper()

                start0 = new_pos - 1
                stop0 = start0 + len(new_ref)

                new = outvcf.new_record(
                    contig=rec.contig,
                    start=start0,
                    stop=stop0,
                    id=rec.id,
                    qual=rec.qual,
                    alleles=(new_ref, new_alt),
                    filter=list(rec.filter.keys()),
                    info=info,
                )
                for s in rec.samples:
                    new.samples[s].update(rec.samples[s])

                new.info["FIXED"] = True
                new.info["FIXNOTE"] = "del_add_left_base"

                fixed += 1
                kept += 1
                outvcf.write(new)

                logf.write(f"{chrom}\t{pos}\t{rec.ref}\t{alt}\t{new_pos}\t{new_ref}\t{new_alt}\tFIX_DEL\tdel_add_left_base\n")
                continue

            # ---------- other types: pass through (sanitized INFO) ----------
            new = outvcf.new_record(
                contig=rec.contig,
                start=rec.start,
                stop=rec.stop,
                id=rec.id,
                qual=rec.qual,
                alleles=rec.alleles,
                filter=list(rec.filter.keys()),
                info=info,
            )
            for s in rec.samples:
                new.samples[s].update(rec.samples[s])
            outvcf.write(new)
            kept += 1

    outvcf.close()
    invcf.close()
    fa.close()

    if not args.no_index:
        pysam.tabix_index(args.out, preset="vcf", force=True)

    print(f"[done] out={args.out}", file=sys.stderr)
    print(f"[done] log={log_path}", file=sys.stderr)
    print(f"[stats] kept={kept} fixed={fixed} dropped={dropped}", file=sys.stderr)
    if not args.no_index:
        print(f"[done] index={args.out}.tbi", file=sys.stderr)


if __name__ == "__main__":
    main()
