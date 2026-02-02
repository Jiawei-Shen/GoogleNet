#!/usr/bin/env python3
"""
fix_indels_with_fasta.py

Fix/normalize a VCF against a reference FASTA, focusing on:
  - Deletions encoded as TYPE=D with ALT="*" (spanning deletion markers):
      * If len(REF)==1 and positions are consecutive -> merge into one anchored deletion
      * If len(REF)>1 -> treat as a deletion segment missing anchor, anchor directly
  - Insertions encoded as TYPE=I but missing the anchor prefix:
      * If REF matches FASTA at POS and ALT does not start with REF -> prefix REF to ALT

Output:
  - bgzipped VCF (.vcf.gz) + tabix index
  - TSV log of changes

Usage:
  python fix_indels_with_fasta.py \
    --vcf pansoma-to_INDEL_COLO829_WGS_P100.linear.vcf.gz \
    --fasta GCA_000001405.15_GRCh38_no_alt_analysis_set.fasta \
    --out pansoma-to_INDEL_COLO829_WGS_P100.linear.fixed.vcf.gz \
    --log fix_log.tsv
"""

import argparse
import sys
from typing import Dict, List, Optional

import pysam


def build_contig_mapper(vcf_contigs: List[str], fasta_contigs: List[str]) -> Dict[str, str]:
    """Map VCF contigs to FASTA contigs, trying chr/non-chr variants."""
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
            m[c] = c  # may fail later if not in fasta
    return m


def fetch_ref(fa: pysam.FastaFile, contig: str, pos_1based: int, length: int) -> str:
    """Fetch reference substring from FASTA using 1-based pos."""
    if pos_1based < 1:
        raise ValueError(f"Invalid position: {pos_1based}")
    start0 = pos_1based - 1
    end0 = start0 + length
    return fa.fetch(contig, start0, end0).upper()


def make_fixed_header(in_header: pysam.VariantHeader) -> pysam.VariantHeader:
    out = in_header.copy()
    if "FIXED" not in out.info:
        out.add_line(
            '##INFO=<ID=FIXED,Number=0,Type=Flag,Description="Record fixed/normalized against FASTA by fix_indels_with_fasta.py">'
        )
    if "FIXNOTE" not in out.info:
        out.add_line(
            '##INFO=<ID=FIXNOTE,Number=1,Type=String,Description="How the record was fixed">'
        )
    return out


def copy_record(out_vcf: pysam.VariantFile, rec: pysam.VariantRecord) -> pysam.VariantRecord:
    """Create a new record in out_vcf header and copy fields from rec."""
    new = out_vcf.new_record(
        contig=rec.contig,
        start=rec.start,
        stop=rec.stop,
        id=rec.id,
        qual=rec.qual,
        alleles=rec.alleles,
        filter=list(rec.filter.keys()),
        info=dict(rec.info),
    )
    for s in rec.samples:
        new.samples[s].update(rec.samples[s])
    return new


# --- Your deletion anchoring helper (unchanged idea) ---

def emit_star_deletion_segment(out_vcf, fa, chrom, fasta_chrom, rec, seg_start, seg_end, note, logf):
    """
    Emit ONE anchored deletion record that deletes reference bases from seg_start..seg_end (1-based, inclusive).
    Anchors at seg_start-1 using FASTA.
    """
    anchor_pos = seg_start - 1
    if anchor_pos < 1:
        logf.write(f"{chrom}\t{rec.pos}\t{rec.ref}\t*\t.\t.\t.\tSKIP_DEL_NO_ANCHOR\tseg={seg_start}-{seg_end}\n")
        return 0

    deleted_len = seg_end - seg_start + 1
    anchor_base = fetch_ref(fa, fasta_chrom, anchor_pos, 1)
    deleted_seq = fetch_ref(fa, fasta_chrom, seg_start, deleted_len)

    new_pos = anchor_pos
    new_ref = (anchor_base + deleted_seq).upper()
    new_alt = anchor_base.upper()

    new = out_vcf.new_record(
        contig=rec.contig,
        start=new_pos - 1,
        stop=(new_pos - 1) + len(new_ref),
        id=rec.id,
        qual=rec.qual,
        alleles=(new_ref, new_alt),
        filter=list(rec.filter.keys()),
        info=dict(rec.info),
    )
    for s in rec.samples:
        new.samples[s].update(rec.samples[s])

    new.info["FIXED"] = True
    new.info["FIXNOTE"] = note

    out_vcf.write(new)
    logf.write(
        f"{chrom}\t{rec.pos}\t{rec.ref}\t*\t{new_pos}\t{new_ref}\t{new_alt}\tFIX_DEL\t{note}\n"
    )
    return 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vcf", required=True, help="Input VCF/VCF.GZ")
    ap.add_argument("--fasta", required=True, help="Reference FASTA (needs .fai)")
    ap.add_argument("--out", required=True, help="Output VCF.GZ")
    ap.add_argument("--log", default=None, help="TSV log (default: <out>.fix_log.tsv)")
    args = ap.parse_args()

    log_path = args.log if args.log else (args.out + ".fix_log.tsv")

    invcf = pysam.VariantFile(args.vcf)
    fa = pysam.FastaFile(args.fasta)

    contig_map = build_contig_mapper(list(invcf.header.contigs), list(fa.references))

    out_header = make_fixed_header(invcf.header)
    outvcf = pysam.VariantFile(args.out, "wz", header=out_header)

    # State for merging consecutive per-base star deletions
    star_run_chrom: Optional[str] = None
    star_run_fasta_chrom: Optional[str] = None
    star_run_start: Optional[int] = None
    star_run_end: Optional[int] = None
    star_run_first_rec: Optional[pysam.VariantRecord] = None

    fixed = 0
    passed = 0

    def flush_star_run():
        nonlocal star_run_chrom, star_run_fasta_chrom, star_run_start, star_run_end, star_run_first_rec, fixed
        if star_run_first_rec is None:
            return
        fixed += emit_star_deletion_segment(
            outvcf, fa,
            star_run_chrom, star_run_fasta_chrom,
            star_run_first_rec,
            star_run_start, star_run_end,
            note=f"merged_star_del:{star_run_start}-{star_run_end}",
            logf=logf
        )
        star_run_chrom = None
        star_run_fasta_chrom = None
        star_run_start = None
        star_run_end = None
        star_run_first_rec = None

    with open(log_path, "w", encoding="utf-8") as logf:
        logf.write("#chrom\told_pos\told_ref\told_alt\tnew_pos\tnew_ref\tnew_alt\taction\tdetails\n")

        for rec in invcf.fetch():
            chrom = rec.contig
            fasta_chrom = contig_map.get(chrom, chrom)
            if fasta_chrom not in fa.references:
                raise SystemExit(f"ERROR: contig {chrom} not found in FASTA (tried {fasta_chrom}).")

            vtype = str(rec.info.get("TYPE", ""))  # your VCF has TYPE=I / TYPE=D
            alts = rec.alts or ()

            # --- handle star deletions ---
            is_star_del = (vtype == "D" and len(alts) == 1 and alts[0] == "*")
            if is_star_del:
                # (B) len(REF)>1 => anchor this deletion segment directly
                if len(rec.ref) > 1:
                    flush_star_run()
                    seg_start = rec.pos
                    seg_end = rec.pos + len(rec.ref) - 1
                    fixed += emit_star_deletion_segment(
                        outvcf, fa, chrom, fasta_chrom, rec,
                        seg_start, seg_end,
                        note=f"anchor_star_del_segment:{seg_start}-{seg_end}",
                        logf=logf
                    )
                    continue

                # (A) len(REF)==1 => merge consecutive positions
                if star_run_first_rec is None:
                    star_run_chrom = chrom
                    star_run_fasta_chrom = fasta_chrom
                    star_run_start = rec.pos
                    star_run_end = rec.pos
                    star_run_first_rec = rec
                else:
                    if chrom == star_run_chrom and rec.pos == (star_run_end + 1):
                        star_run_end = rec.pos
                    else:
                        flush_star_run()
                        star_run_chrom = chrom
                        star_run_fasta_chrom = fasta_chrom
                        star_run_start = rec.pos
                        star_run_end = rec.pos
                        star_run_first_rec = rec
                continue

            # Not a star deletion -> flush pending run before other processing
            flush_star_run()

            # --- insertion missing prefix base (simple + safe) ---
            if vtype == "I" and len(alts) == 1:
                alt0 = alts[0].upper()
                ref0 = rec.ref.upper()

                # Ensure ref0 matches FASTA at POS for length(ref0)
                ref_from_fa = fetch_ref(fa, fasta_chrom, rec.pos, len(ref0))

                if ref0 == ref_from_fa and not alt0.startswith(ref0):
                    # fix: prefix the anchor base(s)
                    new_ref = ref0
                    new_alt = (ref0 + alt0).upper()

                    new = outvcf.new_record(
                        contig=rec.contig,
                        start=rec.start,
                        stop=rec.start + len(new_ref),
                        id=rec.id,
                        qual=rec.qual,
                        alleles=(new_ref, new_alt),
                        filter=list(rec.filter.keys()),
                        info=dict(rec.info),
                    )
                    for s in rec.samples:
                        new.samples[s].update(rec.samples[s])

                    new.info["FIXED"] = True
                    new.info["FIXNOTE"] = "ins_add_missing_prefix"

                    outvcf.write(new)
                    logf.write(f"{chrom}\t{rec.pos}\t{rec.ref}\t{alts[0]}\t{rec.pos}\t{new_ref}\t{new_alt}\tFIX_INS\tadd_prefix\n")
                    fixed += 1
                    continue

            # Otherwise: write as-is
            outvcf.write(copy_record(outvcf, rec))
            passed += 1

        # end for
        flush_star_run()

    outvcf.close()
    invcf.close()
    fa.close()

    # index output
    pysam.tabix_index(args.out, preset="vcf", force=True)

    print(f"[done] out: {args.out}", file=sys.stderr)
    print(f"[done] log: {log_path}", file=sys.stderr)
    print(f"[done] idx: {args.out}.tbi", file=sys.stderr)


if __name__ == "__main__":
    main()
