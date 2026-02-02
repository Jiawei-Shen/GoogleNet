#!/usr/bin/env python3
"""
align_INDEL_VCF.py

Realign / normalize Pansoma-style INDEL VCF records against a linear FASTA reference.

Fixes:
  1) Deletions encoded as TYPE=D with ALT="*"
     - If len(REF)==1 and consecutive positions: merge into ONE anchored deletion
     - If len(REF)>1: treat as a deletion segment missing the anchor base; anchor directly
     Anchoring uses FASTA:
        NEW_POS = seg_start - 1
        NEW_REF = FASTA[NEW_POS] + FASTA[seg_start..seg_end]
        NEW_ALT = FASTA[NEW_POS]

  2) Insertions encoded as TYPE=I where ALT is missing the anchor prefix:
        If REF matches FASTA at POS and ALT does not start with REF:
            NEW_REF = REF
            NEW_ALT = REF + ALT

  3) INFO/AD arity mismatch (your file stores single-value AD like AD=59).
     Many VCF headers define AD as Number=R (vector), which breaks pysam when writing new records.
     This script preserves your single-value allele depth by moving:
         INFO/AD  -> INFO/ADALT
     and dropping INFO/AD from output records (only when it's single-valued).

Outputs:
  - bgzipped VCF (.vcf.gz)
  - tabix index (.tbi) unless --no-index
  - TSV log of fixes (optional)

Usage:
  python scripts/align_INDEL_VCF.py \
    --vcf  in.vcf.gz \
    --fasta ref.fa \
    --out  out.realigned.vcf.gz \
    --log  out.fix_log.tsv
"""

import argparse
import sys
from typing import Dict, List, Optional, Tuple, Any

import pysam


# -------------------------- FASTA / contig helpers ---------------------------

def build_contig_mapper(vcf_contigs: List[str], fasta_contigs: List[str]) -> Dict[str, str]:
    """Map VCF contig names to FASTA contig names using exact and chr/non-chr conversions."""
    fasta_set = set(fasta_contigs)
    mapping: Dict[str, str] = {}
    for c in vcf_contigs:
        if c in fasta_set:
            mapping[c] = c
        elif c.startswith("chr") and c[3:] in fasta_set:
            mapping[c] = c[3:]
        elif ("chr" + c) in fasta_set:
            mapping[c] = "chr" + c
        else:
            mapping[c] = c  # may fail later if not present in FASTA
    return mapping


def fetch_ref(fa: pysam.FastaFile, contig: str, pos_1based: int, length: int) -> str:
    """Fetch reference substring from FASTA (1-based position)."""
    if pos_1based < 1:
        raise ValueError(f"Invalid 1-based position: {pos_1based}")
    start0 = pos_1based - 1
    end0 = start0 + length
    return fa.fetch(contig, start0, end0).upper()


# -------------------------- Header / INFO sanitizing --------------------------

def make_fixed_header(in_header: pysam.VariantHeader) -> pysam.VariantHeader:
    out = in_header.copy()

    # Track modifications
    if "FIXED" not in out.info:
        out.add_line(
            '##INFO=<ID=FIXED,Number=0,Type=Flag,Description="Record fixed/realigned against FASTA by align_INDEL_VCF.py">'
        )
    if "FIXNOTE" not in out.info:
        out.add_line(
            '##INFO=<ID=FIXNOTE,Number=1,Type=String,Description="How the record was fixed">'
        )

    # Preserve single-value allele depth without violating AD Number=R semantics.
    if "ADALT" not in out.info:
        out.add_line(
            '##INFO=<ID=ADALT,Number=1,Type=Integer,Description="Alt allele depth (copied from original INFO/AD when AD was single-valued)">'
        )

    return out


def _as_list_of_ints(x: Any) -> List[int]:
    """Normalize an INFO value to a list of ints when possible."""
    if x is None:
        return []
    if isinstance(x, (tuple, list)):
        out = []
        for v in x:
            try:
                out.append(int(v))
            except Exception:
                continue
        return out
    if isinstance(x, (int, float)):
        return [int(x)]
    # last resort: parse as int
    try:
        return [int(str(x))]
    except Exception:
        return []


def sanitize_info(rec: pysam.VariantRecord) -> Dict[str, Any]:
    """
    Make INFO safe for writing with pysam:
      - Copy fields
      - If AD exists and is a SINGLE value: move to ADALT and drop AD
        (prevents pysam error when header expects Number=R)
    """
    info_out: Dict[str, Any] = {}
    for k in rec.info.keys():
        try:
            info_out[k] = rec.info[k]
        except Exception:
            # If pysam can't materialize a value, skip it
            continue

    if "AD" in info_out:
        ad_list = _as_list_of_ints(info_out["AD"])
        if len(ad_list) == 1:
            info_out["ADALT"] = int(ad_list[0])
            info_out.pop("AD", None)

    return info_out


# -------------------------- Record creation helpers --------------------------

def new_record_from_rec(
    outvcf: pysam.VariantFile,
    rec: pysam.VariantRecord,
    alleles: Optional[Tuple[str, ...]] = None,
    start0: Optional[int] = None,
    stop0: Optional[int] = None,
    add_fixnote: Optional[str] = None,
) -> pysam.VariantRecord:
    """
    Create a record in outvcf with outvcf's header and copy over fields from rec.
    Optionally override alleles/start/stop, and add FIXED/FIXNOTE.
    """
    if alleles is None:
        alleles = rec.alleles

    if start0 is None:
        start0 = rec.start
    if stop0 is None:
        stop0 = rec.stop

    info_dict = sanitize_info(rec)

    new = outvcf.new_record(
        contig=rec.contig,
        start=int(start0),
        stop=int(stop0),
        id=rec.id,
        qual=rec.qual,
        alleles=alleles,
        filter=list(rec.filter.keys()),
        info=info_dict,
    )
    for s in rec.samples:
        new.samples[s].update(rec.samples[s])

    if add_fixnote is not None:
        new.info["FIXED"] = True
        new.info["FIXNOTE"] = add_fixnote

    return new


# -------------------------- Deletion anchoring logic --------------------------

def emit_star_deletion_segment(
    outvcf: pysam.VariantFile,
    fa: pysam.FastaFile,
    chrom: str,
    fasta_chrom: str,
    rec: pysam.VariantRecord,
    seg_start: int,   # 1-based
    seg_end: int,     # 1-based inclusive
    logf,
    note: str,
) -> int:
    """
    Emit ONE anchored deletion for reference bases seg_start..seg_end (1-based inclusive).
    Uses anchor at seg_start-1.
    """
    anchor_pos = seg_start - 1
    if anchor_pos < 1:
        logf.write(f"{chrom}\t{rec.pos}\t{rec.ref}\t*\t.\t.\t.\tSKIP_DEL_NO_ANCHOR\tseg={seg_start}-{seg_end}\n")
        return 0

    deleted_len = seg_end - seg_start + 1
    anchor_base = fetch_ref(fa, fasta_chrom, anchor_pos, 1)
    deleted_seq = fetch_ref(fa, fasta_chrom, seg_start, deleted_len)

    new_pos = anchor_pos  # 1-based
    new_ref = (anchor_base + deleted_seq).upper()
    new_alt = anchor_base.upper()

    # start/stop for pysam are 0-based half-open
    start0 = new_pos - 1
    stop0 = start0 + len(new_ref)

    new = new_record_from_rec(
        outvcf,
        rec,
        alleles=(new_ref, new_alt),
        start0=start0,
        stop0=stop0,
        add_fixnote=note,
    )
    outvcf.write(new)

    logf.write(
        f"{chrom}\t{rec.pos}\t{rec.ref}\t*\t{new_pos}\t{new_ref}\t{new_alt}\tFIX_DEL\t{note}\n"
    )
    return 1


# -------------------------- Main ---------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Realign/anchor Pansoma-style INDEL VCF against a reference FASTA.")
    ap.add_argument("--vcf", required=True, help="Input VCF or VCF.GZ")
    ap.add_argument("--fasta", required=True, help="Reference FASTA (indexed with .fai)")
    ap.add_argument("--out", required=True, help="Output VCF.GZ")
    ap.add_argument("--log", default=None, help="TSV log (default: <out>.fix_log.tsv)")
    ap.add_argument("--no-index", action="store_true", help="Do not create .tbi index for output")
    args = ap.parse_args()

    log_path = args.log if args.log else (args.out + ".fix_log.tsv")

    invcf = pysam.VariantFile(args.vcf)
    fa = pysam.FastaFile(args.fasta)

    contig_map = build_contig_mapper(list(invcf.header.contigs), list(fa.references))

    out_header = make_fixed_header(invcf.header)
    outvcf = pysam.VariantFile(args.out, "wz", header=out_header)

    # State for merging consecutive per-base star deletions (len(REF)==1)
    star_run_chrom: Optional[str] = None
    star_run_fasta_chrom: Optional[str] = None
    star_run_start: Optional[int] = None
    star_run_end: Optional[int] = None
    star_run_first_rec: Optional[pysam.VariantRecord] = None

    fixed = 0
    written = 0

    def flush_star_run(logf):
        nonlocal star_run_chrom, star_run_fasta_chrom, star_run_start, star_run_end, star_run_first_rec, fixed
        if star_run_first_rec is None:
            return
        fixed += emit_star_deletion_segment(
            outvcf=outvcf,
            fa=fa,
            chrom=star_run_chrom,
            fasta_chrom=star_run_fasta_chrom,
            rec=star_run_first_rec,
            seg_start=int(star_run_start),
            seg_end=int(star_run_end),
            logf=logf,
            note=f"merged_star_del:{star_run_start}-{star_run_end}",
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
                raise SystemExit(f"ERROR: contig '{chrom}' not found in FASTA (tried '{fasta_chrom}').")

            vtype = str(rec.info.get("TYPE", ""))  # expected I / D in your file
            alts = rec.alts or ()

            # ---------------- Deletion: TYPE=D ALT=* ----------------
            is_star_del = (vtype == "D" and len(alts) == 1 and alts[0] == "*")
            if is_star_del:
                # Case (B): len(REF)>1 -> anchor this deletion segment directly (missing left base)
                if len(rec.ref) > 1:
                    flush_star_run(logf)
                    seg_start = rec.pos
                    seg_end = rec.pos + len(rec.ref) - 1
                    fixed += emit_star_deletion_segment(
                        outvcf=outvcf,
                        fa=fa,
                        chrom=chrom,
                        fasta_chrom=fasta_chrom,
                        rec=rec,
                        seg_start=seg_start,
                        seg_end=seg_end,
                        logf=logf,
                        note=f"anchor_star_del_segment:{seg_start}-{seg_end}",
                    )
                    written += 1
                    continue

                # Case (A): len(REF)==1 -> run-merge consecutive positions
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
                        flush_star_run(logf)
                        star_run_chrom = chrom
                        star_run_fasta_chrom = fasta_chrom
                        star_run_start = rec.pos
                        star_run_end = rec.pos
                        star_run_first_rec = rec
                continue

            # Not a star deletion -> flush pending run first
            flush_star_run(logf)

            # ---------------- Insertion: TYPE=I fix missing prefix ----------------
            if vtype == "I" and len(alts) == 1:
                alt0 = alts[0].upper()
                ref0 = rec.ref.upper()

                # Only do the safe fix if REF matches FASTA at POS
                try:
                    ref_from_fa = fetch_ref(fa, fasta_chrom, rec.pos, len(ref0))
                except Exception as e:
                    ref_from_fa = None
                    logf.write(f"{chrom}\t{rec.pos}\t{rec.ref}\t{alts[0]}\t.\t.\t.\tWARN\tfasta_fetch_failed:{e}\n")

                if ref_from_fa is not None and ref0 == ref_from_fa and (not alt0.startswith(ref0)):
                    new_ref = ref0
                    new_alt = (ref0 + alt0).upper()

                    new = new_record_from_rec(
                        outvcf,
                        rec,
                        alleles=(new_ref, new_alt),
                        start0=rec.start,
                        stop0=rec.start + len(new_ref),
                        add_fixnote="ins_add_missing_prefix",
                    )
                    outvcf.write(new)
                    fixed += 1
                    written += 1

                    logf.write(
                        f"{chrom}\t{rec.pos}\t{rec.ref}\t{alts[0]}\t{rec.pos}\t{new_ref}\t{new_alt}\tFIX_INS\tadd_prefix\n"
                    )
                    continue

            # ---------------- Otherwise: write record (sanitizing INFO/AD) ----------------
            outvcf.write(new_record_from_rec(outvcf, rec))
            written += 1

        # end for
        flush_star_run(logf)

    outvcf.close()
    invcf.close()
    fa.close()

    if not args.no_index:
        pysam.tabix_index(args.out, preset="vcf", force=True)

    print(f"[done] wrote: {args.out}", file=sys.stderr)
    print(f"[done] log:   {log_path}", file=sys.stderr)
    if not args.no_index:
        print(f"[done] index: {args.out}.tbi", file=sys.stderr)


if __name__ == "__main__":
    main()
