#!/usr/bin/env python3
"""
Extract QSM ROI values — v2 (Curated SynthSeg labels)
======================================================
Uses ONLY the SynthSeg labels specified in "SynthSeg Labels.rtf":
  - Whole cerebral cortex (3, 42) instead of DK parcellation
  - Excludes Accumbens (26, 58)
  - Keeps all other subcortical + ventricular + midline labels
Manual segmentation (Seg_DN_Basal_Ganglia) is unchanged.

Output CSVs:
  output/roi_values_v2/synthseg_roi_values.csv
  output/roi_values_v2/manual_seg_roi_values.csv

Usage:
  python code/extract_qsm_roi_values_v2.py
"""

import os, csv, logging
from pathlib import Path

import numpy as np
import nibabel as nib

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# ── Paths ──────────────────────────────────────────────────────────────────
PROJECT    = Path(".")
NIFTI_ROOT = Path("/path/to/project/"
                   "Reject and Resubmit/Protocal/NIfTI")
REG_DIR    = PROJECT / "output" / "roi_analysis" / "registered"
OUT_DIR    = PROJECT / "output" / "roi_values_v2"

# ── All 28 subject/session pairs ──────────────────────────────────────────
SESSIONS = [
    ("sub-01Yuan",  "ses-5TUIH1"),
    ("sub-02Wang",  "ses-5TShenzhen"),
    ("sub-02Wang",  "ses-5TUIH1"),
    ("sub-02Wang",  "ses-5TUIH2"),
    ("sub-03Xiang", "ses-3TUIH1"),
    ("sub-03Xiang", "ses-3TUIH2"),
    ("sub-03Xiang", "ses-5TChongqing"),
    ("sub-03Xiang", "ses-5TShanghai"),
    ("sub-03Xiang", "ses-5TUIH1"),
    ("sub-03Xiang", "ses-5TUIH2"),
    ("sub-04Qin",   "ses-3TUIH1"),
    ("sub-04Qin",   "ses-5TShanghai"),
    ("sub-04Qin",   "ses-5TUIH1"),
    ("sub-04Qin",   "ses-5TUIH2"),
    ("sub-07Huang",  "ses-3TUIH1"),
    ("sub-07Huang",  "ses-5TJingzhou"),
    ("sub-07Huang",  "ses-5TShanghai"),
    ("sub-07Huang",  "ses-5TUIH1"),
    ("sub-07Huang",  "ses-5TUIH2"),
    ("sub-08Lin",   "ses-3TUIH1"),
    ("sub-08Lin",   "ses-5TShanghai"),
    ("sub-08Lin",   "ses-5TShenzhen"),
    ("sub-08Lin",   "ses-5TUIH1"),
    ("sub-08Lin",   "ses-5TUIH2"),
    ("sub-09Song",  "ses-3TUIH1"),
    ("sub-09Song",  "ses-5TShanghai"),
    ("sub-09Song",  "ses-5TUIH1"),
    ("sub-09Song",  "ses-5TUIH2"),
]

# ── Curated SynthSeg labels (from "SynthSeg Labels.rtf") ─────────────────
# Only the labels specified in the RTF file are included.
# Key changes vs. v1:
#   - ADDED:   3 (L Cerebral-Cortex), 42 (R Cerebral-Cortex)
#   - REMOVED: 26 (L Accumbens), 58 (R Accumbens)
#   - REMOVED: All DK cortical parcellation labels (1001-1035, 2001-2035)
SYNTHSEG_LABELS = {
    # Left hemisphere
    2:  ("Cerebral-White-Matter",   "L"),
    3:  ("Cerebral-Cortex",         "L"),
    4:  ("Lateral-Ventricle",       "L"),
    5:  ("Inf-Lat-Vent",            "L"),
    7:  ("Cerebellum-White-Matter", "L"),
    8:  ("Cerebellum-Cortex",       "L"),
    10: ("Thalamus",                "L"),
    11: ("Caudate",                 "L"),
    12: ("Putamen",                 "L"),
    13: ("Pallidum",                "L"),
    17: ("Hippocampus",             "L"),
    18: ("Amygdala",                "L"),
    28: ("VentralDC",               "L"),
    # Midline
    14: ("3rd-Ventricle",           "M"),
    15: ("4th-Ventricle",           "M"),
    16: ("Brain-Stem",              "M"),
    24: ("CSF",                     "M"),
    # Right hemisphere
    41: ("Cerebral-White-Matter",   "R"),
    42: ("Cerebral-Cortex",         "R"),
    43: ("Lateral-Ventricle",       "R"),
    44: ("Inf-Lat-Vent",            "R"),
    46: ("Cerebellum-White-Matter", "R"),
    47: ("Cerebellum-Cortex",       "R"),
    49: ("Thalamus",                "R"),
    50: ("Caudate",                 "R"),
    51: ("Putamen",                 "R"),
    52: ("Pallidum",                "R"),
    53: ("Hippocampus",             "R"),
    54: ("Amygdala",                "R"),
    60: ("VentralDC",               "R"),
}

# ── Seg_DN_Basal_Ganglia label definitions (unchanged from v1) ────────────
MANUAL_SEG_LABELS = {
    1:  ("Dentate_nucleus",  "R"),
    2:  ("Dentate_nucleus",  "L"),
    3:  ("Substantia_nigra", "R"),
    4:  ("Substantia_nigra", "L"),
    5:  ("Nucleus_ruber",    "R"),
    6:  ("Nucleus_ruber",    "L"),
    7:  ("Thalamus",         "R"),
    8:  ("Thalamus",         "L"),
    9:  ("Caudate_nucleus",  "R"),
    10: ("Caudate_nucleus",  "L"),
    11: ("Putamen",          "R"),
    12: ("Putamen",          "L"),
    13: ("Globus_pallidus",  "R"),
    14: ("Globus_pallidus",  "L"),
}


# ── Statistics ─────────────────────────────────────────────────────────────
def roi_stats(vals):
    """Compute descriptive statistics for an array of QSM values."""
    if len(vals) == 0:
        return None
    return {
        "n_voxels": int(len(vals)),
        "mean":     float(np.mean(vals)),
        "std":      float(np.std(vals, ddof=0)),
        "median":   float(np.median(vals)),
        "q25":      float(np.percentile(vals, 25)),
        "q75":      float(np.percentile(vals, 75)),
    }


def field_strength(ses):
    """Infer field strength from session name."""
    return 3.0 if "3T" in ses else 5.0


# ── Extraction ─────────────────────────────────────────────────────────────
def extract_synthseg(qsm_data, seg_data, sub, ses):
    """Extract QSM values from curated SynthSeg labels only."""
    rows = []
    present_labels = np.unique(seg_data[seg_data > 0]).tolist()
    fs = field_strength(ses)

    for lid in sorted(SYNTHSEG_LABELS.keys()):
        if lid not in present_labels:
            continue
        roi_name, side = SYNTHSEG_LABELS[lid]
        mask = (seg_data == lid)
        vals = qsm_data[mask]
        s = roi_stats(vals)
        if s is None:
            continue
        rows.append({
            "subject":        sub,
            "session":        ses,
            "field_strength": fs,
            "mask":           "synthseg",
            "label_id":       lid,
            "roi":            roi_name,
            "side":           side,
            "n_voxels":       s["n_voxels"],
            "mean_ppb":       f"{s['mean']:.4f}",
            "std_ppb":        f"{s['std']:.4f}",
            "median_ppb":     f"{s['median']:.4f}",
            "q25_ppb":        f"{s['q25']:.4f}",
            "q75_ppb":        f"{s['q75']:.4f}",
        })
    return rows


def extract_manual_seg(qsm_data, seg_data, sub, ses):
    """Extract QSM values from Seg_DN_Basal_Ganglia (unchanged from v1)."""
    rows = []
    fs = field_strength(ses)

    for lid, (roi_name, side) in sorted(MANUAL_SEG_LABELS.items()):
        mask = (seg_data == lid)
        vals = qsm_data[mask]
        s = roi_stats(vals)
        if s is None:
            continue
        rows.append({
            "subject":        sub,
            "session":        ses,
            "field_strength": fs,
            "mask":           "Seg_DN_Basal_Ganglia",
            "label_id":       lid,
            "roi":            roi_name,
            "side":           side,
            "n_voxels":       s["n_voxels"],
            "mean_ppb":       f"{s['mean']:.4f}",
            "std_ppb":        f"{s['std']:.4f}",
            "median_ppb":     f"{s['median']:.4f}",
            "q25_ppb":        f"{s['q25']:.4f}",
            "q75_ppb":        f"{s['q75']:.4f}",
        })
    return rows


# ── Main ───────────────────────────────────────────────────────────────────
def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    all_synthseg = []
    all_manual   = []
    skipped      = []

    log.info("=" * 65)
    log.info("QSM ROI Extraction v2 — Curated SynthSeg + Seg_DN_Basal_Ganglia")
    log.info(f"SynthSeg labels: {len(SYNTHSEG_LABELS)} "
             f"(excl. background, Accumbens, DK parcellation)")
    log.info(f"Sessions: {len(SESSIONS)}")
    log.info("=" * 65)

    for sub, ses in SESSIONS:
        log.info(f"\n── {sub} / {ses} ──")

        # 1. Load QSM
        qsm_path = NIFTI_ROOT / sub / ses / "qsm" / "qsm_10echo_run-01.nii.gz"
        if not qsm_path.exists():
            log.warning(f"  QSM not found: {qsm_path}")
            skipped.append((sub, ses, "QSM missing"))
            continue
        qsm_data = nib.load(str(qsm_path)).get_fdata()
        if qsm_data.ndim > 3:
            qsm_data = qsm_data[..., 0]
        log.info(f"  QSM: {qsm_data.shape}")

        # 2. Load SynthSeg
        ss_path = REG_DIR / sub / ses / "synthseg_in_qsm.nii.gz"
        if ss_path.exists():
            ss_data = np.asanyarray(
                nib.load(str(ss_path)).dataobj).astype(np.int32)
            if ss_data.shape[:3] == qsm_data.shape[:3]:
                rows = extract_synthseg(qsm_data, ss_data, sub, ses)
                all_synthseg.extend(rows)
                log.info(f"  SynthSeg: {len(rows)} ROIs extracted")
            else:
                log.warning(f"  SynthSeg shape mismatch: "
                            f"seg={ss_data.shape} qsm={qsm_data.shape}")
                skipped.append((sub, ses, "synthseg shape mismatch"))
        else:
            log.warning(f"  SynthSeg not found: {ss_path}")
            skipped.append((sub, ses, "synthseg missing"))

        # 3. Load Seg_DN_Basal_Ganglia
        bg_path = REG_DIR / sub / ses / "Seg_DN_Basal_Ganglia.nii.gz"
        if bg_path.exists():
            bg_data = np.asanyarray(
                nib.load(str(bg_path)).dataobj).astype(np.int32)
            if bg_data.shape[:3] == qsm_data.shape[:3]:
                rows = extract_manual_seg(qsm_data, bg_data, sub, ses)
                all_manual.extend(rows)
                log.info(f"  Manual seg: {len(rows)} ROIs extracted")
            else:
                log.warning(f"  Manual seg shape mismatch: "
                            f"seg={bg_data.shape} qsm={qsm_data.shape}")
                skipped.append((sub, ses, "manual_seg shape mismatch"))
        else:
            log.warning(f"  Manual seg not found: {bg_path}")
            skipped.append((sub, ses, "manual_seg missing"))

    # ── Write CSVs ─────────────────────────────────────────────────────────
    FIELDNAMES = [
        "subject", "session", "field_strength", "mask", "label_id",
        "roi", "side", "n_voxels",
        "mean_ppb", "std_ppb", "median_ppb", "q25_ppb", "q75_ppb",
    ]

    csv_ss = OUT_DIR / "synthseg_roi_values.csv"
    with open(csv_ss, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDNAMES, extrasaction="ignore")
        w.writeheader()
        w.writerows(all_synthseg)
    log.info(f"\n✓ SynthSeg CSV:  {csv_ss}  ({len(all_synthseg)} rows)")

    csv_ms = OUT_DIR / "manual_seg_roi_values.csv"
    with open(csv_ms, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDNAMES, extrasaction="ignore")
        w.writeheader()
        w.writerows(all_manual)
    log.info(f"✓ Manual CSV:    {csv_ms}  ({len(all_manual)} rows)")

    # ── Summary ────────────────────────────────────────────────────────────
    if skipped:
        log.warning(f"\nSkipped ({len(skipped)}):")
        for s in skipped:
            log.warning(f"  {s}")

    # Summary table — SynthSeg curated labels
    KEY_SS = ["Thalamus", "Caudate", "Putamen", "Pallidum",
              "Hippocampus", "Amygdala", "VentralDC",
              "Cerebral-Cortex", "Cerebral-White-Matter"]
    log.info("\n" + "=" * 90)
    log.info("SUMMARY — SynthSeg curated (key structures, mean ± std, ppb)")
    log.info(f"{'Subject':14s} {'Session':18s} {'ROI':25s} {'Side':>5s} "
             f"{'Mean':>10s} {'Std':>10s} {'Median':>10s} {'N':>8s}")
    log.info("-" * 96)
    for r in all_synthseg:
        if r['roi'] in KEY_SS:
            log.info(f"  {r['subject']:14s} {r['session']:18s} "
                     f"{r['roi']:25s} {r['side']:>5s} "
                     f"{r['mean_ppb']:>10s} {r['std_ppb']:>10s} "
                     f"{r['median_ppb']:>10s} {r['n_voxels']:>8d}")

    # Summary table — Manual seg
    log.info("\n" + "=" * 90)
    log.info("SUMMARY — Seg_DN_Basal_Ganglia (mean ± std, ppb)")
    log.info(f"{'Subject':14s} {'Session':18s} {'ROI':20s} {'Side':>5s} "
             f"{'Mean':>10s} {'Std':>10s} {'Median':>10s} {'N':>6s}")
    log.info("-" * 93)
    for r in all_manual:
        log.info(f"  {r['subject']:14s} {r['session']:18s} "
                 f"{r['roi']:20s} {r['side']:>5s} "
                 f"{r['mean_ppb']:>10s} {r['std_ppb']:>10s} "
                 f"{r['median_ppb']:>10s} {r['n_voxels']:>6d}")

    log.info(f"\nDone! Total: {len(all_synthseg)} SynthSeg + "
             f"{len(all_manual)} manual-seg rows.")


if __name__ == "__main__":
    main()
