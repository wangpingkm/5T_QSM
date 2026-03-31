#!/usr/bin/env python3
"""
Extract QSM values from manual Seg_DN_Basal_Ganglia segmentation.

The Seg_DN_Basal_Ganglia.nii.gz is in the same QSM space for all sessions
of a given subject, so no cross-session registration is needed.

Label map:
  1  Dentate_nucleus_R    2  Dentate_nucleus_L
  3  Substantia_nigra_R   4  Substantia_nigra_L
  5  Nucleus_ruber_R      6  Nucleus_ruber_L
  7  Thalamus_R           8  Thalamus_L
  9  Caudate_nucleus_R   10  Caudate_nucleus_L
 11  Putamen_R           12  Putamen_L
 13  Globus_pallidus_R   14  Globus_pallidus_L

Usage:
  python code/roi_manual_seg.py
  python code/roi_manual_seg.py --subj sub-03Xiang
"""

import os, sys, glob, csv, argparse, logging
from pathlib import Path

import numpy as np
import nibabel as nib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import ndimage

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import NIFTI_ROOT, ROI_OUTPUT

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# ── paths ──────────────────────────────────────────────────────────────────
REG_DIR     = os.path.join(ROI_OUTPUT, "registered")
MANUAL_DIR  = os.path.join(ROI_OUTPUT, "manual_seg")
OVERLAY_DIR = os.path.join(ROI_OUTPUT, "overlays_manual")

# ── label table ────────────────────────────────────────────────────────────
LABEL_MAP = {
    1:  "Dentate_nucleus_R",
    2:  "Dentate_nucleus_L",
    3:  "Substantia_nigra_R",
    4:  "Substantia_nigra_L",
    5:  "Nucleus_ruber_R",
    6:  "Nucleus_ruber_L",
    7:  "Thalamus_R",
    8:  "Thalamus_L",
    9:  "Caudate_nucleus_R",
    10: "Caudate_nucleus_L",
    11: "Putamen_R",
    12: "Putamen_L",
    13: "Globus_pallidus_R",
    14: "Globus_pallidus_L",
}

# Paired structures for bilateral + L/R comparison
BILATERAL_PAIRS = {
    "Dentate_nucleus":    (2, 1),   # (L_label, R_label)
    "Substantia_nigra":   (4, 3),
    "Nucleus_ruber":      (6, 5),
    "Thalamus":           (8, 7),
    "Caudate_nucleus":    (10, 9),
    "Putamen":            (12, 11),
    "Globus_pallidus":    (14, 13),
}

# Color for each label in overlay (tab20 colormap)
LABEL_COLORS = {lid: plt.cm.tab20(i / 14) for i, lid in enumerate(LABEL_MAP)}


def find_seg_for_subject(subj_id):
    """Find Seg_DN_Basal_Ganglia.nii.gz for a subject (any session dir)."""
    pattern = os.path.join(REG_DIR, subj_id, "ses-*", "Seg_DN_Basal_Ganglia.nii.gz")
    files = sorted(glob.glob(pattern))
    return files[0] if files else None


def roi_stats(vals):
    return {
        "n_voxels": len(vals),
        "mean":   float(np.mean(vals)),
        "std":    float(np.std(vals)),
        "median": float(np.median(vals)),
        "q25":    float(np.percentile(vals, 25)),
        "q75":    float(np.percentile(vals, 75)),
    }


def extract_roi_values(qsm_data, seg_data):
    """
    Returns:
      individual: {roi_name: stats_dict}  — one entry per label
      bilateral:  {roi_name: stats_dict}  — merged L+R + L/R comparison
    """
    individual = {}
    for lid, name in LABEL_MAP.items():
        mask = seg_data == lid
        n = int(mask.sum())
        if n == 0:
            continue
        s = roi_stats(qsm_data[mask])
        s["label_id"] = lid
        individual[name] = s

    bilateral = {}
    for name, (lid_l, lid_r) in BILATERAL_PAIRS.items():
        mask_l = seg_data == lid_l
        mask_r = seg_data == lid_r
        n_l, n_r = int(mask_l.sum()), int(mask_r.sum())
        combined = mask_l | mask_r
        if combined.sum() == 0:
            continue
        s = roi_stats(qsm_data[combined])
        s["label_id_l"] = lid_l
        s["label_id_r"] = lid_r
        s["mean_L"] = float(np.mean(qsm_data[mask_l])) if n_l > 0 else float("nan")
        s["mean_R"] = float(np.mean(qsm_data[mask_r])) if n_r > 0 else float("nan")
        if n_l > 0 and n_r > 0:
            s["LR_diff"] = s["mean_L"] - s["mean_R"]
            avg = (s["mean_L"] + s["mean_R"]) / 2.0
            s["laterality_index"] = (s["LR_diff"] / avg * 100.0) if abs(avg) > 1e-6 else float("nan")
        else:
            s["LR_diff"] = float("nan")
            s["laterality_index"] = float("nan")
        bilateral[name] = s

    return individual, bilateral


def generate_overlay_pdf(qsm_data, seg_data, subj_id, sess_id, output_path):
    """
    Generate overlay PDF showing only Z slices where Seg_DN_Basal_Ganglia has data.
    Layout: up to 6 slices per row, multiple rows if needed.
    """
    # Find Z slices with any label
    z_with_data = [z for z in range(seg_data.shape[2])
                   if np.any(seg_data[:, :, z] > 0)]
    if not z_with_data:
        log.warning(f"    No seg data found, skipping overlay: {output_path}")
        return

    n_slices = len(z_with_data)
    ncols = min(6, n_slices)
    nrows = (n_slices + ncols - 1) // ncols

    # QSM display range from brain voxels
    brain_vals = qsm_data[seg_data > 0]
    vmax = max(abs(np.percentile(brain_vals, 2)), abs(np.percentile(brain_vals, 98))) if len(brain_vals) > 0 else 100

    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 3, nrows * 3.2), facecolor="black")
    if nrows == 1 and ncols == 1:
        axes = np.array([[axes]])
    elif nrows == 1:
        axes = axes[np.newaxis, :]
    elif ncols == 1:
        axes = axes[:, np.newaxis]

    for idx, z in enumerate(z_with_data):
        row, col = divmod(idx, ncols)
        ax = axes[row, col]

        qsm_sl = qsm_data[:, :, z].T
        seg_sl = seg_data[:, :, z].T

        ax.imshow(qsm_sl, cmap="RdBu_r", vmin=-vmax, vmax=vmax,
                  origin="lower", aspect="equal", interpolation="nearest")

        # Draw filled semi-transparent overlay + contour per label
        for lid in np.unique(seg_sl):
            if lid == 0:
                continue
            color = LABEL_COLORS.get(int(lid), (1, 1, 1, 1))
            mask = (seg_sl == lid).astype(float)
            # Filled overlay at 30% alpha
            rgba = np.zeros((*mask.shape, 4))
            rgba[..., :3] = color[:3]
            rgba[..., 3] = mask * 0.35
            ax.imshow(rgba, origin="lower", aspect="equal", interpolation="nearest")
            # Contour
            contour = mask - ndimage.binary_erosion(mask > 0)
            ax.contour(contour, levels=[0.5], colors=[color[:3]], linewidths=0.8)

        ax.set_title(f"z={z}", color="white", fontsize=8)
        ax.axis("off")

    # Hide unused axes
    for idx in range(n_slices, nrows * ncols):
        row, col = divmod(idx, ncols)
        axes[row, col].set_visible(False)

    # Legend
    legend_handles = [
        plt.Line2D([0], [0], color=LABEL_COLORS[lid][:3], linewidth=3,
                   label=f"{lid}: {LABEL_MAP[lid]}")
        for lid in LABEL_MAP if lid in np.unique(seg_data)
    ]
    if legend_handles:
        fig.legend(handles=legend_handles, loc="lower center",
                   ncol=4, fontsize=7, framealpha=0.3,
                   labelcolor="white", facecolor="black",
                   bbox_to_anchor=(0.5, 0.0))

    fig.suptitle(f"{subj_id} / {sess_id} — Seg_DN_Basal_Ganglia on QSM",
                 color="white", fontsize=12, fontweight="bold", y=1.01)
    fig.tight_layout(rect=[0, 0.06, 1, 1])
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=150, facecolor="black",
                bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)
    log.info(f"    Overlay saved ({n_slices} slices): {output_path}")


def process_subject(subj_id, generate_vis=True, args_subj=""):
    """Process all sessions for one subject."""
    if args_subj and subj_id != args_subj:
        return [], []

    seg_path = find_seg_for_subject(subj_id)
    if not seg_path:
        log.info(f"  {subj_id}: No Seg_DN_Basal_Ganglia found, skipping")
        return [], []

    seg_img = nib.load(seg_path)
    seg_data = np.asanyarray(seg_img.dataobj).astype(np.int32)
    log.info(f"\n{'='*55}")
    log.info(f"  {subj_id}: seg={seg_path}")
    log.info(f"  Seg shape={seg_data.shape}, labels={sorted(np.unique(seg_data[seg_data>0]).tolist())}")

    all_indiv, all_bilat = [], []
    subj_dir = os.path.join(NIFTI_ROOT, subj_id)

    for sess_dir in sorted(glob.glob(os.path.join(subj_dir, "ses-*"))):
        sess_id = os.path.basename(sess_dir)
        qsm_dir = os.path.join(sess_dir, "qsm")
        if not os.path.exists(qsm_dir):
            continue

        qsm_files = sorted(glob.glob(os.path.join(qsm_dir, "qsm_*.nii.gz")))
        if not qsm_files:
            continue

        log.info(f"\n  {subj_id}/{sess_id}: {len(qsm_files)} QSM file(s)")
        fs = 3.0 if "3T" in sess_id else 5.0
        overlay_done = False

        for qf in qsm_files:
            qb = os.path.basename(qf)
            et = "10echo" if "10echo" in qb else "5echo"
            sn = "run-02" if "run-02" in qb else "run-01"

            qi = nib.load(qf)
            qd = qi.get_fdata()
            if qd.ndim > 3:
                qd = qd[..., 0]

            if qd.shape[:3] != seg_data.shape[:3]:
                log.warning(f"    {qb}: shape mismatch qsm={qd.shape[:3]} seg={seg_data.shape}, skip")
                continue

            individual, bilateral = extract_roi_values(qd, seg_data)

            base = {
                "subject": subj_id, "session": sess_id,
                "field_strength": fs, "echo_type": et, "scan": sn,
                "method": "manual_seg",
            }

            for roi_name, s in individual.items():
                side = "R" if roi_name.endswith("_R") else "L"
                all_indiv.append({**base,
                    "roi": roi_name, "side": side,
                    "label_id": s["label_id"],
                    "n_voxels": s["n_voxels"],
                    "mean_ppb":   f"{s['mean']:.4f}",
                    "std_ppb":    f"{s['std']:.4f}",
                    "median_ppb": f"{s['median']:.4f}",
                    "q25_ppb":    f"{s['q25']:.4f}",
                    "q75_ppb":    f"{s['q75']:.4f}",
                })

            for roi_name, s in bilateral.items():
                row = {**base,
                    "roi": roi_name, "side": "bilateral",
                    "n_voxels": s["n_voxels"],
                    "mean_ppb":   f"{s['mean']:.4f}",
                    "std_ppb":    f"{s['std']:.4f}",
                    "median_ppb": f"{s['median']:.4f}",
                    "q25_ppb":    f"{s['q25']:.4f}",
                    "q75_ppb":    f"{s['q75']:.4f}",
                    "mean_L_ppb": f"{s['mean_L']:.4f}" if not np.isnan(s["mean_L"]) else "",
                    "mean_R_ppb": f"{s['mean_R']:.4f}" if not np.isnan(s["mean_R"]) else "",
                    "LR_diff_ppb": f"{s['LR_diff']:.4f}" if not np.isnan(s["LR_diff"]) else "",
                    "laterality_index": f"{s['laterality_index']:.2f}" if not np.isnan(s["laterality_index"]) else "",
                }
                all_bilat.append(row)

            log.info(f"    {qb}: {len(individual)} individual + {len(bilateral)} bilateral ROIs extracted")

            # Overlay: only for 10echo run-01, once per session
            if generate_vis and et == "10echo" and sn == "run-01" and not overlay_done:
                overlay_path = os.path.join(OVERLAY_DIR, f"{subj_id}_{sess_id}_manual_overlay.pdf")
                generate_overlay_pdf(qd, seg_data, subj_id, sess_id, overlay_path)
                overlay_done = True

    return all_indiv, all_bilat


def main():
    parser = argparse.ArgumentParser(description="Extract QSM from Seg_DN_Basal_Ganglia")
    parser.add_argument("--subj", default="", help="Process specific subject only")
    parser.add_argument("--no-vis", dest="vis", action="store_false", default=True)
    args = parser.parse_args()

    os.makedirs(MANUAL_DIR, exist_ok=True)
    os.makedirs(OVERLAY_DIR, exist_ok=True)

    log.info("=" * 55)
    log.info("QSM extraction from Seg_DN_Basal_Ganglia (manual seg)")
    log.info("=" * 55)

    all_indiv, all_bilat = [], []

    for subj_dir in sorted(glob.glob(os.path.join(NIFTI_ROOT, "sub-*"))):
        subj_id = os.path.basename(subj_dir)
        indiv, bilat = process_subject(subj_id, args.vis, args.subj)
        all_indiv.extend(indiv)
        all_bilat.extend(bilat)

    if not all_indiv:
        log.warning("No results. Check Seg_DN_Basal_Ganglia availability.")
        return

    # ── CSV 1: individual L/R ──────────────────────────────────────────────
    fn_indiv = ["subject", "session", "field_strength", "echo_type", "scan", "method",
                "roi", "side", "label_id", "n_voxels",
                "mean_ppb", "std_ppb", "median_ppb", "q25_ppb", "q75_ppb"]
    csv_indiv = os.path.join(MANUAL_DIR, "roi_manual_individual.csv")
    with open(csv_indiv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fn_indiv, extrasaction="ignore")
        w.writeheader(); w.writerows(all_indiv)
    log.info(f"\nSaved individual: {csv_indiv} ({len(all_indiv)} rows)")

    # ── CSV 2: bilateral + L/R comparison ─────────────────────────────────
    fn_bilat = ["subject", "session", "field_strength", "echo_type", "scan", "method",
                "roi", "side", "n_voxels",
                "mean_ppb", "std_ppb", "median_ppb", "q25_ppb", "q75_ppb",
                "mean_L_ppb", "mean_R_ppb", "LR_diff_ppb", "laterality_index"]
    csv_bilat = os.path.join(MANUAL_DIR, "roi_manual_bilateral.csv")
    with open(csv_bilat, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fn_bilat, extrasaction="ignore")
        w.writeheader(); w.writerows(all_bilat)
    log.info(f"Saved bilateral:  {csv_bilat} ({len(all_bilat)} rows)")

    # ── Summary table ──────────────────────────────────────────────────────
    log.info("\n" + "=" * 75)
    log.info("Summary (10echo, run-01, bilateral mean ± SD, ppb)")
    log.info(f"{'Subject':12s} {'Session':18s} {'ROI':20s} {'Mean':>8s} {'±SD':>8s} {'Mean_L':>8s} {'Mean_R':>8s} {'LI%':>7s}")
    log.info("-" * 95)
    for r in all_bilat:
        if r["echo_type"] == "10echo" and r["scan"] == "run-01":
            log.info(f"  {r['subject']:12s} {r['session']:18s} {r['roi']:20s} "
                     f"{r['mean_ppb']:>8s} {r['std_ppb']:>8s} "
                     f"{r.get('mean_L_ppb',''):>8s} {r.get('mean_R_ppb',''):>8s} "
                     f"{r.get('laterality_index',''):>7s}")


if __name__ == "__main__":
    main()
