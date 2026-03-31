#!/usr/bin/env python3
"""
Register SynthSeg segmentation to QSM space and extract ROI values.

Pipeline:
  1. Register T1 → QSM (rigid, using first-echo magnitude as proxy)
  2. Apply same transform to SynthSeg labels (nearest-neighbor interpolation)
  3. Extract QSM values per label
  4. Generate overlay QC figures

Usage:
  python code/register_and_extract.py [--subj sub-03Xiang] [--vis]
"""

import os, sys, glob, csv, argparse, logging
import numpy as np
import nibabel as nib
import ants
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import (
    NIFTI_ROOT, ROI_OUTPUT, OVERLAY_DIR,
    FREESURFER_ROI_LABELS, BILATERAL_ROIS, ROI_ORDER, CENTROID_ROI_DEFS,
)
from utils import create_brain_mask, compute_brain_centroid, voxel_sizes, world_to_voxel

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

SYNTHSEG_DIR = os.path.join(ROI_OUTPUT, "synthseg")
REG_DIR = os.path.join(ROI_OUTPUT, "registered")

# ============================================================
# Full aseg label table for fine-grained extraction
# ============================================================
ASEG_LABELS = {
    2: "Cerebral_WM_L", 3: "Cerebral_Cortex_L",
    4: "Lateral_Ventricle_L", 5: "Inf_Lat_Vent_L",
    7: "Cerebellum_WM_L", 8: "Cerebellum_Cortex_L",
    10: "Thalamus_L", 11: "Caudate_L", 12: "Putamen_L", 13: "GP_L",
    14: "Third_Ventricle", 15: "Fourth_Ventricle", 16: "BrainStem",
    17: "Hippocampus_L", 18: "Amygdala_L", 24: "CSF",
    26: "Accumbens_L", 28: "VentralDC_L",
    41: "Cerebral_WM_R", 42: "Cerebral_Cortex_R",
    43: "Lateral_Ventricle_R", 44: "Inf_Lat_Vent_R",
    46: "Cerebellum_WM_R", 47: "Cerebellum_Cortex_R",
    49: "Thalamus_R", 50: "Caudate_R", 51: "Putamen_R", 52: "GP_R",
    53: "Hippocampus_R", 54: "Amygdala_R",
    58: "Accumbens_R", 60: "VentralDC_R",
}

# Bilateral pairs: name -> (left_label_id, right_label_id)
BILATERAL_PAIRS = {
    "Cerebral_WM":       (2, 41),
    "Cerebral_Cortex":   (3, 42),
    "Lateral_Ventricle": (4, 43),
    "Inf_Lat_Vent":      (5, 44),
    "Cerebellum_WM":     (7, 46),
    "Cerebellum_Cortex": (8, 47),
    "Thalamus":          (10, 49),
    "Caudate":           (11, 50),
    "Putamen":           (12, 51),
    "GP":                (13, 52),
    "Hippocampus":       (17, 53),
    "Amygdala":          (18, 54),
    "Accumbens":         (26, 58),
    "VentralDC":         (28, 60),
}


def register_t1_to_qsm(t1_path, mag_path, seg_path, output_dir):
    """
    Register T1 to QSM space (via magnitude) using ANTs rigid registration.
    Apply transform to SynthSeg labels with nearest-neighbor interpolation.

    Returns path to registered segmentation in QSM space.
    """
    os.makedirs(output_dir, exist_ok=True)
    reg_seg_path = os.path.join(output_dir, "synthseg_in_qsm.nii.gz")

    if os.path.exists(reg_seg_path):
        log.info(f"    Registered seg exists, loading: {reg_seg_path}")
        return reg_seg_path

    log.info(f"    Registering T1 → QSM space (rigid)...")

    # Load as ANTs images
    fixed = ants.image_read(mag_path)   # QSM space (magnitude)
    moving = ants.image_read(t1_path)   # T1 space

    # Rigid registration: T1 → magnitude
    reg = ants.registration(
        fixed=fixed,
        moving=moving,
        type_of_transform="Rigid",
        verbose=False,
    )

    log.info(f"    Registration done. Applying to segmentation labels...")

    # Apply transform to segmentation (nearest neighbor for labels)
    seg_ants = ants.image_read(seg_path)
    seg_in_qsm = ants.apply_transforms(
        fixed=fixed,
        moving=seg_ants,
        transformlist=reg["fwdtransforms"],
        interpolator="nearestNeighbor",
    )

    # Save
    ants.image_write(seg_in_qsm, reg_seg_path)
    log.info(f"    Saved registered seg: {reg_seg_path}")

    # Also save registered T1 for QC
    reg_t1_path = os.path.join(output_dir, "t1_in_qsm.nii.gz")
    ants.image_write(reg["warpedmovout"], reg_t1_path)

    return reg_seg_path


def _roi_stats(vals):
    """Compute statistics for an array of QSM values."""
    return {
        "n_voxels": len(vals),
        "mean": float(np.mean(vals)),
        "std": float(np.std(vals)),
        "median": float(np.median(vals)),
        "q25": float(np.percentile(vals, 25)),
        "q75": float(np.percentile(vals, 75)),
    }


def extract_qsm_from_labels(qsm_data, seg_data, brain_mask=None):
    """
    Extract QSM statistics for every aseg label present in seg_data.
    Returns three dicts: individual (L/R), bilateral (merged), laterality index.
    """
    if brain_mask is None:
        brain_mask = np.ones(qsm_data.shape[:3], dtype=bool)

    individual = {}
    unique_labels = np.unique(seg_data)

    for label_id in unique_labels:
        if label_id == 0:
            continue
        label_name = ASEG_LABELS.get(int(label_id), f"label_{int(label_id)}")
        roi_mask = (seg_data == label_id) & brain_mask
        n = int(roi_mask.sum())
        if n == 0:
            continue
        vals = qsm_data[roi_mask]
        stats = _roi_stats(vals)
        stats["label_id"] = int(label_id)
        individual[label_name] = stats

    # Bilateral: merge L+R voxels
    bilateral = {}
    for name, (lid_l, lid_r) in BILATERAL_PAIRS.items():
        mask_l = (seg_data == lid_l) & brain_mask
        mask_r = (seg_data == lid_r) & brain_mask
        combined = mask_l | mask_r
        n = int(combined.sum())
        if n == 0:
            continue
        vals = qsm_data[combined]
        stats = _roi_stats(vals)
        stats["label_id_l"] = lid_l
        stats["label_id_r"] = lid_r
        # Left-right comparison
        n_l, n_r = int(mask_l.sum()), int(mask_r.sum())
        mean_l = float(np.mean(qsm_data[mask_l])) if n_l > 0 else np.nan
        mean_r = float(np.mean(qsm_data[mask_r])) if n_r > 0 else np.nan
        stats["mean_L"] = mean_l
        stats["mean_R"] = mean_r
        stats["LR_diff"] = mean_l - mean_r if (n_l > 0 and n_r > 0) else np.nan
        # Laterality index: (L-R) / ((L+R)/2) * 100
        avg = (mean_l + mean_r) / 2.0
        stats["laterality_index"] = ((mean_l - mean_r) / avg * 100.0) if (n_l > 0 and n_r > 0 and abs(avg) > 1e-6) else np.nan
        bilateral[name] = stats

    # Midline structures (no L/R)
    for label_id in [14, 15, 16, 24]:  # 3rd/4th ventricle, brainstem, CSF
        label_name = ASEG_LABELS.get(label_id)
        if label_name and label_name in individual:
            stats = individual[label_name].copy()
            bilateral[label_name] = stats

    return individual, bilateral


def generate_overlay_figure(qsm_data, seg_data, subj_id, sess_id, output_path):
    """Generate QSM + segmentation overlay for QC."""
    # Pick 6 axial slices with most labels
    label_counts = []
    for z in range(seg_data.shape[2]):
        n_labels = len(np.unique(seg_data[:, :, z])) - 1  # exclude 0
        label_counts.append((z, n_labels))
    label_counts.sort(key=lambda x: -x[1])
    slices = sorted([s[0] for s in label_counts[:6]])

    fig, axes = plt.subplots(2, 3, figsize=(15, 10), facecolor="black")
    axes = axes.flatten()

    # QSM display range
    brain = seg_data > 0
    if brain.any():
        qsm_brain = qsm_data[brain]
        vmax = max(abs(np.percentile(qsm_brain, 2)), abs(np.percentile(qsm_brain, 98)))
    else:
        vmax = 100

    for ax, z in zip(axes, slices):
        qsm_slice = qsm_data[:, :, z].T
        seg_slice = seg_data[:, :, z].T

        ax.imshow(qsm_slice, cmap="RdBu_r", vmin=-vmax, vmax=vmax,
                  origin="lower", aspect="equal")

        # Overlay segmentation contours
        from scipy import ndimage
        for label_id in np.unique(seg_slice):
            if label_id == 0:
                continue
            mask = (seg_slice == label_id).astype(float)
            contour = mask - ndimage.binary_erosion(mask)
            color = plt.cm.tab20(label_id % 20)
            ax.contour(contour, levels=[0.5], colors=[color], linewidths=0.5)

        ax.set_title(f"z={z}", color="white", fontsize=9)
        ax.axis("off")

    fig.suptitle(f"{subj_id} / {sess_id} — SynthSeg ROI on QSM",
                 color="white", fontsize=13, fontweight="bold", y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(output_path, dpi=150, facecolor="black",
                bbox_inches="tight", pad_inches=0.1)
    plt.close(fig)
    log.info(f"    Overlay saved: {output_path}")


def find_t1_for_subject(subj_id):
    """Find T1 file for a subject (any session)."""
    t1_files = sorted(glob.glob(os.path.join(NIFTI_ROOT, subj_id, "ses-*", "anat", "T1w.nii.gz")))
    return t1_files[0] if t1_files else None


def process_subject_session(subj_id, sess_id, sess_dir, generate_vis=True):
    """Process one subject-session: register, extract, visualize."""
    qsm_dir = os.path.join(sess_dir, "qsm")
    if not os.path.exists(qsm_dir):
        return [], []

    # Find T1 and SynthSeg
    t1_path = find_t1_for_subject(subj_id)
    seg_path_candidates = [
        os.path.join(SYNTHSEG_DIR, subj_id, sess_id, "synthseg.nii.gz"),
        os.path.join(SYNTHSEG_DIR, subj_id, "synthseg.nii.gz"),
    ]
    seg_path = None
    for c in seg_path_candidates:
        if os.path.exists(c):
            seg_path = c
            break

    if not t1_path or not seg_path:
        log.info(f"  {subj_id}/{sess_id}: No T1 or SynthSeg, skipping atlas ROI")
        return [], []

    # Find magnitude for registration target
    mag_path = None
    for pat in ["mag_10echo_run-01_e1.nii.gz", "mag_5echo_run-01_e1.nii.gz"]:
        cand = os.path.join(qsm_dir, pat)
        if os.path.exists(cand):
            mag_path = cand
            break
    if not mag_path:
        cands = sorted(glob.glob(os.path.join(qsm_dir, "mag_*_e1.nii.gz")))
        if cands:
            mag_path = cands[0]
    if not mag_path:
        log.warning(f"  {subj_id}/{sess_id}: No magnitude image found")
        return [], []

    log.info(f"\n  {subj_id}/{sess_id}")

    # Register
    reg_out = os.path.join(REG_DIR, subj_id, sess_id)
    reg_seg_path = register_t1_to_qsm(t1_path, mag_path, seg_path, reg_out)

    # Load registered segmentation
    reg_seg_img = nib.load(reg_seg_path)
    reg_seg_data = np.asanyarray(reg_seg_img.dataobj).astype(np.int32)

    # Process each QSM file
    indiv_results = []
    bilat_results = []
    fs = 3.0 if "3T" in sess_id else 5.0
    qsm_files = sorted(glob.glob(os.path.join(qsm_dir, "qsm_*.nii.gz")))

    for qf in qsm_files:
        qb = os.path.basename(qf)
        et = "10echo" if "10echo" in qb else "5echo"
        sn = "run-02" if "run-02" in qb else "run-01"

        qi = nib.load(qf)
        qd = qi.get_fdata()
        if qd.ndim > 3:
            qd = qd[:, :, :, 0]

        # Verify shape match
        if reg_seg_data.shape != qd.shape[:3]:
            log.warning(f"    {qb}: shape mismatch seg={reg_seg_data.shape} qsm={qd.shape[:3]}, skip")
            continue

        # Extract individual + bilateral
        individual, bilateral = extract_qsm_from_labels(qd, reg_seg_data)

        base_info = {
            "subject": subj_id, "session": sess_id,
            "field_strength": fs, "echo_type": et,
            "scan": sn, "method": "atlas_registered",
        }

        # Individual L/R results
        for roi_name, stats in individual.items():
            row = {**base_info,
                "roi": roi_name, "side": "L" if roi_name.endswith("_L") else ("R" if roi_name.endswith("_R") else "M"),
                "aseg_label": stats["label_id"],
                "n_voxels": stats["n_voxels"],
                "mean_ppb": f"{stats['mean']:.4f}",
                "std_ppb": f"{stats['std']:.4f}",
                "median_ppb": f"{stats['median']:.4f}",
                "q25_ppb": f"{stats['q25']:.4f}",
                "q75_ppb": f"{stats['q75']:.4f}",
            }
            indiv_results.append(row)

        # Bilateral merged results
        for roi_name, stats in bilateral.items():
            row = {**base_info,
                "roi": roi_name, "side": "bilateral",
                "n_voxels": stats["n_voxels"],
                "mean_ppb": f"{stats['mean']:.4f}",
                "std_ppb": f"{stats['std']:.4f}",
                "median_ppb": f"{stats['median']:.4f}",
                "q25_ppb": f"{stats['q25']:.4f}",
                "q75_ppb": f"{stats['q75']:.4f}",
            }
            # Add L/R comparison columns if available
            if "mean_L" in stats:
                row["mean_L_ppb"] = f"{stats['mean_L']:.4f}"
                row["mean_R_ppb"] = f"{stats['mean_R']:.4f}"
                row["LR_diff_ppb"] = f"{stats['LR_diff']:.4f}" if not np.isnan(stats['LR_diff']) else ""
                row["laterality_index"] = f"{stats['laterality_index']:.2f}" if not np.isnan(stats['laterality_index']) else ""
            bilat_results.append(row)

        log.info(f"    {qb}: {len(individual)} individual + {len(bilateral)} bilateral ROIs")

        # Generate overlay for first 10echo run-01
        if generate_vis and et == "10echo" and sn == "run-01":
            os.makedirs(OVERLAY_DIR, exist_ok=True)
            overlay_path = os.path.join(OVERLAY_DIR,
                f"{subj_id}_{sess_id}_atlas_overlay.pdf")
            generate_overlay_figure(qd, reg_seg_data, subj_id, sess_id, overlay_path)

    return indiv_results, bilat_results


def main():
    parser = argparse.ArgumentParser(description="Register SynthSeg to QSM and extract ROIs")
    parser.add_argument("--subj", default="", help="Process specific subject (e.g. sub-03Xiang)")
    parser.add_argument("--vis", action="store_true", default=True, help="Generate overlay figures")
    parser.add_argument("--no-vis", dest="vis", action="store_false")
    args = parser.parse_args()

    os.makedirs(REG_DIR, exist_ok=True)
    os.makedirs(OVERLAY_DIR, exist_ok=True)

    log.info("=" * 60)
    log.info("Register SynthSeg → QSM space + Extract ROI values")
    log.info("=" * 60)

    all_indiv = []
    all_bilat = []

    for subj_dir in sorted(glob.glob(os.path.join(NIFTI_ROOT, "sub-*"))):
        subj_id = os.path.basename(subj_dir)
        if args.subj and subj_id != args.subj:
            continue

        for sess_dir in sorted(glob.glob(os.path.join(subj_dir, "ses-*"))):
            sess_id = os.path.basename(sess_dir)
            indiv, bilat = process_subject_session(subj_id, sess_id, sess_dir, args.vis)
            all_indiv.extend(indiv)
            all_bilat.extend(bilat)

    if not all_indiv and not all_bilat:
        log.warning("No results. Check T1/SynthSeg availability.")
        return

    # --- CSV 1: Individual L/R ROIs ---
    csv_indiv = os.path.join(ROI_OUTPUT, "roi_values_individual.csv")
    fn_indiv = ["subject", "session", "field_strength", "echo_type", "scan", "method",
                "roi", "side", "aseg_label", "n_voxels",
                "mean_ppb", "std_ppb", "median_ppb", "q25_ppb", "q75_ppb"]
    with open(csv_indiv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fn_indiv, extrasaction="ignore")
        w.writeheader()
        w.writerows(all_indiv)
    log.info(f"\nSaved individual: {csv_indiv} ({len(all_indiv)} entries)")

    # --- CSV 2: Bilateral merged + L/R comparison ---
    csv_bilat = os.path.join(ROI_OUTPUT, "roi_values_bilateral.csv")
    fn_bilat = ["subject", "session", "field_strength", "echo_type", "scan", "method",
                "roi", "side", "n_voxels",
                "mean_ppb", "std_ppb", "median_ppb", "q25_ppb", "q75_ppb",
                "mean_L_ppb", "mean_R_ppb", "LR_diff_ppb", "laterality_index"]
    with open(csv_bilat, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fn_bilat, extrasaction="ignore")
        w.writeheader()
        w.writerows(all_bilat)
    log.info(f"Saved bilateral:  {csv_bilat} ({len(all_bilat)} entries)")

    # --- CSV 3: Combined (backward compatible) ---
    csv_all = os.path.join(ROI_OUTPUT, "roi_values_atlas_registered.csv")
    fn_all = ["subject", "session", "field_strength", "echo_type", "scan", "method",
              "roi", "side", "n_voxels",
              "mean_ppb", "std_ppb", "median_ppb", "q25_ppb", "q75_ppb",
              "mean_L_ppb", "mean_R_ppb", "LR_diff_ppb", "laterality_index"]
    combined = all_indiv + all_bilat
    with open(csv_all, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fn_all, extrasaction="ignore")
        w.writeheader()
        w.writerows(combined)
    log.info(f"Saved combined:   {csv_all} ({len(combined)} entries)")

    # Quick summary
    log.info("\n" + "=" * 60)
    log.info("Bilateral ROI summary (10echo, run-01, ppb)")
    log.info(f"{'ROI':20s} {'n':>4s} {'Mean':>8s} {'±SD':>8s} {'Mean_L':>8s} {'Mean_R':>8s} {'L-R':>8s} {'LI%':>7s}")
    log.info("-" * 75)
    for r in all_bilat:
        if r["echo_type"] == "10echo" and r["scan"] == "run-01" and r.get("mean_L_ppb"):
            log.info(f"  {r['subject']:12s} {r['roi']:15s} {r['n_voxels']:>5d} "
                     f"{r['mean_ppb']:>8s} {r['std_ppb']:>8s} "
                     f"{r.get('mean_L_ppb',''):>8s} {r.get('mean_R_ppb',''):>8s} "
                     f"{r.get('LR_diff_ppb',''):>8s} {r.get('laterality_index',''):>7s}")


if __name__ == "__main__":
    main()
