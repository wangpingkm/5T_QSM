#!/usr/bin/env python3
"""
Step 4v2: Atlas-Based ROI Analysis for QSM
============================================
Uses FreeSurfer SynthSeg segmentation to define anatomically accurate ROIs,
then extracts QSM values from each brain region.

Strategy:
  1. Run SynthSeg on first-echo magnitude (or T1 if available) to get
     subcortical segmentation labels in native space.
  2. Map FreeSurfer aseg labels to target ROIs (finest parcellation).
  3. For ROIs not in standard aseg (RN, SN, DN), fall back to centroid-based.
  4. Extract QSM statistics per ROI per dataset.
  5. Generate overlay figures for visual QC.

The script supports THREE levels of ROI granularity:
  - "fine":   Every individual FreeSurfer aseg label (~100 regions)
  - "medium": Grouped into ~16 anatomical regions (bilateral merge)
  - "coarse": The 8 standard ROIs used in the QSM literature

Dependencies:
  - FreeSurfer 7+ with mri_synthseg
  - nibabel, numpy, scipy, matplotlib

Usage:
  python step4_roi_analysis_v2.py [--method atlas|centroid|hybrid]
                                  [--granularity fine|medium|coarse]
                                  [--skip-synthseg]
"""

import os, csv, glob, json, argparse, subprocess, time, logging
import numpy as np
import nibabel as nib
from scipy import ndimage

from config import (
    NIFTI_ROOT, ROI_OUTPUT, OVERLAY_DIR, FREESURFER_HOME, FS_LICENSE,
    SYNTHSEG_BIN_CANDIDATES, FREESURFER_ROI_LABELS, BILATERAL_ROIS,
    ROI_ORDER, CENTROID_ROI_DEFS, UNIT,
)
from utils import create_brain_mask, compute_brain_centroid, voxel_sizes, world_to_voxel

SYNTHSEG_DIR = os.path.join(ROI_OUTPUT, "synthseg")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# ============================================================
# FreeSurfer aseg label table (finest granularity)
# ============================================================
# Complete mapping: aseg label ID -> (name, group)
ASEG_LABELS = {
    2:  ("Left-Cerebral-White-Matter",       "Cerebral_WM_L"),
    3:  ("Left-Cerebral-Cortex",             "Cerebral_Cortex_L"),
    4:  ("Left-Lateral-Ventricle",           "Lateral_Ventricle_L"),
    5:  ("Left-Inf-Lat-Vent",               "Inf_Lat_Vent_L"),
    7:  ("Left-Cerebellum-White-Matter",     "Cerebellum_WM_L"),
    8:  ("Left-Cerebellum-Cortex",           "Cerebellum_Cortex_L"),
    10: ("Left-Thalamus",                    "Thalamus_L"),
    11: ("Left-Caudate",                     "Caudate_L"),
    12: ("Left-Putamen",                     "Putamen_L"),
    13: ("Left-Pallidum",                    "GP_L"),
    14: ("3rd-Ventricle",                    "Third_Ventricle"),
    15: ("4th-Ventricle",                    "Fourth_Ventricle"),
    16: ("Brain-Stem",                       "BrainStem"),
    17: ("Left-Hippocampus",                 "Hippocampus_L"),
    18: ("Left-Amygdala",                    "Amygdala_L"),
    24: ("CSF",                              "CSF"),
    26: ("Left-Accumbens-area",              "Accumbens_L"),
    28: ("Left-VentralDC",                   "VentralDC_L"),
    41: ("Right-Cerebral-White-Matter",      "Cerebral_WM_R"),
    42: ("Right-Cerebral-Cortex",            "Cerebral_Cortex_R"),
    43: ("Right-Lateral-Ventricle",          "Lateral_Ventricle_R"),
    44: ("Right-Inf-Lat-Vent",              "Inf_Lat_Vent_R"),
    46: ("Right-Cerebellum-White-Matter",    "Cerebellum_WM_R"),
    47: ("Right-Cerebellum-Cortex",          "Cerebellum_Cortex_R"),
    49: ("Right-Thalamus",                   "Thalamus_R"),
    50: ("Right-Caudate",                    "Caudate_R"),
    51: ("Right-Putamen",                    "Putamen_R"),
    52: ("Right-Pallidum",                   "GP_R"),
    53: ("Right-Hippocampus",                "Hippocampus_R"),
    54: ("Right-Amygdala",                   "Amygdala_R"),
    58: ("Right-Accumbens-area",             "Accumbens_R"),
    60: ("Right-VentralDC",                  "VentralDC_R"),
}

# Medium grouping: merge bilateral + group structures
MEDIUM_GROUPS = {
    "Caudate":         [11, 50],
    "Putamen":         [12, 51],
    "GP":              [13, 52],
    "Thalamus":        [10, 49],
    "Hippocampus":     [17, 53],
    "Amygdala":        [18, 54],
    "Accumbens":       [26, 58],
    "VentralDC":       [28, 60],
    "Cerebellum_Cortex": [8, 47],
    "Cerebellum_WM":   [7, 46],
    "BrainStem":       [16],
    "Cerebral_WM":     [2, 41],
    "Cerebral_Cortex": [3, 42],
    "Lateral_Ventricle": [4, 5, 43, 44],
    "CSF":             [24],
    "Third_Ventricle": [14],
    "Fourth_Ventricle": [15],
}


# ============================================================
# SynthSeg runner
# ============================================================

def find_synthseg_bin():
    """Find mri_synthseg executable."""
    for candidate in SYNTHSEG_BIN_CANDIDATES:
        if os.path.isfile(candidate) and os.access(candidate, os.X_OK):
            return candidate
    # Try PATH
    import shutil
    found = shutil.which("mri_synthseg")
    if found:
        return found
    return None


def run_synthseg(input_image, output_dir, use_parc=True, use_robust=True):
    """
    Run SynthSeg on a single image.

    Parameters
    ----------
    input_image : str, path to input NIfTI
    output_dir : str, directory for outputs
    use_parc : bool, request cortical parcellation
    use_robust : bool, use robust mode for clinical data

    Returns
    -------
    seg_path : str or None, path to segmentation output
    """
    synthseg_bin = find_synthseg_bin()
    if synthseg_bin is None:
        log.error("mri_synthseg not found. Set FREESURFER_HOME in config.py")
        return None

    os.makedirs(output_dir, exist_ok=True)
    seg_path = os.path.join(output_dir, "synthseg.nii.gz")
    vol_path = os.path.join(output_dir, "synthseg_volumes.csv")
    qc_path = os.path.join(output_dir, "synthseg_qc.csv")

    if os.path.exists(seg_path):
        log.info(f"  SynthSeg output exists, skipping: {seg_path}")
        return seg_path

    cmd = [
        synthseg_bin,
        "--i", input_image,
        "--o", seg_path,
        "--vol", vol_path,
        "--qc", qc_path,
        "--robust" if use_robust else "",
        "--parc" if use_parc else "",
        "--cpu",
    ]
    cmd = [c for c in cmd if c]  # remove empty strings

    env = os.environ.copy()
    env["FREESURFER_HOME"] = FREESURFER_HOME
    env["FS_LICENSE"] = FS_LICENSE
    env["PATH"] = os.path.join(FREESURFER_HOME, "bin") + os.pathsep + env.get("PATH", "")

    log.info(f"  Running SynthSeg: {' '.join(cmd)}")
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=1800, env=env)
        if result.returncode != 0:
            log.error(f"  SynthSeg failed: {result.stderr[:500]}")
            return None
        log.info(f"  SynthSeg completed: {seg_path}")
        return seg_path
    except subprocess.TimeoutExpired:
        log.error("  SynthSeg timed out (30min)")
        return None
    except Exception as e:
        log.error(f"  SynthSeg error: {e}")
        return None


# ============================================================
# ROI extraction from atlas labels
# ============================================================

def extract_roi_from_atlas(qsm_data, seg_data, brain_mask, label_ids, roi_name):
    """
    Extract QSM values from voxels matching atlas label IDs.

    Parameters
    ----------
    qsm_data : 3D array
    seg_data : 3D array of integer labels
    brain_mask : 3D bool array
    label_ids : list of int, FreeSurfer label IDs
    roi_name : str

    Returns
    -------
    dict with statistics, or None if no voxels found
    """
    roi_mask = np.isin(seg_data, label_ids) & brain_mask
    n_voxels = int(roi_mask.sum())

    if n_voxels == 0:
        return None

    vals = qsm_data[roi_mask]

    return {
        "roi": roi_name,
        "n_voxels": n_voxels,
        "mean": float(np.mean(vals)),
        "std": float(np.std(vals)),
        "median": float(np.median(vals)),
        "q25": float(np.percentile(vals, 25)),
        "q75": float(np.percentile(vals, 75)),
        "min": float(np.min(vals)),
        "max": float(np.max(vals)),
    }


def extract_roi_centroid(qsm_data, brain_mask, affine, com_world, roi_cfg, roi_name):
    """
    Legacy centroid-based ROI extraction (for RN, SN, DN not in SynthSeg aseg).
    Vectorized implementation replacing the original triple-nested loop.
    """
    vs = voxel_sizes(affine)
    shape = qsm_data.shape[:3]
    target_world = com_world + np.array(roi_cfg["offset"], dtype=float)
    target_vox = world_to_voxel(affine, target_world)

    # Optional peak refinement
    if roi_cfg["refine"] in ("max", "min"):
        sr = roi_cfg["search_r"]
        sr_vox = sr / vs
        qsm_smooth = ndimage.gaussian_filter(qsm_data * brain_mask, sigma=2.0 / vs)

        # Build search bounding box
        slices = tuple(
            slice(max(0, int(target_vox[ax] - sr_vox[ax])),
                  min(shape[ax], int(target_vox[ax] + sr_vox[ax]) + 1))
            for ax in range(3)
        )
        sub_mask = brain_mask[slices]
        sub_smooth = qsm_smooth[slices]

        # Build distance grid within bounding box
        offsets = [np.arange(s.start, s.stop) for s in slices]
        gi, gj, gk = np.meshgrid(*offsets, indexing='ij')
        dist2 = sum(((g - t) * s) ** 2 for g, t, s in zip((gi, gj, gk), target_vox, vs))
        search_valid = (dist2 <= sr ** 2) & sub_mask

        if search_valid.any():
            sub_vals = np.where(search_valid, sub_smooth, -np.inf if roi_cfg["refine"] == "max" else np.inf)
            if roi_cfg["refine"] == "max":
                best_idx = np.unravel_index(np.argmax(sub_vals), sub_vals.shape)
            else:
                best_idx = np.unravel_index(np.argmin(sub_vals), sub_vals.shape)
            target_vox = np.array([
                offsets[0][best_idx[0]],
                offsets[1][best_idx[1]],
                offsets[2][best_idx[2]],
            ], dtype=float)

    # Extract sphere ROI (vectorized)
    rr = roi_cfg["roi_r"]
    rv = rr / vs
    slices = tuple(
        slice(max(0, int(target_vox[ax] - rv[ax]) - 1),
              min(shape[ax], int(target_vox[ax] + rv[ax]) + 2))
        for ax in range(3)
    )
    offsets = [np.arange(s.start, s.stop) for s in slices]
    gi, gj, gk = np.meshgrid(*offsets, indexing='ij')
    dist2 = sum(((g - t) * s) ** 2 for g, t, s in zip((gi, gj, gk), target_vox, vs))
    roi_mask = (dist2 <= rr ** 2) & brain_mask[slices]
    vals = qsm_data[slices][roi_mask]

    if len(vals) == 0:
        return None

    return {
        "roi": roi_name,
        "n_voxels": len(vals),
        "mean": float(np.mean(vals)),
        "std": float(np.std(vals)),
        "median": float(np.median(vals)),
        "q25": float(np.percentile(vals, 25)),
        "q75": float(np.percentile(vals, 75)),
        "min": float(np.min(vals)),
        "max": float(np.max(vals)),
        "center_vox": target_vox.tolist(),
    }


# ============================================================
# Main analysis loop
# ============================================================

def find_input_image(qsm_dir, sess_dir):
    """Find best input for SynthSeg: prefer T1, fall back to magnitude."""
    anat_dir = os.path.join(sess_dir, "anat")
    t1_path = os.path.join(anat_dir, "T1w.nii.gz")
    if os.path.exists(t1_path):
        return t1_path, "T1"

    # Fall back to first-echo magnitude
    for pat in ["mag_10echo_run-01_e1.nii.gz", "mag_5echo_run-01_e1.nii.gz"]:
        cand = os.path.join(qsm_dir, pat)
        if os.path.exists(cand):
            return cand, "magnitude"

    cands = sorted(glob.glob(os.path.join(qsm_dir, "mag_*_e1.nii.gz")))
    if cands:
        return cands[0], "magnitude"
    return None, None


def analyze_single_session(subj_id, sess_id, sess_dir, method="hybrid",
                           granularity="coarse", skip_synthseg=False):
    """
    Analyze one subject-session.

    Returns list of result dicts.
    """
    qsm_dir = os.path.join(sess_dir, "qsm")
    if not os.path.exists(qsm_dir):
        return []

    qsm_files = sorted(glob.glob(os.path.join(qsm_dir, "qsm_*.nii.gz")))
    if not qsm_files:
        return []

    log.info(f"\n  Processing {subj_id}/{sess_id}")

    # Step 1: Get segmentation
    seg_data = None
    seg_affine = None
    if method in ("atlas", "hybrid"):
        # Look for existing SynthSeg output: try subj/sess, then subj-level
        seg_candidates = [
            os.path.join(SYNTHSEG_DIR, subj_id, sess_id, "synthseg.nii.gz"),
            os.path.join(SYNTHSEG_DIR, subj_id, "synthseg.nii.gz"),
        ]
        seg_path = None
        for cand in seg_candidates:
            if os.path.exists(cand):
                seg_path = cand
                break

        # If no existing seg and not skipping, run SynthSeg
        if seg_path is None and not skip_synthseg:
            input_img, input_type = find_input_image(qsm_dir, sess_dir)
            if input_img:
                synthseg_out = os.path.join(SYNTHSEG_DIR, subj_id, sess_id)
                seg_path = run_synthseg(input_img, synthseg_out)

        if seg_path and os.path.exists(seg_path):
            seg_img = nib.load(seg_path)
            seg_data = np.asanyarray(seg_img.dataobj).astype(np.int32)
            seg_affine = seg_img.affine
            log.info(f"    Loaded segmentation: {seg_data.shape}, "
                     f"{len(np.unique(seg_data))} labels from {seg_path}")

    # Step 2: Brain mask and centroid from magnitude
    mag1_path = None
    for pat in ["mag_10echo_run-01_e1.nii.gz", "mag_5echo_run-01_e1.nii.gz"]:
        cand = os.path.join(qsm_dir, pat)
        if os.path.exists(cand):
            mag1_path = cand
            break
    if mag1_path is None:
        cands = sorted(glob.glob(os.path.join(qsm_dir, "mag_*_e1.nii.gz")))
        if cands:
            mag1_path = cands[0]

    brain_mask = None
    com_world = None
    if mag1_path:
        try:
            mi = nib.load(mag1_path)
            md = mi.get_fdata()
            if md.ndim == 4:
                md = md[:, :, :, 0]
            brain_mask = create_brain_mask(md, threshold_pct=15)
            com_world, _ = compute_brain_centroid(brain_mask, mi.affine)
        except Exception as e:
            log.warning(f"    Magnitude mask failed: {e}")

    # Step 3: Extract ROI values from each QSM file
    all_results = []
    fs = 3.0 if "3T" in sess_id else 5.0

    for qf in qsm_files:
        qb = os.path.basename(qf)
        et = "10echo" if "10echo" in qb else "5echo"
        sn = "run-02" if "run-02" in qb else "run-01"

        try:
            qi = nib.load(qf)
            qd = qi.get_fdata()
            qa = qi.affine
            if qd.ndim > 3:
                qd = qd[:, :, :, 0]

            # Use brain_mask if shapes match
            if brain_mask is not None and brain_mask.shape == qd.shape[:3]:
                bm = brain_mask
            else:
                bm = (np.abs(qd) > 0).astype(bool)

            if com_world is None:
                com_world, _ = compute_brain_centroid(bm, qa)

            # Atlas-based extraction
            if seg_data is not None and seg_data.shape == qd.shape[:3]:
                if granularity == "fine":
                    # Every individual aseg label
                    for label_id, (label_name, group) in ASEG_LABELS.items():
                        res = extract_roi_from_atlas(qd, seg_data, bm, [label_id], group)
                        if res:
                            all_results.append({
                                "subject": subj_id, "session": sess_id,
                                "field_strength": fs, "echo_type": et,
                                "scan": sn, "method": "atlas",
                                "roi": group, "aseg_label": label_id,
                                "n_voxels": res["n_voxels"],
                                "mean_ppb": f"{res['mean']:.4f}",
                                "std_ppb": f"{res['std']:.4f}",
                                "median_ppb": f"{res['median']:.4f}",
                            })

                elif granularity == "medium":
                    # Grouped anatomical regions
                    for group_name, label_ids in MEDIUM_GROUPS.items():
                        res = extract_roi_from_atlas(qd, seg_data, bm, label_ids, group_name)
                        if res:
                            all_results.append({
                                "subject": subj_id, "session": sess_id,
                                "field_strength": fs, "echo_type": et,
                                "scan": sn, "method": "atlas",
                                "roi": group_name,
                                "n_voxels": res["n_voxels"],
                                "mean_ppb": f"{res['mean']:.4f}",
                                "std_ppb": f"{res['std']:.4f}",
                                "median_ppb": f"{res['median']:.4f}",
                            })

                else:  # coarse — standard 8 ROIs
                    # Atlas for: Caudate, Putamen, GP, Thalamus
                    atlas_rois = {
                        "Caudate_L": [11], "Caudate_R": [50],
                        "Putamen_L": [12], "Putamen_R": [51],
                        "GP_L": [13], "GP_R": [52],
                        "Thalamus_L": [10], "Thalamus_R": [49],
                    }
                    for roi_name, label_ids in atlas_rois.items():
                        res = extract_roi_from_atlas(qd, seg_data, bm, label_ids, roi_name)
                        if res:
                            all_results.append({
                                "subject": subj_id, "session": sess_id,
                                "field_strength": fs, "echo_type": et,
                                "scan": sn, "method": "atlas",
                                "roi": roi_name,
                                "n_voxels": res["n_voxels"],
                                "mean_ppb": f"{res['mean']:.4f}",
                                "std_ppb": f"{res['std']:.4f}",
                                "median_ppb": f"{res['median']:.4f}",
                            })

                    # Centroid fallback for: RN, SN, DN, FrontalWM
                    centroid_rois = ["RN_L","RN_R","SN_L","SN_R","DN_L","DN_R",
                                     "FrontalWM_L","FrontalWM_R"]
                    for roi_name in centroid_rois:
                        if roi_name in CENTROID_ROI_DEFS:
                            res = extract_roi_centroid(
                                qd, bm, qa, com_world,
                                CENTROID_ROI_DEFS[roi_name], roi_name)
                            if res:
                                all_results.append({
                                    "subject": subj_id, "session": sess_id,
                                    "field_strength": fs, "echo_type": et,
                                    "scan": sn, "method": "centroid",
                                    "roi": roi_name,
                                    "n_voxels": res["n_voxels"],
                                    "mean_ppb": f"{res['mean']:.4f}",
                                    "std_ppb": f"{res['std']:.4f}",
                                    "median_ppb": f"{res['median']:.4f}",
                                })

            else:
                # Pure centroid method
                for roi_name, roi_cfg in CENTROID_ROI_DEFS.items():
                    res = extract_roi_centroid(qd, bm, qa, com_world, roi_cfg, roi_name)
                    if res:
                        all_results.append({
                            "subject": subj_id, "session": sess_id,
                            "field_strength": fs, "echo_type": et,
                            "scan": sn, "method": "centroid",
                            "roi": roi_name,
                            "n_voxels": res["n_voxels"],
                            "mean_ppb": f"{res['mean']:.4f}",
                            "std_ppb": f"{res['std']:.4f}",
                            "median_ppb": f"{res['median']:.4f}",
                        })

            n_ok = len([r for r in all_results if r.get("scan") == sn])
            log.info(f"    {qb}: extracted {n_ok} ROIs")

        except Exception as e:
            log.error(f"    {qb}: ERROR {e}")
            import traceback
            traceback.print_exc()

    return all_results


def compute_bilateral_averages(results):
    """Merge left/right ROI values into bilateral averages."""
    bilateral = []
    groups = {}
    for r in results:
        k = (r["subject"], r["session"], r["echo_type"], r["scan"])
        groups.setdefault(k, {})[r["roi"]] = r

    for key, rois in groups.items():
        subj, sess, et, sn = key
        fs = 3.0 if "3T" in sess else 5.0
        for bn, (ln, rn) in BILATERAL_ROIS.items():
            if ln in rois and rn in rois:
                try:
                    lm = float(rois[ln]["mean_ppb"])
                    rm = float(rois[rn]["mean_ppb"])
                    ls = float(rois[ln]["std_ppb"])
                    rs = float(rois[rn]["std_ppb"])
                    lnv = int(rois[ln]["n_voxels"])
                    rnv = int(rois[rn]["n_voxels"])
                    total = lnv + rnv
                    if total == 0:
                        continue
                    bm = (lm * lnv + rm * rnv) / total
                    bs = np.sqrt((ls ** 2 * lnv + rs ** 2 * rnv) / total)
                    bmed = (float(rois[ln].get("median_ppb", 0)) * lnv +
                            float(rois[rn].get("median_ppb", 0)) * rnv) / total
                    method = rois[ln].get("method", "unknown")
                    bilateral.append({
                        "subject": subj, "session": sess,
                        "field_strength": fs, "echo_type": et,
                        "scan": sn, "method": method, "roi": bn,
                        "n_voxels": total,
                        "mean_ppb": f"{bm:.4f}",
                        "std_ppb": f"{bs:.4f}",
                        "median_ppb": f"{bmed:.4f}",
                    })
                except (ValueError, TypeError):
                    pass
    return bilateral


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Atlas-based ROI analysis for QSM")
    parser.add_argument("--method", choices=["atlas", "centroid", "hybrid"],
                        default="hybrid",
                        help="ROI method: atlas (SynthSeg), centroid (legacy), hybrid (default)")
    parser.add_argument("--granularity", choices=["fine", "medium", "coarse"],
                        default="coarse",
                        help="ROI granularity: fine (~100 labels), medium (~16), coarse (8)")
    parser.add_argument("--skip-synthseg", action="store_true",
                        help="Skip SynthSeg, use existing segmentations only")
    args = parser.parse_args()

    log.info("=" * 60)
    log.info(f"ROI Analysis v2 — method={args.method}, granularity={args.granularity}")
    log.info("=" * 60)

    all_results = []

    for subj_dir in sorted(glob.glob(os.path.join(NIFTI_ROOT, "sub-*"))):
        subj_id = os.path.basename(subj_dir)
        for sess_dir in sorted(glob.glob(os.path.join(subj_dir, "ses-*"))):
            sess_id = os.path.basename(sess_dir)
            results = analyze_single_session(
                subj_id, sess_id, sess_dir,
                method=args.method,
                granularity=args.granularity,
                skip_synthseg=args.skip_synthseg,
            )
            all_results.extend(results)

    if not all_results:
        log.warning("No ROI data extracted. Check your NIfTI directory.")
        return

    # Save individual results
    fieldnames = ["subject", "session", "field_strength", "echo_type",
                  "scan", "method", "roi", "n_voxels",
                  "mean_ppb", "std_ppb", "median_ppb"]
    if all_results and "aseg_label" in all_results[0]:
        fieldnames.insert(7, "aseg_label")

    csv_path = os.path.join(ROI_OUTPUT, "roi_values_individual.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        w.writeheader()
        w.writerows(all_results)
    log.info(f"\nIndividual ROI values: {csv_path} ({len(all_results)} entries)")

    # Bilateral averages
    bilateral = compute_bilateral_averages(all_results)
    if bilateral:
        bi_fields = ["subject", "session", "field_strength", "echo_type",
                      "scan", "method", "roi", "n_voxels",
                      "mean_ppb", "std_ppb", "median_ppb"]
        bi_path = os.path.join(ROI_OUTPUT, "roi_values_bilateral.csv")
        with open(bi_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=bi_fields, extrasaction='ignore')
            w.writeheader()
            w.writerows(bilateral)
        log.info(f"Bilateral ROI values: {bi_path} ({len(bilateral)} entries)")

    # Summary table
    log.info("\n" + "=" * 80)
    log.info("ROI SUMMARY (10-echo, run-01, bilateral means in ppb)")
    log.info("=" * 80)
    sd_3t, sd_5t = {}, {}
    for r in bilateral:
        if r["echo_type"] != "10echo" or r["scan"] != "run-01":
            continue
        try:
            v = float(r["mean_ppb"])
        except ValueError:
            continue
        if float(r["field_strength"]) == 3.0:
            sd_3t.setdefault(r["roi"], []).append(v)
        else:
            sd_5t.setdefault(r["roi"], []).append(v)

    header = f"{'ROI':<12} {'3T N':>5} {'3T Mean±SD':>16} {'5T N':>5} {'5T Mean±SD':>16}"
    log.info(header)
    log.info("-" * 60)
    for roi in ROI_ORDER:
        v3 = np.array(sd_3t.get(roi, []))
        v5 = np.array(sd_5t.get(roi, []))
        c3 = f"{np.mean(v3):>6.1f} ± {np.std(v3):<6.1f}" if len(v3) else "    -"
        c5 = f"{np.mean(v5):>6.1f} ± {np.std(v5):<6.1f}" if len(v5) else "    -"
        log.info(f"{roi:<12} {len(v3):>5} {c3:>16} {len(v5):>5} {c5:>16}")

    log.info("\nROI analysis v2 complete!")


if __name__ == "__main__":
    main()
