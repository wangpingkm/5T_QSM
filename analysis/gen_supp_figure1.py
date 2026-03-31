#!/usr/bin/env python3
"""
Generate Supplement Figure 1 v3 — single-page montage
======================================================
For each subject/session pair, produce a **single-page** PDF with all axial
z-slices tiled in an 8-column × 14-row grid (112 slices).  QSM is shown in
grayscale; synthseg ROI boundaries are drawn as thin dotted contour lines.

Output directory:  output/supplement figure 1/v3/
One PDF per subject-session:  <sub>_<ses>_QSM_synthseg.pdf
"""

import os, math
import numpy as np
import nibabel as nib
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT = Path(__file__).resolve().parent.parent
NIFTI_ROOT = Path("/path/to/project/"
                   "Reject and Resubmit/Protocal/NIfTI")
REG_DIR = PROJECT / "output" / "roi_analysis" / "registered"
OUT_DIR = PROJECT / "output" / "supplement figure 1" / "v3"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------
plt.rcParams.update({
    "font.family": "Arial",
    "font.size": 4,
    "figure.dpi": 200,
    "savefig.dpi": 200,
})

# ---------------------------------------------------------------------------
# Display constants
# ---------------------------------------------------------------------------
QSM_VMIN = -118
QSM_VMAX = 150

# Grid layout: 8 columns × 14 rows = 112 cells
N_COLS = 8
N_ROWS = 14

# ---------------------------------------------------------------------------
# Distinct colours for contour lines (cycled if > len)
# ---------------------------------------------------------------------------
CONTOUR_COLORS = [
    "#FF0000", "#00FF00", "#0000FF", "#FFFF00", "#FF00FF", "#00FFFF",
    "#FF8000", "#8000FF", "#00FF80", "#FF0080", "#80FF00", "#0080FF",
    "#FF4040", "#40FF40", "#4040FF", "#FFAA00", "#AA00FF", "#00FFAA",
    "#FF6060", "#60FF60", "#6060FF", "#FFCC00", "#CC00FF", "#00FFCC",
]


def qsm_path(sub, ses):
    return NIFTI_ROOT / sub / ses / "qsm" / "qsm_10echo_run-01.nii.gz"


def synthseg_path(sub, ses):
    return REG_DIR / sub / ses / "synthseg_in_qsm.nii.gz"


# ---------------------------------------------------------------------------
# Discover all valid subject / session pairs
# ---------------------------------------------------------------------------
def discover_pairs():
    pairs = []
    for syn in sorted(REG_DIR.glob("sub-*/ses-*/synthseg_in_qsm.nii.gz")):
        sub = syn.parent.parent.name
        ses = syn.parent.name
        qsm = qsm_path(sub, ses)
        if qsm.exists():
            pairs.append((sub, ses))
    return pairs


# ---------------------------------------------------------------------------
# Generate one single-page PDF for a subject / session
# ---------------------------------------------------------------------------
def generate_pdf(sub, ses, pair_idx, total_pairs):
    print(f"  [{pair_idx+1}/{total_pairs}]  {sub} / {ses} ... ", end="", flush=True)

    qsm_vol = nib.load(str(qsm_path(sub, ses))).get_fdata(dtype=np.float32)
    seg_vol = nib.load(str(synthseg_path(sub, ses))).get_fdata().astype(np.int32)
    if qsm_vol.ndim > 3:
        qsm_vol = qsm_vol[:, :, :, 0]
    if seg_vol.ndim > 3:
        seg_vol = seg_vol[:, :, :, 0]

    n_slices = qsm_vol.shape[2]
    n_rows = math.ceil(n_slices / N_COLS)

    # Aspect ratio of one slice (after transpose: rows=504, cols=477)
    slice_aspect = qsm_vol.shape[1] / qsm_vol.shape[0]  # 504/477 ≈ 1.057

    # Figure sizing: each sub-plot cell ~1.6 in wide
    cell_w = 1.6
    cell_h = cell_w * slice_aspect
    fig_w = cell_w * N_COLS
    fig_h = cell_h * n_rows + 0.6  # +0.6 for suptitle

    fig, axes = plt.subplots(n_rows, N_COLS, figsize=(fig_w, fig_h))
    axes = np.atleast_2d(axes)

    fig.suptitle(f"{sub}   {ses}", fontsize=8, fontweight="bold", y=1.0)

    for z in range(n_slices):
        row = z // N_COLS
        col = z % N_COLS
        ax = axes[row, col]

        qsm_slice = qsm_vol[:, :, z].T
        seg_slice = seg_vol[:, :, z].T

        ax.imshow(qsm_slice, cmap="gray", vmin=QSM_VMIN, vmax=QSM_VMAX,
                  origin="lower", aspect="equal", interpolation="bilinear")

        # --- synthseg contours ---
        labels_in_slice = np.unique(seg_slice)
        labels_in_slice = labels_in_slice[labels_in_slice > 0]
        for idx, lbl in enumerate(labels_in_slice):
            mask = (seg_slice == lbl).astype(np.float64)
            if mask.sum() < 2:
                continue
            color = CONTOUR_COLORS[idx % len(CONTOUR_COLORS)]
            ax.contour(mask, levels=[0.5], colors=[color],
                       linewidths=0.2, linestyles="dotted",
                       origin="lower")

        # Slice label
        ax.text(0.02, 0.96, f"z = {z}", transform=ax.transAxes,
                fontsize=3, color="white", va="top", ha="left")
        ax.set_xticks([])
        ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)

    # Hide unused cells (if n_slices < n_rows * N_COLS)
    for z in range(n_slices, n_rows * N_COLS):
        row = z // N_COLS
        col = z % N_COLS
        axes[row, col].set_visible(False)

    fig.subplots_adjust(left=0, right=1, bottom=0, top=0.97,
                        wspace=0.02, hspace=0.02)

    pdf_name = f"{sub}_{ses}_QSM_synthseg.pdf"
    pdf_path = OUT_DIR / pdf_name
    fig.savefig(str(pdf_path), bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)

    size_kb = os.path.getsize(pdf_path) / 1024
    print(f"1 page, {n_slices} slices, {size_kb:.0f} KB")


# ===========================================================================
# Main
# ===========================================================================
def main():
    print("=" * 65)
    print("Supplement Figure 1 v3 — single-page montage (grayscale)")
    print("=" * 65)

    pairs = discover_pairs()
    print(f"\nFound {len(pairs)} subject/session pairs.\n")

    for i, (sub, ses) in enumerate(pairs):
        generate_pdf(sub, ses, i, len(pairs))

    print(f"\nAll PDFs saved in: {OUT_DIR}")
    print("Done!")


if __name__ == "__main__":
    main()
