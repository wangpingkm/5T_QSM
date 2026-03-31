#!/usr/bin/env python3
"""
Generate Supplement Figure 2 v1 — single-page montage
======================================================
For each subject/session pair that has a Seg_DN_Basal_Ganglia segmentation,
produce a **single-page** PDF showing only the z-slices that contain
segmentation data.  QSM is shown in grayscale; each basal-ganglia / deep-
nuclei ROI is drawn as a semi-transparent colour fill (alpha = 0.20).
A colour legend at the bottom identifies the 7 structures.

Output directory:  output/supplement figure 2/v1/
One PDF per subject-session:  <sub>_<ses>_QSM_BG.pdf
"""

import os, math
import numpy as np
import nibabel as nib
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT = Path(__file__).resolve().parent.parent
NIFTI_ROOT = Path("/path/to/project/"
                   "Reject and Resubmit/Protocal/NIfTI")
REG_DIR = PROJECT / "output" / "roi_analysis" / "registered"
OUT_DIR = PROJECT / "output" / "supplement figure 2" / "v1"
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

# Grid layout (columns; rows computed dynamically from # slices with data)
N_COLS = 8

# ---------------------------------------------------------------------------
# Label table  (from gen_overlays.py)
# ---------------------------------------------------------------------------
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

# 7 structure colours — L and R share the same colour
_STRUCT_COLORS = {
    "Dentate_nucleus":  (0.00, 0.74, 0.83),   # cyan
    "Substantia_nigra": (0.96, 0.26, 0.21),   # red
    "Nucleus_ruber":    (0.30, 0.69, 0.31),   # green
    "Thalamus":         (0.61, 0.15, 0.69),   # purple
    "Caudate_nucleus":  (0.98, 0.92, 0.22),   # yellow
    "Putamen":          (0.13, 0.59, 0.95),   # blue
    "Globus_pallidus":  (1.00, 0.60, 0.00),   # orange
}

LABEL_COLORS = {
    1:  _STRUCT_COLORS["Dentate_nucleus"],
    2:  _STRUCT_COLORS["Dentate_nucleus"],
    3:  _STRUCT_COLORS["Substantia_nigra"],
    4:  _STRUCT_COLORS["Substantia_nigra"],
    5:  _STRUCT_COLORS["Nucleus_ruber"],
    6:  _STRUCT_COLORS["Nucleus_ruber"],
    7:  _STRUCT_COLORS["Thalamus"],
    8:  _STRUCT_COLORS["Thalamus"],
    9:  _STRUCT_COLORS["Caudate_nucleus"],
    10: _STRUCT_COLORS["Caudate_nucleus"],
    11: _STRUCT_COLORS["Putamen"],
    12: _STRUCT_COLORS["Putamen"],
    13: _STRUCT_COLORS["Globus_pallidus"],
    14: _STRUCT_COLORS["Globus_pallidus"],
}

_STRUCT_LEGEND_LABEL = {
    "Dentate_nucleus":  "Dentate nucleus",
    "Substantia_nigra": "Substantia nigra",
    "Nucleus_ruber":    "Red nucleus",
    "Thalamus":         "Thalamus",
    "Caudate_nucleus":  "Caudate nucleus",
    "Putamen":          "Putamen",
    "Globus_pallidus":  "Globus pallidus",
}


# ---------------------------------------------------------------------------
# Helper: paths
# ---------------------------------------------------------------------------
def qsm_path(sub, ses):
    return NIFTI_ROOT / sub / ses / "qsm" / "qsm_10echo_run-01.nii.gz"


def seg_path(sub, ses):
    return REG_DIR / sub / ses / "Seg_DN_Basal_Ganglia.nii.gz"


# ---------------------------------------------------------------------------
# Discover all valid subject / session pairs
# ---------------------------------------------------------------------------
def discover_pairs():
    pairs = []
    for s in sorted(REG_DIR.glob("sub-*/ses-*/Seg_DN_Basal_Ganglia.nii.gz")):
        sub = s.parent.parent.name
        ses = s.parent.name
        if qsm_path(sub, ses).exists():
            pairs.append((sub, ses))
    return pairs


# ---------------------------------------------------------------------------
# Generate one single-page PDF for a subject / session
# ---------------------------------------------------------------------------
def generate_pdf(sub, ses, pair_idx, total_pairs):
    print(f"  [{pair_idx+1}/{total_pairs}]  {sub} / {ses} ... ",
          end="", flush=True)

    qsm_vol = nib.load(str(qsm_path(sub, ses))).get_fdata(dtype=np.float32)
    seg_vol = nib.load(str(seg_path(sub, ses))).get_fdata().astype(np.int32)
    if qsm_vol.ndim > 3:
        qsm_vol = qsm_vol[:, :, :, 0]
    if seg_vol.ndim > 3:
        seg_vol = seg_vol[:, :, :, 0]

    # Only keep z-slices that contain segmentation
    z_with_data = [z for z in range(seg_vol.shape[2])
                   if np.any(seg_vol[:, :, z] > 0)]
    if not z_with_data:
        print("SKIP (no seg data)")
        return

    n_slices = len(z_with_data)
    n_rows = math.ceil(n_slices / N_COLS)

    # Aspect ratio of one slice (after transpose: rows=dim1, cols=dim0)
    slice_aspect = qsm_vol.shape[1] / qsm_vol.shape[0]  # 504/477 ≈ 1.057

    # Figure sizing
    cell_w = 2.0   # slightly larger cells since fewer slices
    cell_h = cell_w * slice_aspect
    fig_w = cell_w * N_COLS
    legend_h = 0.6
    fig_h = cell_h * n_rows + 0.6 + legend_h  # +0.6 suptitle, +legend

    fig, axes = plt.subplots(n_rows, N_COLS, figsize=(fig_w, fig_h))
    axes = np.atleast_2d(axes)

    fig.suptitle(f"{sub}   {ses}", fontsize=8, fontweight="bold", y=1.0)

    for idx_z, z in enumerate(z_with_data):
        row = idx_z // N_COLS
        col = idx_z % N_COLS
        ax = axes[row, col]

        qsm_slice = qsm_vol[:, :, z].T
        seg_slice = seg_vol[:, :, z].T

        # Grayscale QSM background
        ax.imshow(qsm_slice, cmap="gray", vmin=QSM_VMIN, vmax=QSM_VMAX,
                  origin="lower", aspect="equal", interpolation="bilinear")

        # --- filled colour overlays (same as gen_overlays.py) ---
        labels_in_slice = np.unique(seg_slice)
        labels_in_slice = labels_in_slice[labels_in_slice > 0]
        for lbl in labels_in_slice:
            color = LABEL_COLORS.get(int(lbl), (1.0, 1.0, 1.0))
            mask = (seg_slice == lbl).astype(np.float64)
            if mask.sum() < 2:
                continue
            rgba = np.zeros((*mask.shape, 4))
            rgba[..., :3] = color
            rgba[..., 3] = mask * 0.20
            ax.imshow(rgba, origin="lower", aspect="equal",
                      interpolation="nearest")

        # Slice label
        ax.text(0.02, 0.96, f"z = {z}", transform=ax.transAxes,
                fontsize=3, color="white", va="top", ha="left")
        ax.set_xticks([])
        ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)

    # Hide unused cells
    for idx_z in range(n_slices, n_rows * N_COLS):
        row = idx_z // N_COLS
        col = idx_z % N_COLS
        axes[row, col].set_visible(False)

    # --- Colour legend (one entry per structure, no L/R duplication) ---
    present_labels = set(int(l) for l in np.unique(seg_vol) if l > 0)
    handles = []
    for struct, label_text in _STRUCT_LEGEND_LABEL.items():
        color = _STRUCT_COLORS[struct]
        struct_ids = [lid for lid, name in LABEL_MAP.items()
                      if name.rsplit("_", 1)[0] == struct]
        if any(sid in present_labels for sid in struct_ids):
            handles.append(mpatches.Patch(
                facecolor=color, edgecolor="white", linewidth=0.3,
                alpha=0.85, label=label_text))

    fig.subplots_adjust(left=0, right=1, bottom=0.04, top=0.97,
                        wspace=0.02, hspace=0.02)

    if handles:
        fig.legend(handles=handles,
                   loc="lower center",
                   ncol=len(handles),
                   bbox_to_anchor=(0.5, 0.0),
                   bbox_transform=fig.transFigure,
                   fontsize=5,
                   framealpha=0.3,
                   facecolor="white",
                   edgecolor="#999999",
                   labelcolor="black",
                   handleheight=1.0,
                   handlelength=1.5,
                   borderpad=0.5,
                   columnspacing=1.0)

    pdf_name = f"{sub}_{ses}_QSM_BG.pdf"
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
    print("Supplement Figure 2 v1 — Seg_DN_Basal_Ganglia montage (grayscale)")
    print("=" * 65)

    pairs = discover_pairs()
    print(f"\nFound {len(pairs)} subject/session pairs.\n")

    for i, (sub, ses) in enumerate(pairs):
        generate_pdf(sub, ses, i, len(pairs))

    print(f"\nAll PDFs saved in: {OUT_DIR}")
    print("Done!")


if __name__ == "__main__":
    main()
