#!/usr/bin/env python3
"""
Generate Figure 2 v4 — Segmentation comparison (SynthSeg vs Manual)
====================================================================
Subject:  sub-07Huang / ses-5TUIH1
Layout:   5 columns × 2 rows
  - Top row:    SynthSeg — thin dotted contour lines
  - Bottom row: Manual segmentation — filled colour overlays

Changes from v3 → v4:
  1. Column titles (anatomy + Z) and row labels moved INSIDE images
     as white text — external margins removed for tighter layout
  2. Both legends merged into ONE unified legend box
  3. Legend is placed as close as possible to the bottom images

Slices: Z = 8, 33, 40, 56, 74
Output: output/Figure2/Figure 2 v4.pdf
"""

import os
import numpy as np
import nibabel as nib
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines

# ── Paths ──────────────────────────────────────────────────────────────────
PROJECT    = Path(__file__).resolve().parent.parent
NIFTI_ROOT = Path("/path/to/project/"
                   "Reject and Resubmit/Protocal/NIfTI")
REG_DIR    = PROJECT / "output" / "roi_analysis" / "registered"
OUT_DIR    = PROJECT / "output" / "Figure2"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Subject / session ─────────────────────────────────────────────────────
SUB = "sub-07Huang"
SES = "ses-5TUIH1"

# ── Slices and anatomical labels ──────────────────────────────────────────
SLICES = [8, 33, 40, 56, 74]
SLICE_ANATOMY = {
    8:  "Cerebellum",
    33: "Midbrain",
    40: "Basal ganglia",
    56: "Internal capsule",
    74: "Corona radiata",
}

# ── Display constants ─────────────────────────────────────────────────────
QSM_VMIN = -118
QSM_VMAX =  150
OVERLAY_ALPHA = 0.35
CONTOUR_LW    = 0.3
CONTOUR_STYLE = "dotted"

# ── Style ─────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":      "Arial",
    "font.size":        8,
    "axes.titlesize":   8,
    "axes.titleweight": "bold",
    "axes.labelsize":   8,
    "figure.dpi":       300,
    "savefig.dpi":      300,
})

# =========================================================================
# SynthSeg colour scheme — contour lines
# =========================================================================
_SS_STRUCT_COLORS = {
    "Cerebral WM":       (0.70, 0.70, 0.70),
    "Ventricles":        (0.20, 0.60, 0.90),
    "CSF":               (0.40, 0.75, 0.95),
    "Cerebellum WM":     (0.55, 0.55, 0.55),
    "Cerebellum cortex": (0.85, 0.45, 0.55),
    "Brain stem":        (0.60, 0.40, 0.70),
    "Thalamus":          (0.61, 0.15, 0.69),
    "Caudate":           (0.98, 0.92, 0.22),
    "Putamen":           (0.13, 0.59, 0.95),
    "Globus pallidus":   (1.00, 0.60, 0.00),
    "Hippocampus":       (0.30, 0.69, 0.31),
    "Amygdala":          (0.96, 0.26, 0.21),
    "Accumbens":         (0.00, 0.74, 0.83),
    "Ventral DC":        (0.55, 0.35, 0.17),
}

_CORTICAL_COLOR = (1.00, 0.65, 0.00)

_SS_ID_TO_STRUCT = {
    2:  "Cerebral WM",       41: "Cerebral WM",
    4:  "Ventricles",        43: "Ventricles",
    5:  "Ventricles",        44: "Ventricles",
    14: "Ventricles",        15: "Ventricles",
    24: "CSF",
    7:  "Cerebellum WM",     46: "Cerebellum WM",
    8:  "Cerebellum cortex", 47: "Cerebellum cortex",
    16: "Brain stem",
    10: "Thalamus",          49: "Thalamus",
    11: "Caudate",           50: "Caudate",
    12: "Putamen",           51: "Putamen",
    13: "Globus pallidus",   52: "Globus pallidus",
    17: "Hippocampus",       53: "Hippocampus",
    18: "Amygdala",          54: "Amygdala",
    26: "Accumbens",         58: "Accumbens",
    28: "Ventral DC",        60: "Ventral DC",
}

def _ss_color_for_label(lid):
    if lid in _SS_ID_TO_STRUCT:
        return _SS_STRUCT_COLORS[_SS_ID_TO_STRUCT[lid]]
    if (1001 <= lid <= 1035) or (2001 <= lid <= 2035):
        return _CORTICAL_COLOR
    return None

# =========================================================================
# Manual seg colour scheme
# =========================================================================
_MAN_STRUCT_COLORS = {
    "Dentate nucleus":   (0.00, 0.74, 0.83),
    "Substantia nigra":  (0.96, 0.26, 0.21),
    "Red nucleus":       (0.30, 0.69, 0.31),
    "Thalamus":          (0.61, 0.15, 0.69),
    "Caudate nucleus":   (0.98, 0.92, 0.22),
    "Putamen":           (0.13, 0.59, 0.95),
    "Globus pallidus":   (1.00, 0.60, 0.00),
}

MAN_LABEL_MAP = {
    1:  ("Dentate nucleus",  "R"),  2:  ("Dentate nucleus",  "L"),
    3:  ("Substantia nigra", "R"),  4:  ("Substantia nigra", "L"),
    5:  ("Red nucleus",      "R"),  6:  ("Red nucleus",      "L"),
    7:  ("Thalamus",         "R"),  8:  ("Thalamus",         "L"),
    9:  ("Caudate nucleus",  "R"),  10: ("Caudate nucleus",  "L"),
    11: ("Putamen",          "R"),  12: ("Putamen",          "L"),
    13: ("Globus pallidus",  "R"),  14: ("Globus pallidus",  "L"),
}

MAN_LABEL_COLORS = {}
for _lid, (_struct, _) in MAN_LABEL_MAP.items():
    MAN_LABEL_COLORS[_lid] = _MAN_STRUCT_COLORS[_struct]


# =========================================================================
# Helpers
# =========================================================================
def load_vol(path):
    d = nib.load(str(path)).get_fdata(dtype=np.float32)
    return d[:, :, :, 0] if d.ndim > 3 else d


def draw_contours(ax, seg_slice, color_func):
    labels = np.unique(seg_slice)
    labels = labels[labels > 0]
    for lbl in labels:
        lid = int(lbl)
        c = color_func(lid)
        if c is None:
            continue
        mask = (seg_slice == lbl).astype(np.float64)
        if mask.sum() < 2:
            continue
        ax.contour(mask, levels=[0.5], colors=[c],
                   linewidths=CONTOUR_LW, linestyles=CONTOUR_STYLE,
                   origin="lower")


def draw_filled_overlay(ax, seg_slice, label_colors, alpha=OVERLAY_ALPHA):
    labels = np.unique(seg_slice)
    labels = labels[labels > 0]
    for lbl in labels:
        lid = int(lbl)
        if lid not in label_colors:
            continue
        color = label_colors[lid]
        mask = (seg_slice == lbl).astype(np.float64)
        if mask.sum() < 2:
            continue
        rgba = np.zeros((*mask.shape, 4))
        rgba[..., :3] = color
        rgba[..., 3] = mask * alpha
        ax.imshow(rgba, origin="lower", aspect="equal",
                  interpolation="nearest")


def style_ax(ax):
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)


# =========================================================================
# Build figure
# =========================================================================
def build_figure():
    qsm_path = NIFTI_ROOT / SUB / SES / "qsm" / "qsm_10echo_run-01.nii.gz"
    ss_path  = REG_DIR / SUB / SES / "synthseg_in_qsm.nii.gz"
    bg_path  = REG_DIR / SUB / SES / "Seg_DN_Basal_Ganglia.nii.gz"

    qsm_vol = load_vol(qsm_path)
    ss_vol  = nib.load(str(ss_path)).get_fdata().astype(np.int32)
    bg_vol  = nib.load(str(bg_path)).get_fdata().astype(np.int32)
    if ss_vol.ndim > 3:
        ss_vol = ss_vol[:, :, :, 0]
    if bg_vol.ndim > 3:
        bg_vol = bg_vol[:, :, :, 0]

    n_cols = len(SLICES)   # 5
    n_rows = 2

    slice_h, slice_w = qsm_vol.shape[1], qsm_vol.shape[0]   # 504, 477
    img_aspect = slice_h / slice_w

    # ── Layout — no external margins for titles/row labels ─────────────
    cell_w = 2.0
    cell_h = cell_w * img_aspect
    gap    = 0.02              # very small gap between images
    legend_gap = 0.06          # tiny gap between images and legend
    legend_h   = 0.65          # space for legend (2-3 rows, larger font)

    content_w = n_cols * cell_w + (n_cols - 1) * gap
    content_h = n_rows * cell_h + gap
    fig_w = content_w
    fig_h = content_h + legend_gap + legend_h

    fig = plt.figure(figsize=(fig_w, fig_h), facecolor="white")

    # Fractional sizes
    cw_frac = cell_w / fig_w
    ch_frac = cell_h / fig_h
    gx_frac = gap / fig_w
    gy_frac = gap / fig_h

    # Top of content = top of figure
    top_y = 1.0

    # Tracking
    ss_present_structs  = set()
    ss_has_cortical     = False
    man_present_structs = set()

    axes_grid = {}   # (row, col) -> ax, for placing labels after

    # ── Draw image panels ──────────────────────────────────────────────
    for col_idx, z in enumerate(SLICES):
        for row_idx in range(n_rows):
            x0 = col_idx * (cw_frac + gx_frac)
            y0 = top_y - (row_idx + 1) * ch_frac - row_idx * gy_frac

            ax = fig.add_axes([x0, y0, cw_frac, ch_frac])
            axes_grid[(row_idx, col_idx)] = ax

            qsm_slice = qsm_vol[:, :, z].T
            ax.imshow(qsm_slice, cmap="gray", vmin=QSM_VMIN, vmax=QSM_VMAX,
                      origin="lower", aspect="equal", interpolation="bilinear")

            if row_idx == 0:
                seg_slice = ss_vol[:, :, z].T
                draw_contours(ax, seg_slice, _ss_color_for_label)
                for lbl in np.unique(seg_slice):
                    lid = int(lbl)
                    if lid <= 0:
                        continue
                    if lid in _SS_ID_TO_STRUCT:
                        ss_present_structs.add(_SS_ID_TO_STRUCT[lid])
                    elif (1001 <= lid <= 1035) or (2001 <= lid <= 2035):
                        ss_has_cortical = True
            else:
                seg_slice = bg_vol[:, :, z].T
                labels_here = np.unique(seg_slice)
                labels_here = labels_here[labels_here > 0]
                if len(labels_here) > 0:
                    draw_filled_overlay(ax, seg_slice, MAN_LABEL_COLORS)
                    for lbl in labels_here:
                        lid = int(lbl)
                        if lid in MAN_LABEL_MAP:
                            man_present_structs.add(MAN_LABEL_MAP[lid][0])

            style_ax(ax)

    # ── Inside labels: column titles on TOP-ROW images ─────────────────
    for col_idx, z in enumerate(SLICES):
        ax = axes_grid[(0, col_idx)]
        anatomy = SLICE_ANATOMY[z]
        ax.text(0.50, 0.97, f"{anatomy} (Z = {z})",
                transform=ax.transAxes, ha="center", va="top",
                fontsize=6.5, fontweight="bold", color="white")

    # ── Inside labels: row labels on LEFT-COLUMN images ────────────────
    row_labels = ["SynthSeg", "Manual segmentation"]
    for row_idx, label in enumerate(row_labels):
        ax = axes_grid[(row_idx, 0)]
        ax.text(0.03, 0.50, label,
                transform=ax.transAxes, ha="left", va="center",
                fontsize=6.5, fontweight="bold", color="white",
                rotation=90)

    # ── Unified legend ─────────────────────────────────────────────────
    # Build SynthSeg handles (dotted lines)
    all_handles = []

    _SS_LEGEND_ORDER = [
        "Cerebral WM", "Ventricles", "CSF",
        "Cerebellum WM", "Cerebellum cortex", "Brain stem",
        "Thalamus", "Caudate", "Putamen", "Globus pallidus",
        "Hippocampus", "Amygdala", "Accumbens", "Ventral DC",
    ]
    for struct_name in _SS_LEGEND_ORDER:
        if struct_name in ss_present_structs:
            c = _SS_STRUCT_COLORS[struct_name]
            all_handles.append(mlines.Line2D(
                [], [], color=c, linewidth=1.2, linestyle="dotted",
                label=struct_name))
    if ss_has_cortical:
        all_handles.append(mlines.Line2D(
            [], [], color=_CORTICAL_COLOR, linewidth=1.2, linestyle="dotted",
            label="Cortical parcellation"))

    # Build Manual seg handles (filled patches) — labelled with "(M)" prefix
    _MAN_LEGEND_ORDER = [
        "Dentate nucleus", "Substantia nigra", "Red nucleus",
        "Thalamus", "Caudate nucleus", "Putamen", "Globus pallidus",
    ]
    for struct_name in _MAN_LEGEND_ORDER:
        if struct_name in man_present_structs:
            c = _MAN_STRUCT_COLORS[struct_name]
            all_handles.append(mpatches.Patch(
                facecolor=c, edgecolor="grey", linewidth=0.3,
                alpha=0.85, label=struct_name))

    # Bottom of images in figure fraction
    img_bottom_frac = top_y - 2 * ch_frac - gy_frac
    legend_anchor_y = img_bottom_frac - legend_gap / fig_h

    # Center of content area
    content_center_x = (n_cols * (cw_frac + gx_frac) - gx_frac) / 2

    # Use 7 columns to wrap into 3 rows, larger font filling side space
    ncols = 7

    if all_handles:
        leg = fig.legend(
            handles=all_handles, loc="upper center",
            bbox_to_anchor=(content_center_x, legend_anchor_y),
            bbox_transform=fig.transFigure,
            ncol=ncols,
            fontsize=8.0, framealpha=0.0,
            edgecolor="none",
            handleheight=0.8, handlelength=1.4,
            borderpad=0.2, columnspacing=0.6, labelspacing=0.3,
            labelcolor="black",
        )

    return fig


# =========================================================================
# Main
# =========================================================================
def main():
    print("=" * 60)
    print("Figure 2 v4 — Segmentation comparison")
    print(f"  Subject: {SUB} / {SES}")
    print(f"  Slices:  {SLICES}")
    print("=" * 60)

    fig = build_figure()
    out_path = OUT_DIR / "Figure 2 v4.pdf"
    fig.savefig(str(out_path), format="pdf", facecolor="white",
                bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    size_kb = os.path.getsize(out_path) / 1024
    print(f"\n  Saved: {out_path}  ({size_kb:.0f} KB)")
    print("Done!")


if __name__ == "__main__":
    main()
