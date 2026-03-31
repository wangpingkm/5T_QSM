#!/usr/bin/env python3
"""
Generate Figure 1 (v4) — QSM 5T Reproducibility Workflow
==========================================================
Changes from v3:
  - No text stroke/border — pure text only
  - Panel A stretched to match Panel B/C full width (3 images fill 4-col width)
  - Panel label "A" in white, upper-left inside T1 image
  - No vertical gaps between panels — tight stacking
  - Panel labels "B" / "C" in black, upper-left inside their first image
  - Panel C labels: "Subject 03, 5T-xxx" on one line (not 03Xiang)
  - Colorbar on right, spanning B+C

Output:  output/Figure1/Figure 1.pdf
"""

import os
import numpy as np
import nibabel as nib
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patheffects as patheffects
from matplotlib.colors import Normalize

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT = Path(__file__).resolve().parent.parent
NIFTI_ROOT = Path("/path/to/project/"
                   "Reject and Resubmit/Protocal/NIfTI")
REG_DIR = PROJECT / "output" / "roi_analysis" / "registered"
OUT_DIR = PROJECT / "output" / "Figure1"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------
plt.rcParams.update({
    "font.family": "Arial",
    "font.size": 8,
    "axes.titlesize": 9,
    "axes.titleweight": "bold",
    "axes.labelsize": 8,
    "figure.dpi": 300,
    "savefig.dpi": 300,
})

# ---------------------------------------------------------------------------
# QSM display constants
# ---------------------------------------------------------------------------
QSM_VMIN = -118   # ppb  (midpoint between -100 and -135 for moderate light blue bg)
QSM_VMAX = 150    # ppb
GRAY_PLOW = 1
GRAY_PHIGH = 99

# Slice selections
Z_PANEL_AB = 54
Z_PANEL_C = 44

# ---------------------------------------------------------------------------
# Data paths
# ---------------------------------------------------------------------------
def qsm_path(sub, ses):
    return str(NIFTI_ROOT / sub / ses / "qsm" / "qsm_10echo_run-01.nii.gz")

def t1_reg_path(sub, ses):
    return str(REG_DIR / sub / ses / "t1_in_qsm.nii.gz")

# ---------------------------------------------------------------------------
# Volume cache
# ---------------------------------------------------------------------------
_cache = {}

def load_vol(path):
    if path not in _cache:
        img = nib.load(path)
        d = img.get_fdata(dtype=np.float32)
        if d.ndim > 3:
            d = d[:, :, :, 0]
        _cache[path] = d
    return _cache[path]

def get_slice(path, z):
    vol = load_vol(path)
    return vol[:, :, z].T

# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------
def imshow_gray(ax, data_2d):
    nz = data_2d[data_2d != 0]
    if len(nz) == 0:
        vmin, vmax = 0, 1
    else:
        vmin = np.percentile(nz, GRAY_PLOW)
        vmax = np.percentile(nz, GRAY_PHIGH)
    im = ax.imshow(data_2d, cmap="gray", vmin=vmin, vmax=vmax,
                   origin="lower", aspect="equal", interpolation="bilinear")
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)
    return im


def imshow_qsm_gray(ax, data_2d):
    nz = data_2d[data_2d != 0]
    if len(nz) == 0:
        vmin, vmax = -100, 150
    else:
        vmin = np.percentile(nz, 2)
        vmax = np.percentile(nz, 98)
    im = ax.imshow(data_2d, cmap="gray", vmin=vmin, vmax=vmax,
                   origin="lower", aspect="equal", interpolation="bilinear")
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)
    return im


def imshow_qsm_color(ax, data_2d):
    im = ax.imshow(data_2d, cmap="RdBu_r", vmin=QSM_VMIN, vmax=QSM_VMAX,
                   origin="lower", aspect="equal", interpolation="bilinear")
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)
    return im


def label_inside(ax, text, color="white", fontsize=7, position="top-center"):
    """Place pure text (no stroke/border) inside the image."""
    if position == "top-center":
        x, y, ha, va = 0.50, 0.96, "center", "top"
    elif position == "upper-left":
        x, y, ha, va = 0.03, 0.96, "left", "top"
    else:
        x, y, ha, va = 0.50, 0.96, "center", "top"

    ax.text(x, y, text,
            transform=ax.transAxes,
            ha=ha, va=va,
            fontsize=fontsize, fontweight="bold",
            color=color)


# ===========================================================================
# Pre-load
# ===========================================================================
def preload():
    SUB_X = "sub-03Xiang"
    SUB_H = "sub-07Huang"
    paths = set()
    paths.add(t1_reg_path(SUB_X, "ses-3TUIH1"))
    paths.add(qsm_path(SUB_X, "ses-3TUIH1"))
    paths.add(qsm_path(SUB_X, "ses-5TUIH1"))
    for ses in ["ses-5TChongqing", "ses-5TShanghai", "ses-5TUIH1", "ses-5TUIH2"]:
        paths.add(qsm_path(SUB_X, ses))
    for ses in ["ses-5TJingzhou", "ses-5TShanghai", "ses-5TUIH1", "ses-5TUIH2"]:
        paths.add(qsm_path(SUB_H, ses))
    for p in sorted(paths):
        if os.path.exists(p):
            print(f"  Loading .../{'/'.join(Path(p).parts[-3:])}")
            load_vol(p)
        else:
            print(f"  WARNING — missing: {p}")


# ===========================================================================
# Build figure
# ===========================================================================
def build_figure():
    SUB_X = "sub-03Xiang"
    SUB_H = "sub-07Huang"

    xiang_5t_ses    = ["ses-5TChongqing", "ses-5TShanghai", "ses-5TUIH1", "ses-5TUIH2"]
    huang_5t_ses    = ["ses-5TJingzhou",  "ses-5TShanghai", "ses-5TUIH1", "ses-5TUIH2"]
    xiang_5t_labels = ["5T-Chongqing",   "5T-Shanghai",    "5T-UIH1",    "5T-UIH2"]
    huang_5t_labels = ["5T-Jingzhou",     "5T-Shanghai",    "5T-UIH1",    "5T-UIH2"]

    # ------------------------------------------------------------------
    # Image aspect ratio: display is 504 rows × 477 cols after transpose
    # ------------------------------------------------------------------
    img_aspect = 504.0 / 477.0  # height / width

    # ------------------------------------------------------------------
    # Layout — all panels use the SAME total content width (4-col based).
    # Panel A: 3 images stretched to fill the same width as 4 images.
    # No gaps between panels (tight vertical stacking).
    # ------------------------------------------------------------------
    n_cols = 4
    cell_w = 2.0              # inches per image cell in B/C
    cbar_w = 0.50             # right side for colorbar
    content_w = n_cols * cell_w   # total content width

    fig_w = content_w + cbar_w

    # Cell heights — Panel A cells are wider so taller to maintain aspect
    a_cell_w = content_w / 3.0
    a_cell_h = a_cell_w * img_aspect
    bc_cell_h = cell_w * img_aspect

    # Total figure height — no gaps, no margins (tight)
    # Panel A: 1 row
    # Panel B: 1 row
    # Panel C: 2 rows
    h_a = a_cell_h
    h_b = bc_cell_h
    h_c = 2 * bc_cell_h
    fig_h = h_a + h_b + h_c

    fig = plt.figure(figsize=(fig_w, fig_h), facecolor="white")

    # Fraction helpers
    left = 0.0
    right = content_w / fig_w
    cbar_left_frac = right + 0.01

    def y_frac_from_top(inches):
        return 1.0 - inches / fig_h

    # ==================================================================
    # Panel A — T1 + QSM gray 3T + QSM gray 5T  (1 row × 3, stretched)
    # ==================================================================
    y_a_top = 1.0
    y_a_bot = y_frac_from_top(h_a)
    a_w_frac = (content_w / 3.0) / fig_w

    # Track first axis of each panel for panel-letter placement
    ax_a_first = None
    for i in range(3):
        x0 = left + i * a_w_frac
        ax = fig.add_axes([x0, y_a_bot, a_w_frac, y_a_top - y_a_bot])
        if i == 0:
            s = get_slice(t1_reg_path(SUB_X, "ses-3TUIH1"), Z_PANEL_AB)
            imshow_gray(ax, s)
            label_inside(ax, "T1", color="white", fontsize=8, position="top-center")
            # Panel "A" — white, upper-left inside T1 image
            label_inside(ax, "A", color="white", fontsize=14, position="upper-left")
            ax_a_first = ax
        elif i == 1:
            s = get_slice(qsm_path(SUB_X, "ses-3TUIH1"), Z_PANEL_AB)
            imshow_qsm_gray(ax, s)
            label_inside(ax, "QSM 3T", color="white", fontsize=8, position="top-center")
        else:
            s = get_slice(qsm_path(SUB_X, "ses-5TUIH1"), Z_PANEL_AB)
            imshow_qsm_gray(ax, s)
            label_inside(ax, "QSM 5T", color="white", fontsize=8, position="top-center")

    # ==================================================================
    # Panel B — pseudo-color QSM × 4 scanners (1 row × 4)
    # ==================================================================
    y_b_top = y_a_bot
    y_b_bot = y_frac_from_top(h_a + h_b)
    b_w_frac = (content_w / 4.0) / fig_w

    last_im = None
    for i, (ses, lbl) in enumerate(zip(xiang_5t_ses, xiang_5t_labels)):
        x0 = left + i * b_w_frac
        ax = fig.add_axes([x0, y_b_bot, b_w_frac, y_b_top - y_b_bot])
        s = get_slice(qsm_path(SUB_X, ses), Z_PANEL_AB)
        last_im = imshow_qsm_color(ax, s)
        label_inside(ax, lbl, color="black", fontsize=7, position="top-center")
        if i == 0:
            # Panel "B" — black, upper-left inside first image
            label_inside(ax, "B", color="black", fontsize=14, position="upper-left")

    # ==================================================================
    # Panel C — pseudo-color QSM, 2 subjects × 4 scanners (2 rows × 4)
    # ==================================================================
    y_c_top = y_b_bot
    y_c_bot = 0.0
    c_w_frac = (content_w / 4.0) / fig_w
    c_row_h = (y_c_top - y_c_bot) / 2.0

    # Row 0 (upper): sub-03Xiang
    for i, (ses, lbl) in enumerate(zip(xiang_5t_ses, xiang_5t_labels)):
        x0 = left + i * c_w_frac
        y0 = y_c_bot + c_row_h   # upper row
        ax = fig.add_axes([x0, y0, c_w_frac, c_row_h])
        s = get_slice(qsm_path(SUB_X, ses), Z_PANEL_C)
        last_im = imshow_qsm_color(ax, s)
        label_inside(ax, f"Subject 03, {lbl}", color="black", fontsize=6,
                     position="top-center")
        if i == 0:
            # Panel "C" — black, upper-left inside first image
            label_inside(ax, "C", color="black", fontsize=14, position="upper-left")

    # Row 1 (lower): sub-07Huang
    for i, (ses, lbl) in enumerate(zip(huang_5t_ses, huang_5t_labels)):
        x0 = left + i * c_w_frac
        y0 = y_c_bot
        ax = fig.add_axes([x0, y0, c_w_frac, c_row_h])
        s = get_slice(qsm_path(SUB_H, ses), Z_PANEL_C)
        last_im = imshow_qsm_color(ax, s)
        label_inside(ax, f"Subject 07, {lbl}", color="black", fontsize=6,
                     position="top-center")

    # ==================================================================
    # Vertical colorbar — right side, spanning Panels B + C
    # ==================================================================
    if last_im is not None:
        cbar_x = right + 0.008
        cbar_w_frac = 0.015
        cbar_bot = y_c_bot
        cbar_top = y_b_top
        cax = fig.add_axes([cbar_x, cbar_bot + 0.02, cbar_w_frac,
                            cbar_top - cbar_bot - 0.04])
        cb = fig.colorbar(last_im, cax=cax, orientation="vertical")
        cb.ax.set_title("ppb", fontsize=7, pad=4)
        cb.ax.tick_params(labelsize=6)

    return fig


# ===========================================================================
# Main
# ===========================================================================
def main():
    print("=" * 60)
    print("Figure 1 (v4) Generator — QSM 5T Reproducibility")
    print("=" * 60)

    print("\nPre-loading volumes ...")
    preload()

    print("\nBuilding Figure 1 ...")
    fig = build_figure()
    out_path = OUT_DIR / "Figure 1.pdf"
    fig.savefig(str(out_path), format="pdf", facecolor="white",
                pad_inches=0)
    plt.close(fig)
    print(f"  Saved: {out_path}  ({os.path.getsize(out_path)/1024:.0f} KB)")
    print("\nDone!")


if __name__ == "__main__":
    main()
