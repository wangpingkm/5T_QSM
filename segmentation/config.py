#!/usr/bin/env python3
"""
QSM Pipeline — Unified Configuration
======================================
All paths, parameters, and shared constants in one place.
"""

import os
from pathlib import Path

# ============================================================
# Paths — adjust these for your environment
# ============================================================
PROJECT_ROOT = Path(__file__).resolve().parent.parent  # QSM_5T_Reproducibility/

# Input data
DICOM_ROOT = "/path/to/dicom/data"
NIFTI_ROOT = str(PROJECT_ROOT / "data" / "nifti")

# Output directories
OUTPUT_DIR = str(PROJECT_ROOT / "output")
QSM_OUTPUT = os.path.join(OUTPUT_DIR, "qsm_maps")
ROI_OUTPUT = os.path.join(OUTPUT_DIR, "roi_analysis")
STATS_DIR = os.path.join(OUTPUT_DIR, "statistics")
FIG_DIR = os.path.join(OUTPUT_DIR, "figures")
OVERLAY_DIR = os.path.join(ROI_OUTPUT, "overlays")

# External tools
DCM2NIIX = "dcm2niix"

# FreeSurfer / SynthSeg
FREESURFER_HOME = "/path/to/freesurfer"
FS_LICENSE = "/path/to/freesurfer/license.txt"
SYNTHSEG_BIN_CANDIDATES = [
    os.path.join(FREESURFER_HOME, "bin", "mri_synthseg"),
    "/path/to/freesurfer/bin/mri_synthseg",
]

# ============================================================
# Subject mapping (canonical)
# ============================================================
SUBJECT_MAP = {
    "01": "Yuan",
    "02": "Wang",
    "03": "Xiang",
    "04": "Qin",
    "05": "Emu",
    "06": "Zhang",
    "07": "Huang",
    "08": "Lin",
    "09": "Song",
}

# ============================================================
# Site mapping
# ============================================================
SITE_MAP = {
    "3T_UIH1": "3TUIH1",
    "3T_UIH2": "3TUIH2",
    "5T_UIH1": "5TUIH1",
    "5T_UIH2": "5TUIH2",
    "5T_上海六院": "5TShanghai",
    "5T_深圳龙华": "5TShenzhen",
    "5T_荆州人民": "5TJingzhou",
    "5T_重庆三峡": "5TChongqing",
}

# ============================================================
# Acquisition parameters
# ============================================================
GAMMA = 42.576e6  # Gyromagnetic ratio for proton (Hz/T)
GAMMA_BAR = 2 * 3.14159265 * GAMMA

ECHO_TIMES = {
    "3T_10echo": [2.9, 6.7, 10.5, 14.3, 18.1, 21.9, 25.7, 29.5, 33.3, 37.1],
    "5T_10echo": [3.0, 6.8, 10.6, 14.4, 18.2, 22.0, 25.8, 29.6, 33.4, 37.2],
    "5T_5echo":  [3.0, 6.8, 10.6, 14.4, 18.2],
}

# ============================================================
# ROI definitions
# ============================================================
# Unit convention: ALL QSM values in this pipeline are in ppb (parts per billion).
# 1 ppm = 1000 ppb.
UNIT = "ppb"

ROI_ORDER = ["GP", "SN", "RN", "Putamen", "Caudate", "DN", "Thalamus", "FrontalWM"]

ROI_LABELS = {
    "GP": "GP", "SN": "SN", "RN": "RN", "Putamen": "Put",
    "Caudate": "CN", "DN": "DN", "Thalamus": "Tha", "FrontalWM": "FWM",
}

ROI_FULL = {
    "GP": "Globus Pallidus", "SN": "Substantia Nigra", "RN": "Red Nucleus",
    "Putamen": "Putamen", "Caudate": "Caudate Nucleus", "DN": "Dentate Nucleus",
    "Thalamus": "Thalamus", "FrontalWM": "Frontal White Matter",
}

# FreeSurfer aseg label IDs for atlas-based ROI
FREESURFER_ROI_LABELS = {
    "Caudate_L": 11,   "Caudate_R": 50,
    "Putamen_L": 12,   "Putamen_R": 51,
    "GP_L": 13,        "GP_R": 52,
    "Thalamus_L": 10,  "Thalamus_R": 49,
    "Hippocampus_L": 17, "Hippocampus_R": 53,
    "Amygdala_L": 18,  "Amygdala_R": 54,
    "VentralDC_L": 28, "VentralDC_R": 60,
    "Cerebellum_Cortex_L": 8, "Cerebellum_Cortex_R": 47,
    "Cerebellum_WM_L": 7, "Cerebellum_WM_R": 46,
    "BrainStem": 16,
}

BILATERAL_ROIS = {
    "Caudate":   ("Caudate_L",   "Caudate_R"),
    "Putamen":   ("Putamen_L",   "Putamen_R"),
    "GP":        ("GP_L",        "GP_R"),
    "Thalamus":  ("Thalamus_L",  "Thalamus_R"),
    "RN":        ("RN_L",        "RN_R"),
    "SN":        ("SN_L",        "SN_R"),
    "DN":        ("DN_L",        "DN_R"),
    "FrontalWM": ("FrontalWM_L", "FrontalWM_R"),
}

# Legacy centroid-based ROI definitions (kept for backward compatibility)
CENTROID_ROI_DEFS = {
    "GP_L":       {"offset": (-20,  14, -6), "search_r": 5, "roi_r": 2.5, "refine": "max"},
    "GP_R":       {"offset": ( 20,  14, -6), "search_r": 5, "roi_r": 2.5, "refine": "max"},
    "Putamen_L":  {"offset": (-24,  17, -3), "search_r": 5, "roi_r": 3.0, "refine": "max"},
    "Putamen_R":  {"offset": ( 24,  17, -3), "search_r": 5, "roi_r": 3.0, "refine": "max"},
    "Caudate_L":  {"offset": (-12,  26,  2), "search_r": 5, "roi_r": 3.0, "refine": "max"},
    "Caudate_R":  {"offset": ( 12,  26,  2), "search_r": 5, "roi_r": 3.0, "refine": "max"},
    "Thalamus_L": {"offset": (-10,  -1,  0), "search_r": 0, "roi_r": 5.0, "refine": "none"},
    "Thalamus_R": {"offset": ( 10,  -1,  0), "search_r": 0, "roi_r": 5.0, "refine": "none"},
    "RN_L":       {"offset": ( -5,  -9,-12), "search_r": 4, "roi_r": 2.0, "refine": "max"},
    "RN_R":       {"offset": (  5,  -9,-12), "search_r": 4, "roi_r": 2.0, "refine": "max"},
    "SN_L":       {"offset": (-11,  -3,-18), "search_r": 4, "roi_r": 2.0, "refine": "max"},
    "SN_R":       {"offset": ( 11,  -3,-18), "search_r": 4, "roi_r": 2.0, "refine": "max"},
    "DN_L":       {"offset": (-16, -42,-33), "search_r": 6, "roi_r": 3.0, "refine": "max"},
    "DN_R":       {"offset": ( 16, -42,-33), "search_r": 6, "roi_r": 3.0, "refine": "max"},
    "FrontalWM_L":{"offset": (-22,  38, 18), "search_r": 5, "roi_r": 3.0, "refine": "min"},
    "FrontalWM_R":{"offset": ( 22,  38, 18), "search_r": 5, "roi_r": 3.0, "refine": "min"},
}

# ============================================================
# Literature reference values (in ppb)
# ============================================================
LITERATURE_VALUES = {
    "Caudate":   {"3T": {"mean": 40,  "std": 15, "source": "Langkammer 2012"},
                  "7T": {"mean": 45,  "std": 18, "source": "Bilgic 2012"}},
    "Putamen":   {"3T": {"mean": 50,  "std": 15, "source": "Langkammer 2012"},
                  "7T": {"mean": 55,  "std": 20, "source": "Bilgic 2012"}},
    "GP":        {"3T": {"mean": 93,  "std": 25, "source": "Langkammer 2012"},
                  "7T": {"mean": 100, "std": 30, "source": "Bilgic 2012"}},
    "Thalamus":  {"3T": {"mean": 2,   "std": 10, "source": "Langkammer 2012"},
                  "7T": {"mean": 5,   "std": 12, "source": "Bilgic 2012"}},
    "RN":        {"3T": {"mean": 75,  "std": 20, "source": "Sun & Wilman 2015"},
                  "7T": {"mean": 82,  "std": 25, "source": "Bilgic 2012"}},
    "SN":        {"3T": {"mean": 95,  "std": 25, "source": "Sun & Wilman 2015"},
                  "7T": {"mean": 105, "std": 30, "source": "Bilgic 2012"}},
    "DN":        {"3T": {"mean": 55,  "std": 20, "source": "Harada 2001"},
                  "7T": {"mean": 60,  "std": 22, "source": "estimated"}},
    "FrontalWM": {"3T": {"mean": -30, "std": 10, "source": "Li 2011"},
                  "7T": {"mean": -28, "std": 12, "source": "estimated"}},
}

# ============================================================
# Visualization
# ============================================================
C3T       = '#2166AC'
C3T_LIGHT = '#92C5DE'
C5T       = '#B2182B'
C5T_LIGHT = '#F4A582'
C_LIT     = '#1B7837'
C_LIT2    = '#762A83'
C_LIT3    = '#E08214'
GREY      = '#636363'

# ============================================================
# Helper: ensure output dirs exist
# ============================================================
def ensure_dirs():
    for d in [OUTPUT_DIR, QSM_OUTPUT, ROI_OUTPUT, STATS_DIR, FIG_DIR, OVERLAY_DIR]:
        os.makedirs(d, exist_ok=True)
