# QSM 5 T Multicenter Reproducibility Study

Quantitative Susceptibility Mapping (QSM) reproducibility analysis pipeline for
a multicenter 3 T / 5 T MRI study. This repository contains all code used for
segmentation, statistical analysis, figure/table generation, and comparison with
published literature.

## Study Overview

- **Subjects**: 7 healthy volunteers (4 M / 3 F, age 25–35 y)
- **Sites**: 3 × 5 T (UIH uMR Jupiter) + 1 × 3 T (UIH uMR 790)
- **Protocol**: Multi-echo GRE (5-echo and 10-echo)
- **ROIs**: Thalamus, Caudate Nucleus, Putamen, Globus Pallidus, Substantia
  Nigra, Red Nucleus, Dentate Nucleus, Frontal White Matter
- **Unit**: All QSM values reported in **ppb** (parts per billion)

## Repository Structure

```
QSM_5T_Reproducibility/
├── README.md                   ← This file
├── segmentation/               ← Brain segmentation pipeline
│   ├── config.py               ← Paths and parameters (edit for your env)
│   ├── roi_analysis.py         ← SynthSeg atlas-based ROI extraction
│   ├── run_synthseg_batch.sh   ← Batch SynthSeg processing
│   └── register_and_extract.py ← Registration + ROI value extraction
├── analysis/                   ← Statistical analysis and figure generation
│   ├── qsm_statistical_analysis.py  ← Main analysis (Figs 3–7, Tables 1–5)
│   ├── gen_figure1.py          ← Figure 1: QSM maps montage
│   ├── gen_figure2.py          ← Figure 2: Segmentation overlays
│   ├── gen_supp_figure1.py     ← Supplementary Figure 1
│   ├── gen_supp_figure2.py     ← Supplementary Figure 2
│   └── generate_tables.py      ← DOCX table generation
├── review/                     ← Literature comparison (Figure 7)
│   ├── regenerate_figure7.py   ← Standalone Figure 7 generator
│   ├── update_literature.py    ← Build literature Excel from verified data
│   └── Figure_7_literature_data.xlsx  ← Literature QSM reference data
├── output/                     ← Generated figures, tables, statistics
│   ├── Figure1/ ... Figure7/
│   ├── Tables/
│   └── statistics/
└── docs/
    ├── code_review.md
    └── Figure7_manuscript_text.md  ← Methods, results, figure legend for Fig 7
```

## Quick Start

### 1. Environment Setup

```bash
# Python dependencies
pip install numpy scipy matplotlib pandas openpyxl nibabel pingouin seaborn

# FreeSurfer / SynthSeg (for segmentation only)
export FREESURFER_HOME=/path/to/freesurfer
export FS_LICENSE=/path/to/license.txt
```

### 2. Configure Paths

Edit `segmentation/config.py` to set paths for your environment:

```python
NIFTI_ROOT = "/path/to/nifti/data"
FREESURFER_HOME = "/path/to/freesurfer"
```

### 3. Segmentation Pipeline

```bash
# Run SynthSeg segmentation on all subjects
bash segmentation/run_synthseg_batch.sh

# Extract ROI QSM values
python segmentation/roi_analysis.py --method hybrid
```

### 4. Statistical Analysis

```bash
# Generate all figures (3–7) and tables (1–5)
python analysis/qsm_statistical_analysis.py
```

### 5. Literature Comparison (Figure 7)

```bash
# Rebuild the literature Excel from verified data (optional — Excel is provided)
python review/update_literature.py

# Regenerate Figure 7 from literature data
python review/regenerate_figure7.py
```

## Pipeline Overview

```
SynthSeg segmentation → ROI value extraction → Statistical analysis → Figures/Tables
                                                                    → Literature review (Fig 7)
```

| Step | Script | Output |
|------|--------|--------|
| Segmentation | `segmentation/roi_analysis.py` | ROI masks, QSM values per region |
| ROI Extraction | `segmentation/register_and_extract.py` | CSV with per-subject ROI values |
| Analysis | `analysis/qsm_statistical_analysis.py` | Figures 3–7, Tables 1–5, ICC, wCV |
| Figure 1 | `analysis/gen_figure1.py` | QSM map montage |
| Figure 2 | `analysis/gen_figure2.py` | Segmentation overlay |
| Figure 7 | `review/regenerate_figure7.py` | Literature comparison across field strengths |
| Tables | `analysis/generate_tables.py` | DOCX tables for manuscript |

## ROI Analysis Methods

The pipeline supports three segmentation approaches:

| Method | ROIs | Source |
|--------|------|--------|
| **Atlas** (SynthSeg) | GP, Putamen, Caudate, Thalamus | FreeSurfer aseg labels |
| **Centroid** | RN, SN, DN, Frontal WM | Coordinate-based peak search |
| **Hybrid** (default) | All 8 ROIs | Atlas for basal ganglia, centroid for brainstem |

## Literature Database

The `review/` folder contains a curated Excel database of published QSM values
across field strengths (1.5 T – 7 T), used to generate Figure 7.

### Curation Process

All entries were:
1. **DOI-verified** via the CrossRef API to confirm correct citation metadata
2. **Values confirmed** against source paper PDFs (tables, text, or
   supplementary materials)
3. **Harmonised** to a common format: ppb units, bilateral L/R averaging,
   age-regression models evaluated at a reference age

### Inclusion / Exclusion Criteria

Studies were included if they reported mean QSM values for **≥ 3 of 4 target
ROIs** (thalamus, caudate, putamen, globus pallidus) in healthy adult controls,
with whole-structure values available in tables or text. Studies using
white-matter referencing or reporting only subregions were excluded. See
[`docs/Figure7_manuscript_text.md`](docs/Figure7_manuscript_text.md) for the
full inclusion/exclusion criteria, data harmonisation procedures, per-study
source notes, and the complete list of excluded studies with rationale.

### Final Dataset: 13 Verified Studies

| Field (T) | Studies (*k*) | Total Subjects (*N*) |
|:---:|:---:|:---:|
| 1.5 | 3 | 368 |
| 3 | 9 | 1,648–1,670 |
| 7 | 1 | 14 |

*Note*: At 3 T, one study (Hinoda et al.) did not report thalamus, resulting
in *k* = 8 and *N* = 1,648 for thalamus vs *k* = 9 and *N* = 1,670 for the
other three ROIs.

Each entry includes: first author, full citation, field strength, healthy
control sample size, mean QSM values (ppb) for Thalamus / Caudate / Putamen /
Pallidum, DOI, and extraction notes.

## Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| numpy | ≥ 1.21 | Numerical operations |
| scipy | ≥ 1.7 | Interpolation, statistics |
| matplotlib | ≥ 3.5 | Figure generation |
| pandas | ≥ 1.3 | Data manipulation |
| openpyxl | ≥ 3.0 | Excel I/O |
| nibabel | ≥ 3.2 | NIfTI file I/O |
| pingouin | ≥ 0.5 | ICC computation |
| seaborn | ≥ 0.11 | Statistical plots |
| python-docx | ≥ 0.8 | DOCX table export |

## License

This code accompanies the manuscript: *"Quantitative Susceptibility Mapping at
5 T: A Multicenter Reproducibility Study"*. Please cite the paper if you use
this code.

## Contact

For questions about the code or data, please open an issue or contact the
corresponding author.
