#!/bin/bash
# Batch SynthSeg segmentation for all subjects with T1
set -e

export FREESURFER_HOME=/path/to/freesurfer
export FS_LICENSE=/path/to/freesurfer/license.txt
export PYTHONPATH="$FREESURFER_HOME/python/packages"
export PYTHONNOUSERSITE=1

FSPY="$FREESURFER_HOME/python/bin/python3"
SYNTHSEG="$FREESURFER_HOME/python/scripts/mri_synthseg"
NIFTI_ROOT="/path/to/nifti/data"
OUT_ROOT="/path/to/output"

# T1 files to process
declare -A T1_FILES
T1_FILES["sub-01Yuan"]="$NIFTI_ROOT/sub-01Yuan/ses-5TUIH1/anat/T1w.nii.gz"
T1_FILES["sub-02Wang"]="$NIFTI_ROOT/sub-02Wang/ses-5TUIH1/anat/T1w.nii.gz"
T1_FILES["sub-03Xiang"]="$NIFTI_ROOT/sub-03Xiang/ses-5TShanghai/anat/T1w.nii.gz"
T1_FILES["sub-04Qin"]="$NIFTI_ROOT/sub-04Qin/ses-5TShanghai/anat/T1w.nii.gz"
T1_FILES["sub-07Huang"]="$NIFTI_ROOT/sub-07Huang/ses-5TShanghai/anat/T1w.nii.gz"
T1_FILES["sub-08Lin"]="$NIFTI_ROOT/sub-08Lin/ses-5TShanghai/anat/T1w.nii.gz"
T1_FILES["sub-09Song"]="$NIFTI_ROOT/sub-09Song/ses-5TShanghai/anat/T1w.nii.gz"

for SUBJ in "${!T1_FILES[@]}"; do
    T1="${T1_FILES[$SUBJ]}"
    OUTDIR="$OUT_ROOT/$SUBJ"
    SEG="$OUTDIR/synthseg.nii.gz"
    VOL="$OUTDIR/synthseg_volumes.csv"
    QC="$OUTDIR/synthseg_qc.csv"
    RESAMPLE="$OUTDIR/synthseg_resampled.nii.gz"

    if [ -f "$SEG" ]; then
        echo "[SKIP] $SUBJ — already segmented"
        continue
    fi

    mkdir -p "$OUTDIR"
    echo "[RUN] $SUBJ — $T1"
    "$FSPY" "$SYNTHSEG" \
        --i "$T1" \
        --o "$SEG" \
        --vol "$VOL" \
        --qc "$QC" \
        --resample "$RESAMPLE" \
        --parc \
        --robust \
        --cpu \
        2>&1 | tee "$OUTDIR/synthseg.log"
    echo "[DONE] $SUBJ"
done

echo ""
echo "=== All done ==="
ls -la "$OUT_ROOT"/*/synthseg.nii.gz 2>/dev/null
