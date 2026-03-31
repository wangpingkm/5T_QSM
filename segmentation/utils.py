#!/usr/bin/env python3
"""
QSM Pipeline — Shared Utilities
=================================
Common functions used across multiple pipeline steps.
"""

import numpy as np
from scipy import ndimage


def create_brain_mask(magnitude_data, threshold_pct=15, erode=False):
    """
    Create a brain mask from magnitude data using:
    1. Percentile-based threshold
    2. Morphological operations (closing, opening, fill holes)
    3. Largest connected component selection
    4. Optional erosion for edge artifact removal

    Parameters
    ----------
    magnitude_data : 3D or 4D array
        Magnitude image. If 4D, uses first echo.
    threshold_pct : float
        Percentile threshold for initial binarization (default 15).
    erode : bool
        If True, apply one iteration of erosion to remove edge effects.

    Returns
    -------
    mask : 3D bool array
    """
    if magnitude_data.ndim == 4:
        mag = magnitude_data[:, :, :, 0]
    else:
        mag = magnitude_data

    abs_data = np.abs(mag).astype(np.float64)
    if abs_data.max() == 0:
        return np.zeros(mag.shape, dtype=bool)

    # Normalize by 98th percentile of nonzero voxels
    nonzero = abs_data[abs_data > 0]
    if len(nonzero) == 0:
        return np.zeros(mag.shape, dtype=bool)

    mag_max = np.percentile(nonzero, 98)
    mag_norm = abs_data / max(mag_max, 1e-10)

    # Threshold
    threshold = threshold_pct / 100.0
    mask = mag_norm > threshold

    # Morphological operations
    struct = ndimage.generate_binary_structure(3, 2)
    mask = ndimage.binary_closing(mask, structure=struct, iterations=3)
    mask = ndimage.binary_fill_holes(mask)
    mask = ndimage.binary_opening(mask, structure=struct, iterations=2)

    # Keep largest connected component
    labeled, num_features = ndimage.label(mask)
    if num_features > 1:
        sizes = ndimage.sum(mask, labeled, range(1, num_features + 1))
        largest = np.argmax(sizes) + 1
        mask = labeled == largest

    # Optional erosion
    if erode:
        mask = ndimage.binary_erosion(mask, structure=struct, iterations=1)

    return mask.astype(bool)


def compute_brain_centroid(mask, affine):
    """Compute brain center-of-mass in both voxel and world coordinates."""
    com_vox = ndimage.center_of_mass(mask)
    com_world = affine @ np.array([*com_vox, 1.0])
    return com_world[:3], np.array(com_vox)


def voxel_sizes(affine):
    """Extract voxel sizes from affine matrix."""
    return np.sqrt(np.sum(affine[:3, :3] ** 2, axis=0))


def world_to_voxel(affine, point):
    """Convert world coordinate to voxel coordinate."""
    return (np.linalg.inv(affine) @ np.array([*point, 1.0]))[:3]
