
# src/features.py
"""
Feature extraction module for classical image retrieval.


1. Pre-processing
   - Resize to fixed size (e.g., 256x256)
   - Convert to grayscale
   - Normalize grayscale to [0, 1]

2. Color features: Color Moments (H, S, V)
   - mean, standard deviation, skewness per channel
   - total 9D vector

3. Edge/shape features: HOG
   - cell-based Histogram of Oriented Gradients on gray image

4. Local texture / interest-point features:
   - SIFT descriptors on gray image
   - Visual Bag-of-Words histogram (TF) using K-means codebook

"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import cv2
import numpy as np



# Codebook paths (for BoW)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BOW_DIR = PROJECT_ROOT / "data" / "bow"
CODEBOOK_PATH = BOW_DIR / "codebook.npz"  # created by build_codebook.py

_CODEBOOK: np.ndarray | None = None  # cached cluster centers


def load_codebook() -> np.ndarray:
    """
    Load BoW codebook (visual vocabulary) from disk.

    Returns:
        centers: [K, 128] float32 (SIFT cluster centers)
    """
    global _CODEBOOK
    if _CODEBOOK is not None:
        return _CODEBOOK

    if not CODEBOOK_PATH.exists():
        raise FileNotFoundError(
            f"BoW codebook not found: {CODEBOOK_PATH}\n"
            f"Run 'python -m src.build_codebook' first."
        )

    data = np.load(CODEBOOK_PATH)
    centers = data["centers"].astype(np.float32)  # [K, 128]
    _CODEBOOK = centers
    return _CODEBOOK



# 1) Pre-processing

def preprocess_image(
    img_bgr: np.ndarray,
    target_size: Tuple[int, int] = (256, 256),
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Preprocess an input BGR image.

    Returns:
      img_bgr_resized : uint8 RBG image (for color features)
      img_gray_norm   : float32 gray image in [0, 1] (for HOG / SIFT / BoW)
    """
    img_bgr_resized = cv2.resize(
        img_bgr,
        target_size,
        interpolation=cv2.INTER_AREA,
    )

    img_gray = cv2.cvtColor(img_bgr_resized, cv2.COLOR_BGR2GRAY)
    img_gray_norm = img_gray.astype(np.float32) / 255.0

    return img_bgr_resized, img_gray_norm



# 2) Color features – Color Moments (HSV)


def _channel_moments(channel: np.ndarray) -> Tuple[float, float, float]:
    """
    Compute color moments (mean, std, skewness) for one channel.
    """
    ch = channel.astype(np.float32).reshape(-1)
    if ch.size == 0:
        return 0.0, 0.0, 0.0

    mean = float(np.mean(ch))
    std = float(np.std(ch))

    # 3rd central moment → skewness (cube root for scale)
    m3 = np.mean((ch - mean) ** 3)
    skew = float(np.cbrt(m3)) if m3 != 0 else 0.0

    return mean, std, skew


def extract_color_moments(img_bgr: np.ndarray) -> np.ndarray:
    """
    Extract global Color Moments from HSV image.

    For each channel (H, S, V):
      - mean
      - standard deviation
      - skewness

    Total dimension = 9.
    """
    img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(img_hsv)

    moments = []
    for ch in (h, s, v):
        mean, std, skew = _channel_moments(ch)
        moments.extend([mean, std, skew])

    moments = np.array(moments, dtype=np.float32)

    # Optional L2 normalization
    norm = np.linalg.norm(moments) + 1e-8
    moments /= norm

    return moments  # shape (9,)



# 3) HOG – edge/shape descriptor


def extract_hog(
    img_gray_norm: np.ndarray,
    cell_size: Tuple[int, int] = (16, 16),
    num_bins: int = 9,
) -> np.ndarray:
    """
    HOG descriptor :

    - Compute gradients (Sobel)
    - Convert to magnitude + orientation [0, 180)
    - Divide image into cells of size cell_size
    - Build orientation histogram for each cell (num_bins)
    - Flatten and L2-normalize
    """
    H, W = img_gray_norm.shape

    gx = cv2.Sobel(img_gray_norm, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(img_gray_norm, cv2.CV_32F, 0, 1, ksize=3)

    mag, ang = cv2.cartToPolar(gx, gy, angleInDegrees=True)
    ang = np.mod(ang, 180.0)  # unsigned gradients

    cell_h, cell_w = cell_size
    n_cells_y = H // cell_h
    n_cells_x = W // cell_w

    hist = np.zeros((n_cells_y, n_cells_x, num_bins), dtype=np.float32)
    bin_width = 180.0 / num_bins

    for y in range(H):
        cy = y // cell_h
        if cy >= n_cells_y:
            continue
        for x in range(W):
            cx = x // cell_w
            if cx >= n_cells_x:
                continue

            m = mag[y, x]
            a = ang[y, x]
            bin_idx = int(a // bin_width)
            if bin_idx >= num_bins:
                bin_idx = num_bins - 1

            hist[cy, cx, bin_idx] += m

    hog_vec = hist.flatten()
    norm = np.linalg.norm(hog_vec) + 1e-8
    hog_vec /= norm

    return hog_vec  # [n_cells_y * n_cells_x * num_bins]



# 4) SIFT descriptors + BoW TF histogram

def extract_sift_descriptors(img_gray_norm: np.ndarray) -> np.ndarray | None:
    """
    Extract SIFT descriptors (N x 128) for local interest points.

    Returns:
      - descriptors: float32 [N_keypoints, 128]
      - or None if no descriptors.
    """
    img_gray_8u = (img_gray_norm * 255.0).astype(np.uint8)

    if not hasattr(cv2, "SIFT_create"):
        raise RuntimeError(
            "SIFT is not available. Install opencv-contrib-python:\n"
            "  python -m pip install opencv-contrib-python"
        )

    sift = cv2.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(img_gray_8u, None)

    if descriptors is None or len(descriptors) == 0:
        return None

    return descriptors.astype(np.float32)


def compute_bow_histogram(
    descriptors: np.ndarray | None,
    centers: np.ndarray,
) -> np.ndarray:
    """
    Compute Bag-of-Words histogram (TF) from SIFT descriptors.

    Args:
        descriptors: [N, 128] SIFT descriptors or None
        centers: [K, 128] visual vocabulary

    Returns:
        hist_tf: 1-D float32 of length K
                 L1-normalized term-frequency (TF) vector.
    """
    K = centers.shape[0]

    if descriptors is None or len(descriptors) == 0:
        return np.zeros(K, dtype=np.float32)

    # Distances to all centers
    diff = descriptors[:, None, :] - centers[None, :, :]  # [N, K, 128]
    dists = np.sum(diff * diff, axis=2)                   # [N, K]
    nearest = np.argmin(dists, axis=1)                    # [N] word indices

    hist, _ = np.histogram(nearest, bins=K, range=(0, K))
    hist = hist.astype(np.float32)

    total = hist.sum()
    if total > 0:
        hist /= total   # L1 normalization → TF

    return hist



# 5) Helper: extract all parts (color, HOG, BoW TF)

def extract_parts_for_image(
    img_bgr: np.ndarray,
    target_size: Tuple[int, int] = (256, 256),
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Extract feature parts for one image:

      - Color Moments (HSV)
      - HOG (edges/shape)
      - BoW TF histogram from SIFT

    Returns:
      color_feat : [C]
      hog_feat   : [H]
      bow_tf     : [K]  (TF only, TF–IDF applied later)
    """
    centers = load_codebook()

    img_bgr_resized, img_gray_norm = preprocess_image(
        img_bgr,
        target_size=target_size,
    )

    color_feat = extract_color_moments(img_bgr_resized)
    hog_feat = extract_hog(img_gray_norm)
    desc = extract_sift_descriptors(img_gray_norm)
    bow_tf = compute_bow_histogram(desc, centers)

    return color_feat, hog_feat, bow_tf
