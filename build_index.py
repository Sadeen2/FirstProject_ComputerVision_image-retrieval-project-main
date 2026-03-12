
"""
Build Visual Bag-of-Words codebook (visual vocabulary) from SIFT descriptors.

Improvements for better retrieval quality:
  - Larger vocabulary (K = 1200 visual words)
  - Limit descriptors PER IMAGE (balanced sampling)
  - Use more images (but still optional MAX_IMAGES)
"""

from pathlib import Path

import cv2
import numpy as np
from sklearn.cluster import MiniBatchKMeans
from tqdm import tqdm

from dataset import list_images, load_image
from features import preprocess_image, extract_sift_descriptors

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BOW_DIR = PROJECT_ROOT / "data" / "bow"
BOW_DIR.mkdir(parents=True, exist_ok=True)

CODEBOOK_PATH = BOW_DIR / "codebook.npz"


# Hyperparameters to TUNE

# 1) Vocabulary size K

VOCAB_SIZE = 1200          # try 800, 1200, 1600, 2000 and compare mAP

# 2) How many images to use for building the codebook.
#    None = use all. can set e.g. 800 for speed.
MAX_IMAGES = None          

# 3) Limit *total* descriptors used for k-means (memory/speed)
MAX_DESCRIPTORS = 200_000  # increase to learn a richer codebook

# 4) Limit descriptors per image so sampling is more balanced
DESCRIPTORS_PER_IMAGE = 400    # 300–800 is typical


def build_codebook():
    items = list_images()  # [(Path, label), ...]

    if MAX_IMAGES is not None:
        items = items[:MAX_IMAGES]

    all_desc = []

    print(f"[build_codebook] Collecting SIFT descriptors from {len(items)} images...")
    print(f"[build_codebook] DESCRIPTORS_PER_IMAGE = {DESCRIPTORS_PER_IMAGE}")

    for img_path, _ in tqdm(items, desc="Images"):
        img_bgr = load_image(img_path, target_size=(256, 256))
        _, img_gray_norm = preprocess_image(img_bgr, target_size=(256, 256))

        desc = extract_sift_descriptors(img_gray_norm)   # shape [N, 128] or None
        if desc is None or len(desc) == 0:
            continue

        desc = desc.astype(np.float32)

        # ---- balanced sampling: take at most DESCRIPTORS_PER_IMAGE from this image
        if desc.shape[0] > DESCRIPTORS_PER_IMAGE:
            idx = np.random.choice(
                desc.shape[0],
                DESCRIPTORS_PER_IMAGE,
                replace=False
            )
            desc = desc[idx]

        all_desc.append(desc)

    if not all_desc:
        raise RuntimeError("No SIFT descriptors found in dataset.")

    all_desc = np.vstack(all_desc).astype(np.float32)

    # Optionally subsample descriptors globally
    if all_desc.shape[0] > MAX_DESCRIPTORS:
        idx = np.random.choice(all_desc.shape[0], MAX_DESCRIPTORS, replace=False)
        all_desc = all_desc[idx]

    print(f"[build_codebook] Total descriptors used for k-means: {all_desc.shape[0]}")

    # -------------------------------------------------
    # K-means (MiniBatch) to learn codebook
    # -------------------------------------------------
    print(f"[build_codebook] Running MiniBatchKMeans with K={VOCAB_SIZE}...")
    kmeans = MiniBatchKMeans(
        n_clusters=VOCAB_SIZE,
        batch_size=4000,     # larger batch for more stable centers
        n_init=3,
        verbose=1,
    )
    kmeans.fit(all_desc)

    centers = kmeans.cluster_centers_.astype(np.float32)  # [K, 128]

    np.savez_compressed(
        CODEBOOK_PATH,
        centers=centers,
    )

    print(f"[build_codebook] Saved codebook to: {CODEBOOK_PATH}")
    print(f"[build_codebook] Vocabulary size K = {centers.shape[0]}")


if __name__ == "__main__":
    build_codebook()
