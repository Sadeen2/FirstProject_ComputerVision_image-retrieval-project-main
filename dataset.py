# src/build_index.py
"""
Build TF–IDF-based feature database + inverted index.

For each image:
  Input → Preprocessing → Color Moments + HOG + BoW(TF)

Then:
  - compute DF and IDF for visual words
  - compute BoW TF–IDF
  -  weight feature parts:
        Feature = [α * Color | β * HOG | γ * BoW_TFIDF]
  - concatenate parts
  - L2-normalize final feature vectors
  - build inverted index (posting lists per visual word)

Output:
  data/index/index_tfidf.npz with:
    - paths     : [N] relative image paths (str)
    - labels    : [N] class labels (str)
    - feats     : [N, D_total] float32, L2-normalized
    - idf       : [K] float32 IDF weights
    - postings  : [K] object array of int arrays (image indices)
    - bow_dim   : scalar K
    - color_dim : scalar
    - hog_dim   : scalar
"""

from pathlib import Path
from typing import List, Optional

import numpy as np
from tqdm import tqdm

from dataset import list_images, load_image
from features import extract_parts_for_image, load_codebook

PROJECT_ROOT = Path(__file__).resolve().parents[1]
INDEX_DIR = PROJECT_ROOT / "data" / "index"
INDEX_DIR.mkdir(parents=True, exist_ok=True)

INDEX_TFIDF_PATH = INDEX_DIR / "index_tfidf.npz"

# Weights for combining feature parts:
#   feature = [ALPHA_COLOR * Color | BETA_HOG * HOG | GAMMA_BOW * BoW_TFIDF]
ALPHA_COLOR = 0.2
BETA_HOG    = 0.2
GAMMA_BOW   = 1.0


def build_index_tfidf(limit: Optional[int] = None) -> None:
    centers = load_codebook()
    K = centers.shape[0]

    items = list_images()
    if limit is not None:
        items = items[:limit]

    print(f"[build_index_tfidf] Number of images: {len(items)}")
    print(f"[build_index_tfidf] BoW vocabulary size K = {K}")

    all_paths: List[str] = []
    all_labels: List[str] = []
    color_list: List[np.ndarray] = []
    hog_list: List[np.ndarray] = []
    bow_tf_list: List[np.ndarray] = []

    # 1) Extract parts (Color Moments, HOG, BoW TF)
    for img_path, label in tqdm(items, desc="Extracting features"):
        img_bgr = load_image(img_path, target_size=(256, 256))

        color_feat, hog_feat, bow_tf = extract_parts_for_image(img_bgr)

        color_list.append(color_feat)
        hog_list.append(hog_feat)
        bow_tf_list.append(bow_tf)

        rel_path = str(img_path.relative_to(PROJECT_ROOT))
        all_paths.append(rel_path)
        all_labels.append(label)

    color_feats = np.stack(color_list, axis=0).astype(np.float32)  # [N, C]
    hog_feats = np.stack(hog_list, axis=0).astype(np.float32)      # [N, H]
    bow_tf = np.stack(bow_tf_list, axis=0).astype(np.float32)      # [N, K]
    N = bow_tf.shape[0]

    color_dim = color_feats.shape[1]
    hog_dim = hog_feats.shape[1]

    print(
        f"[build_index_tfidf] N = {N}, "
        f"color_dim = {color_dim}, hog_dim = {hog_dim}, bow_dim = {K}"
    )

    # 2) Compute DF and IDF for BoW
    df = np.count_nonzero(bow_tf > 0, axis=0).astype(np.float32)  # [K]
    # idf_j = log((N + 1) / (df_j + 1)) + 1
    idf = np.log((N + 1.0) / (df + 1.0)) + 1.0                    # [K]

    # 3) Compute BoW TF–IDF
    bow_tfidf = bow_tf * idf[None, :]                             # [N, K]

    # 4) Apply weights to each part and concatenate full feature vectors
    color_feats_w = ALPHA_COLOR * color_feats
    hog_feats_w = BETA_HOG * hog_feats
    bow_feats_w = GAMMA_BOW * bow_tfidf

    feats = np.concatenate(
        [color_feats_w, hog_feats_w, bow_feats_w],
        axis=1,
    ).astype(np.float32)                                          # [N, D_total]

    # 5) L2-normalize (for cosine similarity)
    norms = np.linalg.norm(feats, axis=1, keepdims=True) + 1e-8
    feats /= norms

    # 6) Build inverted file index from BoW TF (unweighted TF)
    postings: List[np.ndarray] = []
    for j in range(K):
        img_indices = np.nonzero(bow_tf[:, j] > 0)[0]
        postings.append(img_indices.astype(np.int32))
    postings_arr = np.array(postings, dtype=object)

    # 7) Save index
    paths_arr = np.array(all_paths)
    labels_arr = np.array(all_labels)

    np.savez_compressed(
        INDEX_TFIDF_PATH,
        paths=paths_arr,
        labels=labels_arr,
        feats=feats,
        idf=idf.astype(np.float32),
        postings=postings_arr,
        bow_dim=K,
        color_dim=color_dim,
        hog_dim=hog_dim,
    )

    print(f"[build_index_tfidf] Saved TF–IDF index to: {INDEX_TFIDF_PATH}")
    print(f"[build_index_tfidf] D_total = {feats.shape[1]}")


if __name__ == "__main__":
    # 1) Run build_codebook.py first.
    # 2) Then run this to build the TF–IDF index.
    build_index_tfidf(limit=None)
