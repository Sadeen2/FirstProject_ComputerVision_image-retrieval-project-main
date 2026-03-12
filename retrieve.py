"""
main.py

Interactive image-to-image retrieval demo + evaluation summary.

Pipeline:

  Input Image → Preprocessing → Feature Extraction
             → Feature Database (BoW + TF-IDF)
             → Similarity Computation (cosine / euclidean)
             → Ranking → Retrieved Images (top-K)
             → Evaluation (P@K, R@K, mAP, Retrieval Time)

"""

from pathlib import Path
from typing import List, Dict

import time
import numpy as np
import matplotlib.pyplot as plt

from dataset import CALTECH_ROOT, load_image_rgb, list_images
from retrieve import retrieve_similar_images_tfidf
from evaluate import evaluate_retrieval

PROJECT_ROOT = Path(__file__).resolve().parents[1]


# Helper: show query + retrieved results in one figure

def show_retrieval_results(query_path: Path, results: List[dict], top_k: int = 5) -> None:
    """
    Display the query image and top-K retrieved images in one Matplotlib figure.
    """
    top_k = min(top_k, len(results))

    # Load query image as RGB
    query_img = load_image_rgb(query_path)

    plt.figure(figsize=(4 * (top_k + 1), 4))

    # --- subplot 1: query ---
    ax = plt.subplot(1, top_k + 1, 1)
    ax.imshow(query_img)
    ax.axis("off")
    ax.set_title("Query", fontsize=12)

    # --- retrieved images ---
    for i in range(top_k):
        r = results[i]
        img = load_image_rgb(r["path"])

        ax = plt.subplot(1, top_k + 1, i + 2)
        ax.imshow(img)
        ax.axis("off")
        title = (
            f"Rank {r.get('rank', i+1)}\n"
            f"{r['label']}\n"
            f"score={r['score']:.3f}"
        )
        ax.set_title(title, fontsize=9)

    plt.tight_layout()
    plt.show()



# Helper: let the user choose a query from Caltech-101

def choose_query_from_dataset() -> Path:
    """
    Let the user choose a query image from Caltech-101 easily.

    Steps:
      1) List class folders under CALTECH_ROOT.
      2) User chooses a class by index or name (or random).
      3) Show ALL images in that class with indices.
      4) User chooses image index (or random if Enter is pressed).

    Returns:
      Full Path to the chosen image.
    """
    if not CALTECH_ROOT.exists():
        raise FileNotFoundError(f"CALTECH_ROOT does not exist:\n  {CALTECH_ROOT}")

    # All class directories
    class_dirs = sorted([d for d in CALTECH_ROOT.iterdir() if d.is_dir()])
    class_names = [d.name for d in class_dirs]

    print("\n================ Dataset Query Selection ================")
    print("Caltech-101 classes (showing first 20):")
    max_show = min(20, len(class_names))
    for idx in range(max_show):
        print(f"  [{idx}] {class_names[idx]}")

    print("\nYou can:")
    print(f"  - Enter class index (0..{max_show - 1})")
    print("  - Or enter full class name (e.g. 'accordion', 'airplanes', ...)")
    print("  - Or just press Enter to choose a RANDOM class & image\n")

    user_cls = input("Choose class (index or name) [default: random]: ").strip()

    # --- choose class ---
    if user_cls == "":
        # random class
        cls_idx = np.random.randint(0, len(class_dirs))
        class_dir = class_dirs[cls_idx]
        print(f"[main] Random class selected: {class_dir.name}")
    elif user_cls.isdigit():
        idx = int(user_cls)
        if 0 <= idx < len(class_dirs):
            class_dir = class_dirs[idx]
            print(f"[main] Class by index: {class_dir.name}")
        else:
            print("[main] Invalid index → falling back to random class.")
            cls_idx = np.random.randint(0, len(class_dirs))
            class_dir = class_dirs[cls_idx]
            print(f"[main] Random class: {class_dir.name}")
    else:
        # maybe user typed the name
        if user_cls in class_names:
            class_dir = class_dirs[class_names.index(user_cls)]
            print(f"[main] Class by name: {class_dir.name}")
        else:
            print("[main] Unknown class name → falling back to random class.")
            cls_idx = np.random.randint(0, len(class_dirs))
            class_dir = class_dirs[cls_idx]
            print(f"[main] Random class: {class_dir.name}")

    # --- choose image inside class ---
    image_files = sorted([
        p for p in class_dir.iterdir()
        if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp")
    ])

    if not image_files:
        raise RuntimeError(f"No images found in class folder: {class_dir}")

    print(f"\nImages in class '{class_dir.name}':")
    # Show ALL images in this class
    for i, img_path in enumerate(image_files):
        print(f"  [{i}] {img_path.name}")

    print(
        f"\nYou can enter an image index (0..{len(image_files) - 1}) "
        "or press Enter for a RANDOM image."
    )

    user_img = input("Choose image index [default: random]: ").strip()

    if user_img == "":
        img_idx = np.random.randint(0, len(image_files))
        query_path = image_files[img_idx]
        print(f"[main] Random image selected: {query_path.name}")
    elif user_img.isdigit():
        idx = int(user_img)
        if 0 <= idx < len(image_files):
            query_path = image_files[idx]
            print(f"[main] Image by index: {query_path.name}")
        else:
            print("[main] Invalid image index → using first image.")
            query_path = image_files[0]
    else:
        print("[main] Invalid input → using first image.")
        query_path = image_files[0]

    print(f"\n[main] Final query image:\n  class = {class_dir.name}\n  path  = {query_path}")
    return query_path



# MAIN FLOW


print("=======================================================")
print("  Image-to-Image Retrieval System (Classical CV)")
print("  - Color Moments + HOG + SIFT + BoW + TF-IDF")
print("  - Dataset: Caltech-101 (multiple instances per class)")
print("=======================================================\n")

# 1) Let the user choose a query from the dataset
query_path = choose_query_from_dataset()
query_label = query_path.parent.name  # class name

# 2) Ask for top-K
try:
    top_k_str = input("\nEnter top-K (number of images to retrieve) [default 5]: ").strip()
    top_k = int(top_k_str) if top_k_str else 5
except ValueError:
    print("[main] Invalid number → using K=5.")
    top_k = 5

# 3) Retrieval (Similarity Computation + Ranking) with timing
print("\n[main] Running retrieval with cosine similarity...")
t0 = time.time()
results = retrieve_similar_images_tfidf(
    query_image_path=query_path,
    top_k=top_k,
    metric="cosine",   # or "euclidean"
)
t1 = time.time()
query_time = t1 - t0

print(f"[main] Retrieval time for this query: {query_time:.4f} seconds")

print("\n[main] Top-K retrieved images for this query:")
for r in results[:top_k]:
    print(
        f"  Rank {r.get('rank', '?')}: "
        f"score={r['score']:.4f}, label={r['label']}, path={r['path']}"
    )

# 4) Visualize everything in a single figure
show_retrieval_results(query_path, results, top_k=top_k)


# Evaluation Measures for THIS single query


# Build label_counts so we know how many relevant images exist in dataset
items_all = list_images()
label_counts = {}
for p, lbl in items_all:
    label_counts[lbl] = label_counts.get(lbl, 0) + 1

total_relevant = max(label_counts.get(query_label, 1) - 1, 0)  # exclude the query itself

# Relevant flags for retrieved results
relevant_flags = np.array(
    [1 if r["label"] == query_label else 0 for r in results],
    dtype=np.int32,
)
cum_rel = np.cumsum(relevant_flags)
total_rel_in_topk = int(cum_rel[-1]) if cum_rel.size > 0 else 0

def prec_at_k_single(k: int) -> float:
    if len(results) == 0:
        return 0.0
    k_eff = min(k, len(results))
    if k_eff == 0:
        return 0.0
    return float(cum_rel[k_eff - 1]) / float(k_eff)

def rec_at_k_single(k: int) -> float:
    if total_relevant == 0:
        return 0.0
    k_eff = min(k, len(results))
    if k_eff == 0:
        return 0.0
    return float(cum_rel[k_eff - 1]) / float(total_relevant)

# Average Precision (AP) for this query
if total_relevant == 0:
    AP_single = 0.0
else:
    precisions = []
    for rank, rel_flag in enumerate(relevant_flags, start=1):
        if rel_flag == 1:
            precisions.append(cum_rel[rank - 1] / float(rank))
    AP_single = float(np.mean(precisions)) if precisions else 0.0

print("\n=== Single-query Evaluation (this input image) ===")
print(f"  Query class label: {query_label}")
print(f"  Retrieval time:    {query_time:.4f} seconds")
print(f"  Total relevant images in dataset (same class, excluding query): {total_relevant}")
print(f"  Relevant retrieved in top-{top_k}: {total_rel_in_topk}")
for k in (1, 5, 10):
    if k <= len(results):
        print(f"  Precision@{k}: {prec_at_k_single(k):.4f}")
        print(f"  Recall@{k}:    {rec_at_k_single(k):.4f}")
    else:
        print(f"  Precision@{k}: (not defined, K>{len(results)})")
        print(f"  Recall@{k}:    (not defined, K>{len(results)})")
print(f"  Average Precision (AP) for this query: {AP_single:.4f}")
print("===============================================")




# 5. Evaluation Measures – run automatic evaluation


run_eval = input(
    "\nDo you want to run quantitative evaluation on random queries? (y/n): "
).strip().lower()

if run_eval == "y":
    metrics = evaluate_retrieval(
        ks=(1, 5, 10),
        max_queries=100,          
        top_k_for_retrieval=200,
        metric="cosine",
    )

    print("\n5. Evaluation Measures (results):")
    print(f"  Mean Average Precision (mAP): {metrics['mAP']:.4f}")
    for k in (1, 5, 10):
        print(f"  Precision@{k}: {metrics[f'P@{k}']:.4f}")
        print(f"  Recall@{k}:    {metrics[f'R@{k}']:.4f}")
    print(f"  Average retrieval time per query: {metrics['avg_time']:.4f} seconds")
else:
    print("\nEvaluation skipped.")

