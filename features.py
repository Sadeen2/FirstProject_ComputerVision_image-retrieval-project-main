# src/evaluate.py

from pathlib import Path
import time
import numpy as np

from dataset import list_images
from retrieve import retrieve_similar_images_tfidf


def evaluate_retrieval(
    ks=(1, 5, 10),
    max_queries=50,
    top_k_for_retrieval=200,
    metric="cosine",
    random_seed: int = 42,
):
    """
    Evaluate the retrieval system on random query images.

    Returns a dictionary with:
      - "mAP"
      - "avg_time"
      - "P@K" and "R@K" for each K in ks
    """
    # 1) Load all (path, label) from Caltech-101
    items = list_images()
    paths = [p for (p, lbl) in items]
    labels = [lbl for (p, lbl) in items]

    n_images = len(items)
    print(f"[evaluate] Total images in dataset: {n_images}")

    # 2) Choose random indices as query images
    rng = np.random.default_rng(random_seed)
    n_queries = min(max_queries, n_images)

    query_indices = rng.choice(n_images, size=n_queries, replace=False)
    print(f"[evaluate] Using {n_queries} random queries")

    # -------------------------------------------------
    # For each random query:
    #   - run retrieval
    #   - compare labels
    #   - accumulate Precision@K, Recall@K, AP, time
    # -------------------------------------------------
    all_AP = []
    all_P_at_k = {k: [] for k in ks}
    all_R_at_k = {k: [] for k in ks}
    times = []

    for qi, idx in enumerate(query_indices):
        query_path = paths[idx]
        query_label = labels[idx]

        t0 = time.time()
        results = retrieve_similar_images_tfidf(
            query_image_path=query_path,
            top_k=top_k_for_retrieval,
            metric=metric,
        )
        t1 = time.time()
        times.append(t1 - t0)

        # build list of relevant flags (same class label)
        relevant = np.array(
            [1 if r["label"] == query_label else 0 for r in results],
            dtype=np.int32,
        )
        cum_relevant = np.cumsum(relevant)
        total_relevant = cum_relevant[-1] if cum_relevant.size > 0 else 0

        # Precision@K and Recall@K
        for k in ks:
            k_eff = min(k, len(results))
            if k_eff == 0:
                P_k = 0.0
                R_k = 0.0
            else:
                num_rel_at_k = cum_relevant[k_eff - 1]
                P_k = num_rel_at_k / float(k_eff)
                R_k = num_rel_at_k / float(max(total_relevant, 1))
            all_P_at_k[k].append(P_k)
            all_R_at_k[k].append(R_k)

        # Average Precision (AP)
        if total_relevant == 0:
            AP = 0.0
        else:
            precisions = []
            for rank, rel_flag in enumerate(relevant, start=1):
                if rel_flag == 1:
                    precisions.append(cum_relevant[rank - 1] / float(rank))
            AP = float(np.mean(precisions)) if precisions else 0.0
        all_AP.append(AP)

        print(
            f"[{qi+1}/{n_queries}] "
            f"query='{query_label}'  AP={AP:.3f}, time={t1-t0:.3f}s"
        )

    # ---- aggregate & print results ----
    mAP = float(np.mean(all_AP)) if all_AP else 0.0
    avg_time = float(np.mean(times)) if times else 0.0

    metrics = {
        "mAP": mAP,
        "avg_time": avg_time,
    }

    print("\n========== Evaluation Results ==========")
    print(f"mAP:           {mAP:.4f}")
    for k in ks:
        P_mean = float(np.mean(all_P_at_k[k])) if all_P_at_k[k] else 0.0
        R_mean = float(np.mean(all_R_at_k[k])) if all_R_at_k[k] else 0.0
        print(f"P@{k}:          {P_mean:.4f}")
        print(f"R@{k}:          {R_mean:.4f}")
        metrics[f"P@{k}"] = P_mean
        metrics[f"R@{k}"] = R_mean
    print(f"Avg time/query: {avg_time:.4f} seconds")
    print("========================================")

    return metrics   


# ---------------------------------------------------------
# Run directly:  python -m src.evaluate
# ---------------------------------------------------------
if __name__ == "__main__":
    metrics = evaluate_retrieval(
        ks=(1, 5, 10),
        max_queries=20,
        top_k_for_retrieval=200,
        metric="cosine",
    )

    # Extra clear printout of the same metrics
    print("\n5. Evaluation Measures (results):")
    print(f"  Mean Average Precision (mAP): {metrics['mAP']:.4f}")
    for k in (1, 5, 10):
        print(f"  Precision@{k}: {metrics[f'P@{k}']:.4f}")
        print(f"  Recall@{k}:    {metrics[f'R@{k}']:.4f}")
    print(f"  Average retrieval time per query: {metrics['avg_time']:.4f} seconds")