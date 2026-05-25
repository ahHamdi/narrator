import json
import logging
import pickle
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.neighbors import NearestNeighbors

sys.path.insert(0, str(Path(__file__).parent.parent))
from narrativebench import load_jsonl
from narrativebench.data import build_split_indices

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger(__name__)


def run_knn(data_dir: Path, matrix_file: str, k_values: list[int]):
    # Load matrix
    with open(data_dir / matrix_file, "rb") as f:
        obj = pickle.load(f)
    matrix     = obj["data"].astype(np.float32)
    record_ids = list(obj["record_ids"])
    matrix     = matrix[[i for i in range(matrix.shape[0]) if i != 8], :]
    N          = matrix.shape[0]

    # Load splits
    items_train = load_jsonl(data_dir / "benchmark_train.jsonl")
    items_test  = load_jsonl(data_dir / "benchmark_test.jsonl")
    train_idx, _, test_idx = build_split_indices(
        record_ids, items_train, [], items_test
    )

    Y_train = matrix[:, train_idx]
    Y_test  = matrix[:, test_idx]

    # Load ST embeddings
    cache = data_dir / "cache" / "st_all.npy"
    if not cache.exists():
        log.error("ST cache not found — run train.py first to generate st_all.npy")
        return

    st_all   = np.load(cache).astype(np.float32)
    n_tr     = len(items_train)
    st_train = st_all[:n_tr]
    st_test  = st_all[n_tr + len([]) :][:len(items_test)]

    results = {}
    for k in k_values:
        log.info(f"Building KNN index (k={k}) ...")
        knn = NearestNeighbors(n_neighbors=k, metric="cosine", n_jobs=-1)
        knn.fit(st_train)

        log.info(f"Predicting with k={k} ...")
        _, neighbors = knn.kneighbors(st_test)

        preds, labels = [], []
        T_test = len(items_test)
        for j in range(T_test):
            if j % 1000 == 0:
                log.info(f"  {j}/{T_test}")
            nb = neighbors[j]
            for i in range(N):
                nb_labels = Y_train[i, nb]
                nb_labels = nb_labels[nb_labels >= 0]
                if len(nb_labels) == 0:
                    continue
                if Y_test[i, j] < 0:
                    continue
                preds.append(int(nb_labels.mean() > 0.5))
                labels.append(int(Y_test[i, j]))

        acc = accuracy_score(labels, preds)
        results[f"k={k}"] = float(acc)
        log.info(f"KNN k={k}: accuracy={acc:.4f}")

    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir",    default="data/")
    parser.add_argument("--matrix_file", default="prediction_matrix.pkl")
    parser.add_argument("--k",           nargs="+", type=int, default=[5, 10, 20])
    args = parser.parse_args()

    results = run_knn(Path(args.data_dir), args.matrix_file, args.k)
    if results:
        log.info("\n=== KNN Results ===")
        for k, acc in results.items():
            log.info(f"  {k}: {acc:.4f}")
