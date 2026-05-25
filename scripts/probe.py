import json
import logging
import pickle
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from narrativebench import load_jsonl
from narrativebench.probing import run_all

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger(__name__)


def load_best_embeddings(runs_dir: Path):
    best_acc, best_embs, best_run = -1, None, None
    for f in runs_dir.glob("*/results.json"):
        r = json.load(open(f))
        acc = r["routing"].get("MF-1", 0)
        if acc > best_acc:
            emb_path = f.parent / "model_embeddings.npy"
            if emb_path.exists():
                best_acc  = acc
                best_embs = np.load(emb_path)
                best_run  = f.parent
    log.info(f"Best embeddings: {best_embs.shape} from {best_run} (MF-1={best_acc:.4f})")
    return best_embs, best_run


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir",    default="data/")
    parser.add_argument("--runs_dir",    default="runs/final/")
    parser.add_argument("--output_dir",  default="probing/")
    parser.add_argument("--matrix_file", default="prediction_matrix.pkl")
    parser.add_argument("--model_index", default=None)
    args = parser.parse_args()

    np.random.seed(42)
    data_dir  = Path(args.data_dir)
    runs_dir  = Path(args.runs_dir)
    out_dir   = Path(args.output_dir)

    # Load embeddings
    model_embs, best_run = load_best_embeddings(runs_dir)

    # Load model names
    index_path = Path(args.model_index) if args.model_index else (
        best_run / "model_index.json" if best_run else None
    )
    if index_path and index_path.exists():
        idx = json.load(open(index_path))
        model_names = [k for k, v in sorted(idx.items(), key=lambda x: x[1])]
    else:
        model_names = [f"model_{i}" for i in range(len(model_embs))]
        log.warning("model_index.json not found — using generic names")

    # Load matrix
    with open(data_dir / args.matrix_file, "rb") as f:
        obj = pickle.load(f)
    matrix = obj["data"].astype(np.float32)
    matrix = matrix[[i for i in range(matrix.shape[0]) if i != 8], :]

    # Load test questions
    items_test = load_jsonl(data_dir / "benchmark_test.jsonl")

    # Load question embeddings
    q_embs = None
    if best_run:
        q_path = best_run / "question_embeddings_test.npy"
        if q_path.exists():
            q_embs = np.load(q_path)
            log.info(f"Question embeddings: {q_embs.shape}")
        else:
            log.warning("Question embeddings not found — run train.py first")

    run_all(model_embs, model_names, q_embs, items_test, matrix, out_dir)
