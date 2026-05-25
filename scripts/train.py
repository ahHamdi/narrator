import argparse
import json
import logging
import os
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from narrativebench import (
    Config, load_matrix, load_jsonl, build_split_indices,
    encode_questions, train,
    evaluate_routing, evaluate_forecasting, evaluate_benchmark_prediction,
    run_probing,
)
from narrativebench.data import build_task_ids, build_question_features

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


def main(cfg: Config):
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Device: {device} | Config: {cfg.run_name()}")

    out_dir   = Path(cfg.output_dir) / cfg.run_name()
    out_dir.mkdir(parents=True, exist_ok=True)
    data_dir  = Path(cfg.data_dir)
    cache_dir = data_dir / "cache"
    cache_dir.mkdir(exist_ok=True)

    # ── Load data ──
    matrix, model_names, record_ids = load_matrix(data_dir / cfg.matrix_file)
    N, T = matrix.shape

    items_train = load_jsonl(data_dir / "benchmark_train.jsonl")
    items_val   = load_jsonl(data_dir / "benchmark_val.jsonl")
    items_test  = load_jsonl(data_dir / "benchmark_test.jsonl")
    items_all   = items_train + items_val + items_test

    train_idx, val_idx, test_idx = build_split_indices(
        record_ids, items_train, items_val, items_test
    )

    # Save model index
    model_index = {n: i for i, n in enumerate(model_names)}
    with open(out_dir / "model_index.json", "w") as f:
        json.dump(model_index, f, indent=2)

    # ── Sentence transformer encoding (cached) ──
    st_all = encode_questions(
        items_all, cfg.st_model, cfg.st_batch,
        chunk_size=cfg.st_chunk,
        cache_path=str(cache_dir / "st_all.npy"),
    )
    n_tr = len(items_train)
    n_vl = len(items_val)
    st_train, st_val, st_test = st_all[:n_tr], st_all[n_tr:n_tr+n_vl], st_all[n_tr+n_vl:]
    del st_all

    # ── Task IDs and features ──
    task_train = build_task_ids(items_train)
    task_val   = build_task_ids(items_val)
    task_test  = build_task_ids(items_test)

    diff_train = 1.0 - matrix[:, train_idx].mean(axis=0)
    feat_train = build_question_features(items_train, diff_train)
    feat_val   = build_question_features(items_val)
    feat_test  = build_question_features(items_test)

    # ── Train ──
    log.info(f"Training: d={cfg.dim} lr={cfg.lr} bs={cfg.batch_size}")
    model, history = train(
        cfg, matrix,
        train_idx, val_idx,
        st_train, st_val,
        task_train, task_val,
        feat_train, feat_val,
        device,
    )

    # ── Save model and embeddings ──
    torch.save(model.state_dict(), out_dir / "model.pt")
    embs = model.get_model_embeddings()
    np.save(out_dir / "model_embeddings.npy", embs)
    log.info(f"Model embeddings: {embs.shape}")

    # Save question embeddings for probing
    log.info("Saving question embeddings ...")
    st_t   = torch.tensor(st_test,   dtype=torch.float32)
    task_t = torch.tensor(task_test, dtype=torch.long)
    feat_t = torch.tensor(feat_test, dtype=torch.float32)
    q_embs = model.get_question_embeddings(st_t, task_t, feat_t, device)
    np.save(out_dir / "question_embeddings_test.npy", q_embs)
    log.info(f"Question embeddings: {q_embs.shape}")

    # ── Evaluate ──
    routing_results, S = evaluate_routing(
        model, matrix, test_idx,
        st_test, task_test, feat_test, items_test, device,
    )
    np.save(out_dir / "score_matrix.npy", S)

    forecast_results = evaluate_forecasting(
        model, matrix, test_idx,
        st_test, task_test, feat_test, device,
    )

    bench_results = evaluate_benchmark_prediction(embs, matrix)

    # ── Save results ──
    all_results = {
        "config":               cfg.to_dict(),
        "routing":              routing_results,
        "forecasting":          forecast_results,
        "benchmark_prediction": bench_results,
        "history":              history,
    }
    with open(out_dir / "results.json", "w") as f:
        json.dump(all_results, f, indent=2)
    log.info(f"Results → {out_dir}/results.json")

    return model, embs, all_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dim",         type=int,   default=128)
    parser.add_argument("--lr",          type=float, default=1e-4)
    parser.add_argument("--batch_size",  type=int,   default=1024)
    parser.add_argument("--max_epochs",  type=int,   default=50)
    parser.add_argument("--patience",    type=int,   default=5)
    parser.add_argument("--config_id",   type=int,   default=None)
    parser.add_argument("--data_dir",    type=str,   default="data/")
    parser.add_argument("--output_dir",  type=str,   default="runs/")
    parser.add_argument("--matrix_file", type=str,   default="prediction_matrix.pkl")
    parser.add_argument("--seed",        type=int,   default=42)
    parser.add_argument("--st_model",    type=str,
                        default="sentence-transformers/all-mpnet-base-v2")
    parser.add_argument("--no_task_emb", action="store_true")
    parser.add_argument("--no_feat",     action="store_true")
    args = parser.parse_args()

    cfg = Config(
        dim=args.dim, lr=args.lr, batch_size=args.batch_size,
        max_epochs=args.max_epochs, patience=args.patience,
        config_id=args.config_id,
        data_dir=args.data_dir, output_dir=args.output_dir,
        matrix_file=args.matrix_file,
        seed=args.seed, st_model=args.st_model,
        no_task_emb=args.no_task_emb, no_feat=args.no_feat,
    ).apply_grid()

    main(cfg)
