import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from narrativebench import Config
from scripts.train import main

import argparse
import logging

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger(__name__)

ABLATIONS = [
    ("base",      True,  True),   # no_task_emb, no_feat
    ("task_only", False, True),
    ("feat_only", True,  False),
    ("full",      False, False),
]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dim",         type=int,   default=128)
    parser.add_argument("--lr",          type=float, default=1e-4)
    parser.add_argument("--batch_size",  type=int,   default=1024)
    parser.add_argument("--max_epochs",  type=int,   default=50)
    parser.add_argument("--data_dir",    type=str,   default="data/")
    parser.add_argument("--output_dir",  type=str,   default="runs/ablation/")
    parser.add_argument("--matrix_file", type=str,   default="prediction_matrix.pkl")
    parser.add_argument("--st_model",    type=str,
                        default="sentence-transformers/all-mpnet-base-v2")
    parser.add_argument("--seed",        type=int,   default=42)
    args = parser.parse_args()

    for name, no_task, no_feat in ABLATIONS:
        log.info(f"\n{'='*60}")
        log.info(f"Ablation: {name}")
        log.info(f"{'='*60}")
        cfg = Config(
            dim=args.dim, lr=args.lr, batch_size=args.batch_size,
            max_epochs=args.max_epochs,
            data_dir=args.data_dir,
            output_dir=str(Path(args.output_dir) / name),
            matrix_file=args.matrix_file,
            st_model=args.st_model,
            seed=args.seed,
            no_task_emb=no_task,
            no_feat=no_feat,
        )
        main(cfg)
