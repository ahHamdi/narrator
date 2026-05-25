import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from narrativebench import Config, GRID
from scripts.train import main

import argparse
import logging

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger(__name__)


def run_grid(data_dir, output_dir, matrix_file, st_model, seed):
    """Run all 16 configurations sequentially."""
    for config_id, (dim, lr, bs) in enumerate(GRID):
        log.info(f"\n{'='*60}")
        log.info(f"Config {config_id+1}/16 — d={dim} lr={lr:.0e} bs={bs}")
        log.info(f"{'='*60}")
        cfg = Config(
            config_id=config_id,
            data_dir=data_dir,
            output_dir=output_dir,
            matrix_file=matrix_file,
            st_model=st_model,
            seed=seed,
        ).apply_grid()
        main(cfg)

    # Print summary
    results = []
    for f in Path(output_dir).glob("*/results.json"):
        r = json.load(open(f))
        results.append((
            r["routing"]["MF-1"],
            r["config"]["dim"],
            r["config"]["lr"],
            r["config"]["batch_size"],
            f.parent.name,
        ))
    results.sort(reverse=True)
    log.info("\n=== Grid Search Summary (sorted by MF-1) ===")
    log.info(f"{'Config':<30} {'dim':>5} {'lr':>8} {'bs':>6} {'MF-1':>8}")
    log.info("-" * 60)
    for acc, dim, lr, bs, name in results:
        log.info(f"{name:<30} {dim:>5} {lr:>8.0e} {bs:>6} {acc:>8.4f}")
    log.info(f"\nBest: {results[0][4]} (MF-1={results[0][0]:.4f})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_id",   type=int,   default=None,
                        help="Run single config (SLURM array mode)")
    parser.add_argument("--data_dir",    type=str,   default="data/")
    parser.add_argument("--output_dir",  type=str,   default="runs/grid/")
    parser.add_argument("--matrix_file", type=str,   default="prediction_matrix.pkl")
    parser.add_argument("--st_model",    type=str,
                        default="sentence-transformers/all-mpnet-base-v2")
    parser.add_argument("--seed",        type=int,   default=42)
    args = parser.parse_args()

    if args.config_id is not None:
        # SLURM array mode — single config
        cfg = Config(
            config_id=args.config_id,
            data_dir=args.data_dir,
            output_dir=args.output_dir,
            matrix_file=args.matrix_file,
            st_model=args.st_model,
            seed=args.seed,
        ).apply_grid()
        main(cfg)
    else:
        run_grid(
            args.data_dir, args.output_dir,
            args.matrix_file, args.st_model, args.seed,
        )
