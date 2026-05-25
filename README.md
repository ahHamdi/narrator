# Narrator

Narrative competence embeddings for LLMs via matrix factorisation.

## Installation

```bash
pip install -r requirement.txt
```

## Quick start

```bash
# 1. Train (single config)
python scripts/train.py --dim 128 --lr 1e-4 --batch_size 1024

# 2. Grid search (16 configs)
python scripts/gridsearch.py --output_dir runs/grid/

# 3. Ablation study
python scripts/ablation.py --dim 128 --lr 1e-4 --batch_size 1024

# 4. Probing experiments
python scripts/probe.py --runs_dir runs/final/ --output_dir probing/

# 5. KNN baseline
python scripts/knn_baseline.py
```

## Project structure

```
narrativebench/
├── narrativebench/
│   ├── config.py     # Config, GRID, constants
│   ├── data.py       # Matrix loading, JSONL, splits, features
│   ├── encoder.py    # Sentence transformer encoding
│   ├── dataset.py    # CorrectnessDataset
│   ├── model.py      # NarrativeMF (encoder-decoder)
│   ├── trainer.py    # Training loop with early stopping
│   ├── evaluator.py  # Routing, forecasting, benchmark prediction
│   └── probing.py    # Probing experiments 6a-6d
└── scripts/
    ├── train.py
    ├── gridsearch.py
    ├── ablation.py
    ├── probe.py
    └── knn_baseline.py
```
