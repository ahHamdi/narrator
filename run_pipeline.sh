#!/bin/bash

PROJECT=your_project_folder
mkdir -p $PROJECT/logs $PROJECT/runs $PROJECT/probing

# Copy scripts to project
cp train_mf.py   $PROJECT/scripts/train_mf.py
cp probing.py    $PROJECT/scripts/probing.py
cp 01_gridsearch.sh $PROJECT/jobs/01_gridsearch.sh
cp 02_ablation.sh   $PROJECT/jobs/02_ablation.sh
cp 03_probing.sh    $PROJECT/jobs/03_probing.sh

cd $PROJECT

# ── Job 1: Grid search (16 configs in parallel) ──
GS_JOB=$(sbatch --parsable jobs/01_gridsearch.sh)
echo "Grid search submitted: job $GS_JOB (16 array tasks)"

# ── Job 2: Ablation (depends on all grid search tasks) ──
ABL_JOB=$(sbatch --parsable \
    --dependency=afterok:$GS_JOB \
    --export=ALL,GRIDSEARCH_JOB_ID=$GS_JOB \
    jobs/02_ablation.sh)
echo "Ablation submitted:    job $ABL_JOB (depends on $GS_JOB)"

# ── Job 3: Probing (depends on ablation) ──
PROBE_JOB=$(sbatch --parsable \
    --dependency=afterok:$ABL_JOB \
    --export=ALL,ABLATION_JOB_ID=$ABL_JOB \
    jobs/03_probing.sh)
echo "Probing submitted:     job $PROBE_JOB (depends on $ABL_JOB)"

echo ""
echo "=== Pipeline submitted ==="
echo "Monitor with: squeue -u jmoreno"
echo "Results will appear in: $PROJECT/runs/ and $PROJECT/probing/"
echo ""
echo "Expected total wall time: ~3-4h"
echo "  Grid search:  ~2h  (16 configs × ~8min each, parallel)"
echo "  Ablation:     ~1h  (4 configs × ~15min each, parallel)"
echo "  Probing:      ~30min (CPU only)"
