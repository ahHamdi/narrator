#!/bin/bash
#SBATCH --job-name=nb_grid
#SBATCH --array=0-15
#SBATCH --partition=***
#SBATCH --account=***
#SBATCH --qos=***
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --output=logs/grid-%A-%a.out
#SBATCH --error=logs/grid-%A-%a.err

cd $PROJECT
module purge
module load pytorch-gpu/py3/2.1.1
export PYTHONPATH=$WORK/pip_packages:$PYTHONPATH
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

python scripts/gridsearch.py --config_id $SLURM_ARRAY_TASK_ID --data_dir $PROJECT/data/ --output_dir $PROJECT/runs/grid/ --matrix_file prediction_matrix_np1.pkl
