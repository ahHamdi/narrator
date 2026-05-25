#!/bin/bash
#SBATCH --job-name=nb_ablation
#SBATCH --partition=***
#SBATCH --account=***
#SBATCH --qos=***
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --output=logs/ablation-%j.out
#SBATCH --error=logs/ablation-%j.err

cd $PROJECT
module purge
module load pytorch-gpu/py3/2.1.1
export PYTHONPATH=$WORK/pip_packages:$PYTHONPATH
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

python scripts/ablation.py --dim 128 --lr 1e-4 --batch_size 1024 --data_dir $PROJECT/data/ --output_dir $PROJECT/runs/ablation/ --matrix_file prediction_matrix_np1.pkl
