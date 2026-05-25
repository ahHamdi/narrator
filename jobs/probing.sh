#!/bin/bash
#SBATCH --job-name=nb_probe
#SBATCH --partition=***
#SBATCH --account=***
#SBATCH --qos=***
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=02:00:00
#SBATCH --output=logs/probe-%j.out
#SBATCH --error=logs/probe-%j.err

cd $PROJECT
source narrativeBench/bin/activate
PYTHONPATH="" python scripts/probe.py --data_dir $PROJECT/data/ --runs_dir $PROJECT/runs/final/ --output_dir $PROJECT/probing/ --matrix_file prediction_matrix_np1.pkl
