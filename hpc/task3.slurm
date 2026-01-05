#!/bin/bash
#SBATCH --job-name=dl_coding_task3
#SBATCH --time=24:00:00
#SBATCH --mem=200GB
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:1
#SBATCH --partition=paula

# load cuda module
module load CUDA

# load anaconda module and activate conda env
module load Anaconda3
conda init
conda activate dl_coding_task

WORKD_DIR=/work/mb92adej-dl

# task3.1
~/.conda/envs/dl_coding_task/bin/python -m src.task3.fine_tune -D $WORKD_DIR/dat -R $WORKD_DIR
