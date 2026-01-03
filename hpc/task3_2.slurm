#!/bin/bash
#SBATCH --job-name=dl_coding_task3_2
#SBATCH --output=/work/mb92adej-dl/task3_2.out
#SBATCH --error=/work/mb92adej-dl/task3_2.err
#SBATCH --time=24:00:00
#SBATCH --mem=200GB
#SBATCH --cpus-per-task=12
#SBATCH --partition=paula
#SBATCH --mail-user=mb92adej@studserv.uni-leipzig.de
#SBATCH --mail-type=ALL

# load anaconda module and activate conda env
module load Anaconda3
conda init
conda activate dl_coding_task

# task3.1
~/.conda/envs/dl_coding_task/bin/python -m src.task3.reproduce -D /work/mb92adej-dl/dat -R /work/mb92adej-dl