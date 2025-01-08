#!/bin/bash
#SBATCH --job-name=intro_research
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --time=0-12:00:00
#SBATCH --output=logs/%j.out      
#SBATCH --error=logs/%j.err       
#SBATCH --mem=16000MB
#SBATCH --cpus-per-task=2
source ~/data/.bashrc
source ~/data/miniconda3/etc/profile.d/conda.sh

conda activate wandb-tutorial
srun python wandb_tutorial_sbatch.py