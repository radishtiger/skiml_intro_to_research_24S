#!/bin/bash
#SBATCH --job-name=intro_research
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --time=0-12:00:00
#SBATCH --output=/home/moohosong/data/skiml_intro_to_research_24S/logs/%j.out         # Adjustement required: Write directory you want
#SBATCH --error=/home/moohosong/data/skiml_intro_to_research_24S/logs/%j.err          # Adjustement required: Write directory you want
#SBATCH --mem=64000MB
#SBATCH --cpus-per-task=2
source /home/${USER}/data/.bashrc
source ~/data/miniconda3/etc/profile.d/conda.sh

srun python wandb_tutorial_sbatch.py