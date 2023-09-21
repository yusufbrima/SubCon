#!/bin/bash
$SBATCH --job-name="subspace-learning"
$SBATCH --cpu-per-task=128
#SBATCH --mem=128G
#SBATCH --time=48:00:00
#SBATCH --exclude=hpc3-16,hpc3-24,hpc3-12,hpc3-49,hpc3-45


