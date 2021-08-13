#!/bin/sh
#SBATCH -J SEURAT_SCORES
#SBATCH -o /storage/groups/ml01/workspace/carlo.dedonno/lataq_reproduce/benchmarks/seurat/scores.out
#SBATCH -e /storage/groups/ml01/workspace/carlo.dedonno/lataq_reproduce/benchmarks/seurat/scores.err
#SBATCH -p cpu_p

#SBATCH -c 12
#SBATCH -t 2-00:00
#SBATCH --mem=100G
#SBATCH --nice=1

echo $SLURM_NODENAME
source /home/icb/carlo.dedonno/anaconda3/bin/activate lataq_cuda

python /storage/groups/ml01/workspace/carlo.dedonno/lataq_reproduce/benchmarks/seurat/seurat_symphony_scores.py