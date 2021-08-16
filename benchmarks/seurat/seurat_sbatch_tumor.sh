#!/bin/sh
#SBATCH -J SEURAT_LATAQ
#SBATCH -o /storage/groups/ml01/workspace/carlo.dedonno/lataq_reproduce/benchmarks/seurat/sbatch_tumor.out
#SBATCH -e /storage/groups/ml01/workspace/carlo.dedonno/lataq_reproduce/benchmarks/seurat/sbatch_tumor.err
#SBATCH -p cpu_p

#SBATCH -c 20
#SBATCH -t 2-00:00
#SBATCH --mem=150G
#SBATCH --nice=1

echo $SLURM_NODENAME
source /home/icb/carlo.dedonno/anaconda3/bin/activate lataq_r

sh /storage/groups/ml01/workspace/carlo.dedonno/lataq_reproduce/benchmarks/seurat/seurat_tumor.sh