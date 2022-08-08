#!/bin/sh
#SBATCH -J UMAP_LATAQ
#SBATCH -o /home/icb/carlo.dedonno/projects/lataq_reproduce/figure_notebooks/scripts/healthy_umap.out
#SBATCH -e /home/icb/carlo.dedonno/projects/lataq_reproduce/figure_notebooks/scripts/healthy_umap.err
#SBATCH -p gpu_p
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1
#SBATCH --exclude=supergpu02pxe

#SBATCH -c 4
#SBATCH -t 2-00:00:00
#SBATCH --mem=50G
#SBATCH --nice=1000

echo $SLURM_NODENAME
source /home/icb/carlo.dedonno/anaconda3/bin/activate lataq_cuda
python /home/icb/carlo.dedonno/projects/lataq_reproduce/figure_notebooks/scripts/healthy_umap_dataset.py