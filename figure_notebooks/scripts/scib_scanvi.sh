#!/bin/sh
#SBATCH -J scib_scanvi
#SBATCH -o /home/icb/carlo.dedonno/projects/lataq_reproduce/figure_notebooks/scripts/scib_scanvi.out
#SBATCH -e /home/icb/carlo.dedonno/projects/lataq_reproduce/figure_notebooks/scripts/scib_scanvi.err
#SBATCH -p gpu_p
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1
#SBATCH --exclude=supergpu02pxe

#SBATCH -c 4
#SBATCH -t 2-00:00:00
#SBATCH --mem=50G
#SBATCH --nice=1

echo $SLURM_NODENAME
source /home/icb/carlo.dedonno/anaconda3/bin/activate lataq_cuda
python /home/icb/carlo.dedonno/projects/lataq_reproduce/figure_notebooks/scripts/scib_hlca_scanvi.py