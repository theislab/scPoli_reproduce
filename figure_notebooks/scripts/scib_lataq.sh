#!/bin/sh
#SBATCH -J scib_lataq
#SBATCH -o /home/icb/carlo.dedonno/projects/lataq_reproduce/figure_notebooks/scripts/scib_lataq_amir.out
#SBATCH -e /home/icb/carlo.dedonno/projects/lataq_reproduce/figure_notebooks/scripts/scib_lataq_amir.err

#SBATCH -p gpu_p
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1
#SBATCH --exclude=supergpu02pxe


#SBATCH -t 2-00:00:00
#SBATCH --nice=1000


echo $SLURM_NODENAME
source /home/icb/carlo.dedonno/anaconda3/bin/activate lataq_cuda
python /lustre/groups/ml01/workspace/carlo.dedonno/lataq_reproduce/figure_notebooks/scripts/scib_hlca_lataq_amir.py