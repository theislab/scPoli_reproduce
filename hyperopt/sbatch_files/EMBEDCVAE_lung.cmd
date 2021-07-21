#!/bin/bash
#SBATCH -J EMBEDCVAE_HYPEROPT_lung
#SBATCH -o /storage/groups/ml01/workspace/carlo.dedonno/lataq_reproduce/hyperopt/sbatch_files/EMBEDCVAE_lung.out
#SBATCH -e /storage/groups/ml01/workspace/carlo.dedonno/lataq_reproduce/hyperopt/sbatch_files/EMBEDCVAE_lung.err
#SBATCH -p gpu_p
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1
#SBATCH --exclude=icb-gpusrv0[1-2]
#SBATCH -c 2
#SBATCH -t 24:00:00
#SBATCH --mem=15G
#SBATCH --nice=10000

echo 
source /home/icb/carlo.dedonno/anaconda3/bin/activate lataq
python /storage/groups/ml01/workspace/carlo.dedonno/lataq_reproduce/hyperopt/embedcvae_hyperparam_search.py --experiment lung --n_epochs 20 --n_pre_epochs 4

