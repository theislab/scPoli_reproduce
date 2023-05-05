#!/bin/sh
#SBATCH -J scpoli
#SBATCH -o /lustre/groups/ml01/workspace/carlo.dedonno/lataq_reproduce_backup/lataq_reproduce/computational/computational.out
#SBATCH -e /lustre/groups/ml01/workspace/carlo.dedonno/lataq_reproduce_backup/lataq_reproduce/computational/computational.err
#SBATCH -p gpu_p
#SBATCH -t 10:00:00
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1
#SBATCH --nice=10000
#SBATCH -w supergpu02pxe

source activate scarches

sh /lustre/groups/ml01/workspace/carlo.dedonno/lataq_reproduce_backup/lataq_reproduce/computational/computational.sh