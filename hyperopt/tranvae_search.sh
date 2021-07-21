#!/bin/bash

EXPERIMENT=("pancreas" "pbmc" "scvelo" "brain" "tumor" "lung" "lung_h_sub")

for exp in "${EXPERIMENT[@]}"; do
    echo "$exp";
    sleep 0.1
    job_file="/storage/groups/ml01/workspace/carlo.dedonno/lataq_reproduce/hyperopt/sbatch_files/TRANVAE_${exp}.cmd"
                                        echo "#!/bin/bash
#SBATCH -J TRANVAE_HYPEROPT_${exp}
#SBATCH -o /storage/groups/ml01/workspace/carlo.dedonno/lataq_reproduce/hyperopt/sbatch_files/TRANVAE_${exp}.out
#SBATCH -e /storage/groups/ml01/workspace/carlo.dedonno/lataq_reproduce/hyperopt/sbatch_files/TRANVAE_${exp}.err
#SBATCH -p gpu_p
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1
#SBATCH --exclude=icb-gpusrv0[1-2]
#SBATCH -c 2
#SBATCH -t 24:00:00
#SBATCH --mem=15G
#SBATCH --nice=10000

echo $SLURM_NODENAME
source /home/icb/carlo.dedonno/anaconda3/bin/activate lataq
python /storage/groups/ml01/workspace/carlo.dedonno/lataq_reproduce/hyperopt/tranvae_hyperparam_search.py --experiment "$exp" --n_epochs 20 --n_pre_epochs 4
" > ${job_file}
                    sbatch $job_file
done
