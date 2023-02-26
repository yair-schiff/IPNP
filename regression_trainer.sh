#!/bin/bash

#SBATCH
#SBATCH -N 1                                 # Total number of nodes requested
#SBATCH -n 1                                 # Total number of cores requested
#SBATCH --get-user-env                       # retrieve the users login environment
#SBATCH --mem=75G                            # server memory requested (per node; 1000M ~= 1G)
#SBATCH --gres=gpu:3090:1                    # Type/number of GPUs needed
#SBATCH --time=12:00:00                      # Set max runtime for job
#SBATCH --requeue                            # Requeue job


export PYTHONPATH="${PWD}:${PWD}/regression"

# shellcheck source=${HOME}/.bashrc
source "${CONDA_SHELL}"
conda activate tnp
cd ./regression || exit

python trainer.py "${exp}" "${mode}" \
  --train_seed="${seed}" \
  --eval_seed="${seed}" \
  --model="${model}" \
  --expid="${expid}" \
  --lr="${lr}" \
  --min_lr="${min_lr}" \
  --num_epochs="${num_epochs}" \
  --annealer_mult="${annealer_mult}" \
  --max_num_ctx="${max_num_ctx}" \
  --min_num_ctx="${min_num_ctx}" \
  --max_num_tar="${max_num_tar}" \
  --min_num_tar="${min_num_tar}" \
  ${exp_specific_args}
conda deactivate
