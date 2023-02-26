#!/bin/bash

# Arg parsing
while [ $# -gt 0 ]; do
    if [[ $1 == "--"* ]]; then
        v="${1/--/}"
        declare "$v"="$2"
        if [ -z ${2} ]; then
          declare "$v"="True"  # store_true params
          shift
        fi
    fi
    shift
done

programname=$0
function usage {
    echo ""
    echo "Run NP regression experiments."
    echo ""
    echo "usage: ${programname} --exp string --mode string --model string --expid string "
    echo ""
    echo "  --exp string            Which experiment to run: [gp | celeba]"
    echo "  --mode string           Usage: train/eval"
    echo "  --model string          Model to use"
    echo "  --expid string          Unique name to give experiment (optional)"
    echo ""
}

function die {
    printf "Script failed: %s\n\n" "${1}"
    exit 1
}

if [[ -z "${exp}" ]]; then
    usage
    die "Missing parameter --exp"
elif [[ -z "${mode}" ]]; then
    usage
    die "Missing parameter --mode"
elif [[ -z "${model}" ]]; then
    usage
    die "Missing parameter --model"
fi

base_results_dir="./regression/results"

# Start building command line exports
export_str="ALL,exp=${exp},mode=${mode},model=${model}"

if [[ -z "${seed}" ]]; then
  seed=0
  echo "Missing parameter --seed. Defaulting train and eval seeds to ${seed}."
fi
export_str="${export_str},seed=${seed}"
if [[ -z "${max_num_ctx}" ]]; then
  max_num_ctx=64
  echo "Missing parameter --max_num_ctx. Defaulting to ${max_num_ctx}."
fi
export_str="${export_str},max_num_ctx=${max_num_ctx}"
if [[ -z "${min_num_ctx}" ]]; then
  min_num_ctx=4
  echo "Missing parameter --min_num_ctx. Defaulting to ${min_num_ctx}."
fi
export_str="${export_str},min_num_ctx=${min_num_ctx}"
ctx_range="${min_num_ctx}-${max_num_ctx}"
if [[ -z "${max_num_tar}" ]]; then
  max_num_tar=64
  echo "Missing parameter --max_num_tar. Defaulting to ${max_num_tar}."
fi
export_str="${export_str},max_num_tar=${max_num_tar}"
if [[ -z "${min_num_tar}" ]]; then
  min_num_tar=4
  echo "Missing parameter --min_num_tar. Defaulting to ${min_num_tar}."
fi
export_str="${export_str},min_num_tar=${min_num_tar}"
tar_range="${min_num_tar}-${max_num_tar}"

if [[ -z "${lr}" ]]; then
  lr=5e-4
fi
export_str="${export_str},lr=${lr}"
if [[ -z "${num_epochs}" ]]; then
  if [[ "${exp}" == "gp" ]]; then
    num_epochs=100000
  else
    num_epochs=200
  fi
fi
export_str="${export_str},num_epochs=${num_epochs}"
if [[ -z "${min_lr}" ]]; then
  min_lr=0
fi
export_str="${export_str},min_lr=${min_lr}"
if [[ -z "${annealer_mult}" ]]; then
  annealer_mult=1
fi
export_str="${export_str},annealer_mult=${annealer_mult}"

exp_specific_args="--resume=resume"
if [[ -n "${eval_logfile}" ]]; then
  exp_specific_args="${exp_specific_args} --eval_logfile=${eval_logfile}"
fi
# GP:
if [[ "${exp}" == "gp" ]]; then
  if [[ -z "${eval_kernel}" ]]; then
    eval_kernel="rbf"
    echo "Missing GP parameter --eval_kernel. Defaulting to ${eval_kernel}."
  fi
  exp_specific_args="${exp_specific_args} --eval_kernel=${eval_kernel}"
  results_dir="${base_results_dir}/${exp}/ctx-${ctx_range}_tar-${tar_range}/${model}"

# CelebA:
elif [[ "${exp}" == "celeba" ]]; then
  if [[ -z "${resize}" ]]; then
    resize=64
    echo "Missing CelebA parameter --resize. Defaulting to ${resize}."
  fi
  exp_specific_args="${exp_specific_args} --resize=${resize}"
  if [[ -n "${target_all}" ]]; then
    exp_specific_args="${exp_specific_args} --target_all"
    tar_range="all"
  fi
  results_dir="${base_results_dir}/${exp}/${resize}x${resize}/ctx-${ctx_range}_tar-${tar_range}/${model}"

fi

export_str="${export_str},exp_specific_args=${exp_specific_args}"

# Build expid
if [[ -z "${expid}" ]]; then
    echo "Missing parameter --expid; expid will be created automatically."
    last_exp=-1
    if [ -d "${results_dir}" ] && [ "$(ls -A "${results_dir}")" ]; then  # Check if dir is exists and is not empty
      for dir in "${results_dir}/"*; do  # Loop over experiment versions and find most recent "v[expid]"
        if [[ ${dir} = "${results_dir}/v"* ]]; then  # Only look at dirs that mathc v[expid]
          dir=${dir%*/}      # remove the trailing "/"
          current_exp="${dir//$results_dir\/v/}"
          if [[ $current_exp -gt $last_exp ]]; then
            last_exp=$current_exp
          fi
        fi
      done
    fi
    expid="v$((last_exp+1))"
fi
export_str="${export_str},expid=${expid}"

# Build job name and make log dir
if [[ -z ${resize} ]]; then
  job_name="${exp}_${model}_${expid}_ctx-${ctx_range}_tar-${tar_range}"
else
  job_name="${exp}_${resize}x${resize}_${model}_${expid}_ctx-${ctx_range}_tar-${tar_range}"
fi
log_dir="${results_dir}/${expid}/logs"
mkdir -p "${log_dir}"

echo "Scheduling Job: ${job_name}"
sbatch \
  --job-name="${job_name}" \
  --output="${log_dir}/${mode}_%j.out" \
  --error="${log_dir}/${mode}_%j.err" \
  --export="${export_str}" \
  "regression_trainer.sh"
