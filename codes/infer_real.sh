#!/bin/bash
#SBATCH --account=msph
#SBATCH --job-name=infer_real
#SBATCH --gres=gpu:1
#SBATCH -t 12:00:00
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err

set -euo pipefail
mkdir -p logs

module load anaconda
source /insomnia001/shared/apps/anaconda/2023.09/etc/profile.d/conda.sh
conda activate ~/my_envs/superspreading_ml

# Avoid BLAS oversubscription
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1

# ---- edit these paths ----
CKPT="inferred_model/best.pt"
XPATH="real_inf.npy"          # (N, T_total) .npy or .npy.gz

# Choose ONE of the following (comment the other):
APT="training/A_t_1.pt"              # torch .pt (T_total, N, N)
WPATH=""                             # OR HDF5 with dataset 'WM' (T_total, N, N)

# Window
T0=0
T=21

# Inference options
MC=100
SIGMA_FLOOR=""                       # leave empty to use value stored in ckpt; or e.g. "0.5"
MC_SEED=0
JSON_OUT="infer_${T0}_${T}.json"

# ---- runtime env (edit for your cluster/venv) ----
# Example: source your python env
# source /insomnia001/home/qy2290/my_envs/superspreading_ml/bin/activate
# export CUDA_VISIBLE_DEVICES=0
# export TORCH_CUDA_ARCH_LIST="8.9"  # if you care

# ---- build CLI ----
CLI=( python infer_real.py
  --ckpt "$CKPT"
  --x-path "$XPATH"
  --t0 "$T0"
  --T "$T"
  --mc "$MC"
  --mc-seed "$MC_SEED"
  --json-out "$JSON_OUT"
)

if [[ -n "${SIGMA_FLOOR}" ]]; then
  CLI+=( --sigma-floor "$SIGMA_FLOOR" )
fi

if [[ -n "${APT}" && -z "${WPATH}" ]]; then
  CLI+=( --A-pt "$APT" )
elif [[ -n "${WPATH}" && -z "${APT}" ]]; then
  CLI+=( --W-path "$WPATH" )
else
  echo "ERROR: Provide exactly one of APT or WPATH" >&2
  exit 1
fi

echo "Running: ${CLI[*]}"
"${CLI[@]}"