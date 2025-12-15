#!/usr/bin/env bash
set -e

# (optional) activate your env
# source "$(conda info --base)"/etc/profile.d/conda.sh
# conda activate ~/my_envs/superspreading_ml

# helpful defaults for Mac + PyTorch
# Use CPU efficiently on M3 Max
# export OMP_NUM_THREADS=4
# export OPENBLAS_NUM_THREADS=4
# export VECLIB_MAXIMUM_THREADS=4
# export NUMEXPR_NUM_THREADS=4

# Enable safe MPS fallback
# export PYTORCH_ENABLE_MPS_FALLBACK=1   # harmless if not using MPS


# MAX_PAR=5        # how many Python jobs to run at once
R0=2.5
SEED=10

mkdir -p logs

for es in {47..299}; do
  echo "es_idx=$es"
  for s in {0..8}; do
    python branching_R0.py "$s" "$es" "$R0" "$SEED" \
      >"logs/output_${es}_${s}.out" 2>"logs/error_${es}_${s}.err" &
  done
  wait   # wait for the 9 jobs of this es to finish
done

echo "All done."
# #!/bin/bash
# #SBATCH --account=msph
# #SBATCH --job-name=branching_rtx_r-2.0
# #SBATCH --array=0-299              # Submit 300 array jobs 0-299 
# #SBATCH -N 1
# #SBATCH --time=0-5:00
# #SBATCH --mem-per-cpu=4G
# #SBATCH --output=logs/output_%A_%a.out
# #SBATCH --error=logs/error_%A_%a.err


# module load anaconda
# source /insomnia001/shared/apps/anaconda/2023.09/etc/profile.d/conda.sh
# conda activate ~/my_envs/superspreading_ml

# # Get ensemble index from SLURM
# es_idx=${SLURM_ARRAY_TASK_ID}

# # Launch 20 r values in parallel using 20 cores

# for s in $(seq 0 8); do
#     python branching_R0.py $s $es_idx 2.5 3 &
# done

# # wait  # Wait for all background jobs to finish
# wait