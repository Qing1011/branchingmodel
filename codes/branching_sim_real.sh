#!/bin/bash
#SBATCH --account=msph
#SBATCH --job-name=branching_sim_real_rtx_p2
#SBATCH --array=0-1             # 0-299 300 ensembles
#SBATCH -N 1
#SBATCH --time=0-12:00
#SBATCH --mem-per-cpu=16G
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err

set -euo pipefail

# module load anaconda
# source /insomnia001/shared/apps/anaconda/2023.09/etc/profile.d/conda.sh
# conda activate ~/my_envs/superspreading_ml
# unset PYTHONPATH

mkdir -p logs


module load anaconda
source /insomnia001/shared/apps/anaconda/2023.09/etc/profile.d/conda.sh
conda activate ~/my_envs/superspreading_ml


ES_IDX="${SLURM_ARRAY_TASK_ID}"
for s in $(seq 1 2); do  
  python branching_sim_real_p2.py "$s" "$ES_IDX" 
done
wait

# ES_IDX="${SLURM_ARRAY_TASK_ID}"

# # r_list in branching_sim_real.py has 21 entries -> indices 0..20
# R_START=0
# R_END=20


# running=0
# for s in $(seq "$R_START" "$R_END"); do
#   # use srun to give each Python process its own core
#   python branching_sim_real.py "$s" "$ES_IDX" 
# done

# # wait for any remaining tasks
# wait
echo "Ensemble ${ES_IDX} done."
