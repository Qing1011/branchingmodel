#!/bin/bash
#SBATCH --job-name=infer_r_single
#SBATCH --account=msph
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH -t 01:00:00
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err
#SBATCH --array=0-1

module load anaconda
source /insomnia001/shared/apps/anaconda/2023.09/etc/profile.d/conda.sh
conda activate ~/my_envs/superspreading_ml

# If baysian_gcn.py is not next to this script, export its folder:
# export PYTHONPATH="/insomnia001/home/qy2290/branching_Rtx:$PYTHONPATH"

# Strict determinism (optional). If you enable torch.use_deterministic_algorithms(True) in training,
# set this env var *before* Python starts. Not strictly required for inference here.
# export CUBLAS_WORKSPACE_CONFIG=:4096:8

# EDIT these paths/values:
R_VALUES=(0.3 1.0 2.0 0.286 0.6 0.1 0.025 0.06) 
ES_IDXS=(111 7 11 999 999 1 1 999)

MODEL_PATHS=(
  "/insomnia001/home/qy2290/branching_Rtx/infer_results/r0.3_111_seed123_20251215-004801_best_model.pth"
  "/insomnia001/home/qy2290/branching_Rtx/infer_results/r1.0_7_seed123_20251215-004805_best_model.pth" 
  "/insomnia001/home/qy2290/branching_Rtx/infer_results/r2.0_11_seed123_20251213-155944_best_model.pth" 
  "/insomnia001/home/qy2290/branching_Rtx/infer_results/r0.286_seed123_20251007-002107_best_model.pth"
  "/insomnia001/home/qy2290/branching_Rtx/infer_results/r0.6_seed123_20251007-002107_best_model.pth"
  "/insomnia001/home/qy2290/branching_Rtx/infer_results/r0.1_seed123_20251007-002107_best_model.pth"
  "/insomnia001/home/qy2290/branching_Rtx/infer_results/r0.025_seed123_20251007-002048_best_model.pth"
  "/insomnia001/home/qy2290/branching_Rtx/infer_results/r0.06_seed123_20251007-002107_best_model.pth"
)

TASK_ID=${SLURM_ARRAY_TASK_ID}
R_VAL=${R_VALUES[$TASK_ID]}
ES_IDX=${ES_IDXS[$TASK_ID]}
MODEL_PATH=${MODEL_PATHS[$TASK_ID]}

echo "Running task $TASK_ID with R_VAL=${R_VAL}, ES_IDX=${ES_IDX}"
DATA_ROOT="/insomnia001/home/qy2290/branching_Rtx/test_data/rr_pop_sig/"   # contains branching_r-{r}/NewInf...
W_PATH="/insomnia001/home/qy2290/branching_Rtx/source_r-x/W_avg.csv"
# MODEL_PATH="/insomnia001/home/qy2290/branching_Rtx/trained/nrr_den/r2.0_seed123_20251118-234755_best_model.pth"

mkdir -p logs
# remembe to check the reporting rate setting in infer_single_realisation.py before running!
python infer_single_realisation.py \
  --r "$R_VAL" \
  --es-idx "$ES_IDX" \
  --data-root "$DATA_ROOT" \
  --W-path "$W_PATH" \
  --model-path "$MODEL_PATH" \
  --t-start 10 \
  --t-len 49 \
  --samples 1000 \
  --seed 123 \
  --sigma-floor 0.5 \
