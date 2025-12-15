#!/bin/bash
#SBATCH --account=msph
#SBATCH --job-name=create_trainingset
#SBATCH --array=0           # 7 seps below → index 0..6
#SBATCH -N 1
#SBATCH -c 4
#SBATCH --mem=16G
#SBATCH -t 01:00:00
#SBATCH -o logs/create_%A_%a.out
#SBATCH -e logs/create_%A_%a.err

module load anaconda
source /insomnia001/shared/apps/anaconda/2023.09/etc/profile.d/conda.sh
conda activate ~/my_envs/superspreading_ml

mkdir -p logs

R_VAL="0.6"
G_IDX="11"

# Paths
EXPORT_DIR="/insomnia001/depts/msph/users/qy2290/Rtx_rr_pop_sig/r-${R_VAL}_${G_IDX}_simulation_res/"
W_PATH="./source_r-x/W_avg.csv"
SAVE_DIR="/insomnia001/depts/msph/users/qy2290/Rtx_rr_pop_sig/gnn_inference_data/"


SEPS=(49)
SEP=${SEPS[$SLURM_ARRAY_TASK_ID]}

python create_trainingset.py \
  --export-dir "$EXPORT_DIR" \
  --W-path "$W_PATH" \
  --save-dir "$SAVE_DIR" \
  --r "$R_VAL" \
  --g-idx "$G_IDX" \
  --sep "$SEP"
