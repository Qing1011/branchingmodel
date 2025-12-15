#!/bin/bash
#SBATCH --account=msph
#SBATCH --job-name=tv_bayes3_train
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



# ---------------- CONFIG ----------------
# DATA_PT="training/simulation_real_p1.pt"   # <-- set your dataset path
DATA_PT="training/simulation_real_p2.pt"   # dataset WITHOUT A
A_PT="training/A_t.pt"                     # A_t saved once
SAVE_DIR="./inferred_model/"

RESUME_CKPT="${SAVE_DIR}/best.pt"  
RESUME_RESET_LR=false              # true = ignore optimizer state, keep current LR

EPOCHS=300
# Bayesian & eval
SIGMA_FLOOR=0.5
KL_MAX=1e-3
KL_WARMUP_FRAC=0.8
VAL_MC=30
TEST_MC=100
VAL_SEED=123

# Early stopping
PATIENCE=20 #
MIN_DELTA=0.0

# Model shape (match your bayesian_gcn vibe)
H1=64
H2=64
H3=32
DROPOUT=0.5
AGG="mean"
NORM="none"
ADD_SELF_LOOPS=""     # set to "--add-self-loops" if you want it

# Optim
LR=1e-3
WEIGHT_DECAY=1e-6
ELBO_SAMPLES=3
# =======================

# Quick sanity print
python - <<'PY'
import sys, os, torch
print("PY:", sys.executable)
print("CUDA available:", torch.cuda.is_available())
print("CUDA device count:", torch.cuda.device_count())
print("Device 0:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu")
print("CWD:", os.getcwd())
PY

mkdir -p "$SAVE_DIR" logs

# ---------------- TRAIN ----------------
srun -u python -u train_timevarying.py \
  --data-pt "$DATA_PT" \
  --A-pt "$A_PT" \
  --save-dir "$SAVE_DIR" \
  --epochs $EPOCHS \
  --sigma-floor $SIGMA_FLOOR \
  --kl-max $KL_MAX \
  --kl-warmup-frac $KL_WARMUP_FRAC \
  --val-mc $VAL_MC \
  --test-mc $TEST_MC \
  --val-seed $VAL_SEED \
  --patience $PATIENCE \
  --min-delta $MIN_DELTA \
  --h1 $H1 --h2 $H2 --h3 $H3 \
  --dropout $DROPOUT \
  --agg "$AGG" \
  --norm "$NORM" \
  --lr $LR \
  --weight-decay $WEIGHT_DECAY \
  --elbo-samples $ELBO_SAMPLES \
  --resume "$RESUME_CKPT"


#  --resume "$RESUME_CKPT"
