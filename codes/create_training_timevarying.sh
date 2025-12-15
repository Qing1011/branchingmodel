#!/bin/bash
#SBATCH --account=msph
#SBATCH --job-name=mk_ds_timevary
#SBATCH -N 1
#SBATCH --time=0-10:00
#SBATCH --mem=16G
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail


module load anaconda
source /insomnia001/shared/apps/anaconda/2023.09/etc/profile.d/conda.sh
conda activate ~/my_envs/superspreading_ml

mkdir -p logs

# avoid BLAS oversubscription (not strictly needed, but harmless)
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1

# run the minimal builder (uses the hard-coded paths inside the .py)
# python create_training_timevarying.py
python create_training_timevarying.py
