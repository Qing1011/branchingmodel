#!/bin/bash
#SBATCH --job-name=bayes_gcn_R0
#SBATCH --account=msph       # <-- replace with your actual account
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH -t 12:00:00
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err
#SBATCH --array=0-2

module load anaconda
source /insomnia001/shared/apps/anaconda/2023.09/etc/profile.d/conda.sh
conda activate ~/my_envs/superspreading_ml

# Args: my_r seed
# python infer_gpu.py 2.0 123

# list of r values
R_VALUES=(1.5 2.5 5.5)

# pick the right r for this array index
MY_R=${R_VALUES[$SLURM_ARRAY_TASK_ID]}

SEED=123

echo "Running with R0=${MY_R}, seed=${SEED}"
python infer_gpu_R0.py $MY_R $SEED