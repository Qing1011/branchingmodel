#!/bin/bash
#SBATCH --account=msph
#SBATCH --job-name=branching_rtx_r
#SBATCH --array=0-299              # Submit 300 array jobs 0-299 
#SBATCH -N 1
#SBATCH -c 8
#SBATCH --time=0-5:00
#SBATCH --mem-per-cpu=16G
#SBATCH --output=logs/output_%A_%a.out
#SBATCH --error=logs/error_%A_%a.err

module load anaconda
source /insomnia001/shared/apps/anaconda/2023.09/etc/profile.d/conda.sh
conda activate ~/my_envs/superspreading_ml

# Get ensemble index from SLURM
es_idx=${SLURM_ARRAY_TASK_ID}

# Launch 20 r values in parallel using 20 cores
for s in $(seq 0 20); do
    python branching_Rx_seeds.py $s $es_idx 2.0 11 0.15 'rr_pop_sig/' '../test_data/rr_pop_sig/' '/insomnia001/depts/msph/users/qy2290/Rtx_rr_pop_sig/'&
done

wait  # Wait for all background jobs to finish
