#!/bin/bash
#$ -cwd -S /bin/bash
#$ -l mem=2G
#$ -l time=13:00:00
#$ -N test -j y


module load anaconda/conda3                      # load anaconda.
source activate branching_qing                    # activate your environment.
cd /ifs/home/msph/ehs/qy2290/branching/ # move to your project directory.

python branching_cluster.py $SGE_TASK_ID