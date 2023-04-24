#!/bin/bash
#$ -cwd -S /bin/bash
#$ -l mem=2G
#$ -l time=18:30:00
#$ -pe smp 4
#$ -N test -j y
#$ -M qy2290@columbia.edu -m aes

module load anaconda/conda3                      # load anaconda.
source activate branching_qing                    # activate your environment.
cd /ifs/home/msph/ehs/qy2290/branching/ # move to your project directory.

python branching_cluster.py 1