#!/bin/bash
#SBATCH --job-name=dadi_joint # Name of the job
#SBATCH --output=logs/%x_%j.out # Stdout goes to logs/jobname_jobid.out
#SBATCH --error=logs/%x_%j.err # Stderr goes to logs/jobname_jobid.err
#SBATCH --partition=dgx-b200
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=35:00:00

for SEED in 0 1 2 3 4 5 6 7 8 9
do
    # no-mig
    python3 /vast/home/s/smathi/GIT/pg-gan-mosquito/pg_gan.py -m dadi_joint -p NI,TG,NF,TS,NI1,NI2,NF1,NF2 -n 62,162 --pt_lr 1e-3 --pt_dropout 0.25 --phase full_training -s ${SEED}
done
