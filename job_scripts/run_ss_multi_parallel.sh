#!/bin/bash
#$ -pe smp 1
#$ -l h_vmem=24G
#$ -l h_rt=1:0:0
#$ -cwd
#$ -j y
#$ -t 1

ml anaconda3
conda activate pg_gan_util

DATA=nsg
DEMO=dadi_joint_mig
PREFIX=/data/SBBS-FumagalliLab/mosquito_gan/pg-gan-mosquito/sim_out/CM-UG_gam_nsg/dadi_joint_mig/sa/1e-6/0.8/
INPUT=${PREFIX}output.out
OUTPUT=${PREFIX}_multi.png
python summary_stats_multi.py ${INPUT} ${OUTPUT} ${DEMO}
