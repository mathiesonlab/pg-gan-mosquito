#!/bin/bash
#$ -pe smp 1
#$ -l h_vmem=24G
#$ -l h_rt=1:0:0
#$ -cwd
#$ -j y
#$ -t 1-6

ml anaconda3
conda activate /data/home/bty308/pg-gan

DATA=nsg
DEMO=dadi_joint
PREFIX=/data/SBBS-FumagalliLab/mosquito_gan/pg-gan-mosquito/sim_out/CM-UG_gam_${DATA}/${DEMO}/grid/${SGE_TASK_ID}/
INPUT=${PREFIX}CM-UG_gam_nsg.out
OUTPUT=${PREFIX}_multi.png
python summary_stats_multi.py ${INPUT} ${OUTPUT} ${DEMO}
