#!/bin/bash
#$ -pe smp 4
#$ -l h_vmem=4G
#$ -l h_rt=1:0:0
#$ -cwd
#$ -j y

INPUT=./sim_out/GNB-BFA_gamb_sim.out

ml anaconda3
conda activate /data/home/bty308/pg-gan

#python summary_stats_test.py ${INPUT}
#echo "analysing together"
#python dm_gm_analysis.py
#echo "analysing seperately"
python genotype_eval.py
