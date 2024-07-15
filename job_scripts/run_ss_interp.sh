#!/bin/bash
#$ -pe smp 1
#$ -l h_vmem=16G
#$ -l h_rt=1:0:0
#$ -cwd
#$ -j y

ml anaconda3
conda activate /data/home/bty308/pg-gan

python summary_stats_interp.py
