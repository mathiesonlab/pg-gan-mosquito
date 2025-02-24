#!/bin/bash
#$ -pe smp 1
#$ -l h_vmem=4G
#$ -l h_rt=1:0:0
#$ -cwd
#$ -j y


ml anaconda3
conda activate /data/home/bty414/.conda/envs/pg_gan_util

python deme_draw.py
