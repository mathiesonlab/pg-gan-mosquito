#!/bin/bash
#$ -pe smp 1
#$ -l h_vmem=4G
#$ -l h_rt=1:0:0
#$ -cwd
#$ -j y


PREFIX=./sim_out/model_compare/GNB-BFA_gamb/6
INPUT=${PREFIX}/model_comparison.out
OUTPUT=${PREFIX}/model_comparison_eval.png

ml anaconda3
conda activate /data/home/bty308/pg-gan

python demo_sel_ss.py ${INPUT} ${OUTPUT}
