#!/bin/bash
#$ -cwd
#$ -j y
#$ -pe smp 2
#$ -l h_rt=1:0:0
#$ -l h_vmem=2G

INPUT_PREFIX=./sim_out/model_compare/GNB-BFA_gamb/8
INPUT_LABEL=${INPUT_PREFIX}/testSet_labels.txt
INPUT_PRED=${INPUT_PREFIX}/testSet_Predictions.txt
INPUT_EMP_PRED=${INPUT_PREFIX}/Emp_Predictions.txt
OUTPUT=${INPUT_PREFIX}/ABC_output.out

ml anaconda3
conda activate /data/home/bty308/pg-gan

Rscript ABC_reject_analysis.R ${INPUT_LABEL} ${INPUT_PRED} ${INPUT_EMP_PRED} ${OUTPUT} > ${OUTPUT}
