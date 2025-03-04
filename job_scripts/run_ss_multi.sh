#!/bin/bash
#$ -pe smp 1
#$ -l h_vmem=24G
#$ -l h_rt=1:0:0
#$ -cwd
#$ -j y

ml anaconda3
conda activate pg_gan_util


DATA=nsg
DEMO=dadi_joint
POP1=GN # CM
POP2=BF # UG
POP1S=88 # 594
POP2S=334 # 224
SPECIES=gam
LR=25e-6 #1e-6
DROPOUT=0.8
INPUT=./sim_out/${POP1}-${POP2}_${SPECIES}_${DATA}/${DEMO}/sa/${LR}/${DROPOUT}/output.out
OUTPUT=./sim_out/${POP1}-${POP2}_${SPECIES}_${DATA}/${DEMO}/sa/${LR}/${DROPOUT}/${POP1}-${POP2}_${SPECIES}_${DATA}_results_multi.png

echo "python summary_stats_multi.py ${INPUT} ${OUTPUT} ${DEMO}"
#python summary_stats_multi.py ${INPUT} ${OUTPUT} ${DEMO}
