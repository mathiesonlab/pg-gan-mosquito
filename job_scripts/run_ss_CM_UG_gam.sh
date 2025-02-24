#!/bin/bash
#$ -pe smp 1
#$ -l h_vmem=30G
#$ -l h_rt=1:0:0
#$ -cwd
#$ -j y

DATA=nsg
DEMO=dadi_joint_mig
POP1=CM
POP2=UG
POP1S=594
POP2S=224
SPECIES=gam
LR=1e-6
DROPOUT=0.8
INPUT=./sim_out/${POP1}-${POP2}_${SPECIES}_${DATA}/${DEMO}/sa/${LR}/${DROPOUT}/output.out
OUTPUT_PREFIX=./sim_out/${POP1}-${POP2}_${SPECIES}_${DATA}/${DEMO}/sa/${LR}/${DROPOUT}/

# Ensure the output directory exists
mkdir -p ${OUTPUT_PREFIX}

# File paths for output
OUTPUT=${OUTPUT_PREFIX}${POP1}-${POP2}_${SPECIES}_${DATA}_results.png
OUTPUT2=${OUTPUT_PREFIX}${POP1}-${POP2}_${SPECIES}_${DATA}_run.png
OUTPUT3=${OUTPUT_PREFIX}${POP1}-${POP2}_${SPECIES}_${DATA}_param.png
OUTPUT4=${OUTPUT_PREFIX}${POP1}-${POP2}_${SPECIES}_${DATA}_proposal.png

# Load Anaconda and activate environment
ml anaconda3
conda activate pg_gan_util

# Execute the Python script
python summary_stats.py ${INPUT} ${OUTPUT} ${OUTPUT2} ${OUTPUT3} ${OUTPUT4}
