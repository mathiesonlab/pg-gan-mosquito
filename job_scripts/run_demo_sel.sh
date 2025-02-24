#!/bin/bash
#$ -pe smp 1
#$ -l h_vmem=30G
#$ -l h_rt=240:0:0
#$ -cwd
#$ -j y

INPUT=CM-UG_gam_nsg.h5
OUTPUT_PREFIX=./sim_out/model_compare/CM-UG_gam
OUTPUT=${OUTPUT_PREFIX}/model_comparison.out
CPROF=${OUTPUT_PREFIX}/cprof.txt

source ./tfenv_2.13/bin/activate
echo "dadi_joint_mig vs dadi_joint_mig posterior, 2 pool layer, fc1 2 = 160 128, testing" > ${OUTPUT_PREFIX}/README
qstat | tail -n 1 | awk '{print $1}' >> ${OUTPUT_PREFIX}/README
python3 -m cProfile -o ${CPROF} demographic_selection_oop.py -d ${INPUT} -b ${OUTPUT_PREFIX} -s -t > ${OUTPUT}
echo "job completed" >> ${OUTPUT_PREFIX}/README

