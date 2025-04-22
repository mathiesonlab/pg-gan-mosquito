#!/bin/bash
#$ -pe smp 1
#$ -l h_vmem=30G
#$ -l h_rt=240:0:0
#$ -cwd
#$ -j y

INPUT=/bigdata/smathieson/pg-gan/mosquito/GN-BF_gam_biallelic_2017_filter.h5
OUTPUT_PREFIX=./sim_out/GN-BF_gam_biallelic_2017/model_compare # dadi_joint/reduce_mean_filter_param3
OUTPUT=${OUTPUT_PREFIX}/model_comparison.out
#CPROF=${OUTPUT_PREFIX}/cprof.txt

#source ./tfenv_2.13/bin/activate
echo "dadi_joint vs dadi_joint_mig" # posterior, 2 pool layer, fc1 2 = 160 128, testing"
#qstat | tail -n 1 | awk '{print $1}' >> ${OUTPUT_PREFIX}/README
echo "python3 demographic_selection_oop.py -d ${INPUT} -b ${OUTPUT_PREFIX} -s > ${OUTPUT}"
python3 demographic_selection_oop.py -d ${INPUT} -b ${OUTPUT_PREFIX} -s > ${OUTPUT}
#echo "job completed" >> ${OUTPUT_PREFIX}/README

