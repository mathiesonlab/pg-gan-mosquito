#!/bin/bash
#$ -pe smp 1
#$ -l h_vmem=16G
#$ -l h_rt=1:0:0
#$ -cwd
#$ -j y

ml anaconda3
conda activate /data/home/bty308/pg-gan

for i in 8
do
	DATA=nsg
	DEMO=dadi_joint
	PREFIX=/data/SBBS-FumagalliLab/mosquito_gan/pg-gan-mosquito/sim_out/GNB-BFA_gamb_${DATA}/${DEMO}/${i}/GNB-BFA_gamb_${DATA}
	INPUT=${PREFIX}.out
	OUTPUT=${PREFIX}_multi.png
	python summary_stats_multi.py ${INPUT} ${OUTPUT} ${DEMO}
done
