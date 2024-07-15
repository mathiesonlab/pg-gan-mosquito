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
	OUTPUT=${PREFIX}.png
	OUTPUT2=${PREFIX}_run.png
	OUTPUT3=${PREFIX}_param.png
	OUTPUT4=${PREFIX}_proposal.png

	python summary_stats.py ${INPUT} ${OUTPUT} ${OUTPUT2} ${OUTPUT3} ${OUTPUT4}
done
