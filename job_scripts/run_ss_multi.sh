#!/bin/bash
#$ -pe smp 1
#$ -l h_vmem=24G
#$ -l h_rt=1:0:0
#$ -cwd
#$ -j y

#ml anaconda3
#conda activate pg_gan_util

DEMO=$1 # i.e. dadi_joint or dadi_joint_mig

#DATA=nsg
#DEMO=dadi_joint
#POP1=GN # CM
#POP2=BF # UG
#POP1S=88 # 594
#POP2S=334 # 224
#SPECIES=gam
#LR=25e-6 #1e-6
#DROPOUT=0.8
#INPUT=./sim_out/${POP1}-${POP2}_${SPECIES}_${DATA}/${DEMO}/sa/${LR}/${DROPOUT}/output.out
#OUTPUT=./sim_out/${POP1}-${POP2}_${SPECIES}_${DATA}/${DEMO}/sa/${LR}/${DROPOUT}/${POP1}-${POP2}_${SPECIES}_${DATA}_results_multi.png

for SEED in 4 #0 1 2 3
do
    # summary stats
    echo "python3 /homes/smathieson/GIT/pg-gan-mosquito/summary_stats_multi.py output${SEED}.txt output${SEED}.pdf ${DEMO}"
    python3 /homes/smathieson/GIT/pg-gan-mosquito/summary_stats_multi.py output${SEED}.txt output${SEED}.pdf ${DEMO}

    # loss plot
    echo "python3 /homes/smathieson/GIT/pg-gan-mosquito/plot_loss.py -i output${SEED}.txt -o output${SEED}_loss.pdf"
    python3 /homes/smathieson/GIT/pg-gan-mosquito/plot_loss.py -i output${SEED}.txt -o output${SEED}_loss.pdf
done
