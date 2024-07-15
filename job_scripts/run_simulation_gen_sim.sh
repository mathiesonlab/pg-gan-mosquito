#!/bin/bash
#$ -pe smp 1
#$ -l h_vmem=1G
#$ -l h_rt=1:0:0
#$ -cwd
#$ -j y

INPUT=GNB-BFA_gamb_nsg.h5
OUTPUT=./sim_out/GNB-BFA_gamb_nsg.out

ml anaconda3
conda activate /data/home/bty308/pg-gan

python simulation_gen_sim_migration.py
