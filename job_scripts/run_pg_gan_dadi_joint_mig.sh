#!/bin/bash

for SEED in 0 1 2 3 4 5 6 7 8 9
do
    # no-mig
    python3 /home/mathiesonlab-adm/GIT/pg-gan-mosquito/pg_gan.py -m dadi_joint_mig -p NI,TG,NF,TS,NI1,NI2,NF1,NF2,MG -n 62,162 --pt_lr 1e-3 --pt_dropout 0.25 --phase full_training -s ${SEED}
done
