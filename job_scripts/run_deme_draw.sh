#!/bin/bash
#$ -pe smp 1
#$ -l h_vmem=4G
#$ -l h_rt=1:0:0
#$ -cwd
#$ -j y

# reduce_mean_filter_param3, dadi_joint, output2
MODEL=no_mig
NI=579516
TG=75249
NF=2784452
TS=3519
NI1=117287221
NI2=62771288
NF1=59092640
NF2=189066655
OUTDIR=/Users/saramathieson/Dropbox/pg-gan/jacky/redo_results/demes/$MODEL

echo "python3 deme_draw.py --model $MODEL --NI $NI --TG $TG --NF $NF --TS $TS --NI1 $NI1 --NI2 $NI2 --NF1 $NF1 --NF2 $NF2 --out_dir $OUTDIR"
python3 deme_draw.py --model $MODEL --NI $NI --TG $TG --NF $NF --TS $TS --NI1 $NI1 --NI2 $NI2 --NF1 $NF1 --NF2 $NF2 --out_dir $OUTDIR

# reduce_mean_filter_param3, dadi_joint_mig, output1
MODEL=mig
NI=560796
TG=57254
NF=7772061
TS=2265
NI1=16137900
NI2=20976274
NF1=134237900
NF2=218082996
MG=27
OUTDIR=/Users/saramathieson/Dropbox/pg-gan/jacky/redo_results/demes/$MODEL

echo "python3 deme_draw.py --model $MODEL --NI $NI --TG $TG --NF $NF --TS $TS --NI1 $NI1 --NI2 $NI2 --NF1 $NF1 --NF2 $NF2 --MG $MG --out_dir $OUTDIR"
python3 deme_draw.py --model $MODEL --NI $NI --TG $TG --NF $NF --TS $TS --NI1 $NI1 --NI2 $NI2 --NF1 $NF1 --NF2 $NF2 --MG $MG --out_dir $OUTDIR


