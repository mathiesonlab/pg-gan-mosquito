#!/bin/bash
#$ -pe smp 1
#$ -l h_vmem=4G
#$ -l h_rt=1:0:0
#$ -cwd
#$ -j y

# Load the Anaconda module and activate the conda environment
ml anaconda3
conda activate /data/home/bty308/pg-gan

# Example usage
# python script_name.py dadi_joint --NI 5591.367991008426 --TG 8989.8024202365 --NF 7148911.180861727 --TS 101.83233386492524 --NI1 25459200.34194682 --NI2 150778616.78471956 --NF1 158124479.9095534 --NF2 54259529.821181044 --out_dir output
# python script_name.py dadi_joint_mig --NI 5591.367991008426 --TG 8989.8024202365 --NF 7148911.180861727 --TS 101.83233386492524 --NI1 25459200.34194682 --NI2 150778616.78471956 --NF1 158124479.9095534 --NF2 54259529.821181044 --MG 75.15298681179857 --out_dir output

# Set variables for each parameter
MODEL="dadi_joint"
NI=5591.367991008426
TG=8989.8024202365
NF=7148911.180861727
TS=101.83233386492524
NI1=25459200.34194682
NI2=150778616.78471956
NF1=158124479.9095534
NF2=54259529.821181044
MG=75.15298681179857
OUT_DIR="OUTPUT"

# Check the value of MODEL and run the appropriate command
if [ "$MODEL" == "dadi_joint" ]; then
    python script_name.py $MODEL --NI $NI --TG $TG --NF $NF --TS $TS --NI1 $NI1 --NI2 $NI2 --NF1 $NF1 --NF2 $NF2 --out_dir $OUT_DIR
elif [ "$MODEL" == "dadi_joint_mig" ]; then
    python script_name.py $MODEL --NI $NI --TG $TG --NF $NF --TS $TS --NI1 $NI1 --NI2 $NI2 --NF1 $NF1 --NF2 $NF2 --MG $MG --out_dir $OUT_DIR
else
    echo "Invalid model specified. Use 'dadi_joint' or 'dadi_joint_mig'."
    exit 1
fi

