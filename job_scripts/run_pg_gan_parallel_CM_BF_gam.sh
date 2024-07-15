#!/bin/bash
#$ -pe smp 1
#$ -l h_vmem=30G
#$ -l h_rt=24:0:0
#$ -cwd
#$ -j y
#$ -t 1 # Run 5 parallel tasks with the same parameters

#!/bin/bash

#!/bin/bash

DATA=nsg
DEMO=dadi_joint
POP1=CM
POP2=BF
POP1S=594
POP2S=184
SPECIES=gam
INPUT=${POP1}-${POP2}_${SPECIES}_${DATA}.h5


# Toy mode
TOY="-t"

# Define arrays of learning rates and dropout rates
# learning_rates=("5e-2" "1e-3" "5e-3")
# dropout_rates=("0.25" "0.5")
learning_rates=("1e-3")
dropout_rates=("0.25")

# Calculate the total number of combinations
total_combinations=$((${#learning_rates[@]} * ${#dropout_rates[@]}))

# Ensure that the number of tasks matches the number of combinations
if [ $SGE_TASK_ID -gt $total_combinations ]; then
    echo "Task ID exceeds the number of parameter combinations."
    exit 1
fi

# Get the specific learning rate and dropout rate for this task
index_lr=$((($SGE_TASK_ID - 1) / ${#dropout_rates[@]}))
index_dropout=$((($SGE_TASK_ID - 1) % ${#dropout_rates[@]}))
pt_lr=${learning_rates[$index_lr]}
pt_dropout=${dropout_rates[$index_dropout]}

# Output prefix and ensure the output directory exists
OUTPUT_PREFIX=./sim_out/${POP1}-${POP2}_${SPECIES}_${DATA}/${DEMO}/${pt_lr}/${pt_dropout}/
if [ ! -d "${OUTPUT_PREFIX}" ]; then
    mkdir -p "${OUTPUT_PREFIX}"
    echo "Directory created: ${OUTPUT_PREFIX}"
fi

# Load Python virtual environment
source ./tfenv_2.13/bin/activate

# Define specific output path for this configuration
OUTPUT=${OUTPUT_PREFIX}output.out
CPROF=${OUTPUT_PREFIX}cprof.txt

# Parameters based on DEMO
if [ "${DEMO}" == "dadi_joint" ]; then
   PARAM="NI,TG,NF,TS,NI1,NI2,NF1,NF2"
else
   PARAM="NI,TG,NF,TS,NI1,NI2,NF1,NF2,MG"
fi

# Update README with current config
echo "Configuration: pt_lr=${pt_lr}, pt_dropout=${pt_dropout}, toy mode=${TOY}, run instance ${SGE_TASK_ID}" > ${OUTPUT_PREFIX}README
qstat | tail -n 1 | awk '{print $1}' >> ${OUTPUT_PREFIX}README

# Run the Python script with current learning rate, dropout, and toy mode
echo "python3 pg_gan.py -m ${DEMO} -p ${PARAM} -n ${POP1S},${POP2S} -d ${INPUT} --pt_lr ${pt_lr} --pt_dropout ${pt_dropout} --pre_trained_dir ${OUTPUT_PREFIX} --load_pm --save_pm ${TOY} > ${OUTPUT}" >> ${OUTPUT_PREFIX}README
python3 pg_gan.py -m ${DEMO} -p ${PARAM} -n ${POP1S},${POP2S} -d ${INPUT} --pt_lr ${pt_lr} --pt_dropout ${pt_dropout} --pre_trained_dir ${OUTPUT_PREFIX} --save_pm ${TOY} > ${OUTPUT}
