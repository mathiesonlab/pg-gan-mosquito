#!/bin/bash
#$ -pe smp 1
#$ -l h_vmem=20G
#$ -l h_rt=240:0:0
#$ -cwd
#$ -j y
#$ -t 1-6  # Total 9 tasks (3 learning rates x 3 dropout rates)

DATA=nsg
DEMO=dadi_joint_mig
POP1=CM
POP2=UG
POP1S=594
POP2S=224
SPECIES=gam
INPUT=${POP1}-${POP2}_${SPECIES}_${DATA}.h5

# Array of learning rates
declare -a LearningRates=("5e-3" "1e-3")
# Array of dropout rates
declare -a Dropouts=("0" "0.25" "0.5")

# Calculate the index for learning rates and dropouts
let "lr_index=(${SGE_TASK_ID}-1)/${#Dropouts[@]}"
let "dropout_index=(${SGE_TASK_ID}-1)%${#Dropouts[@]}"

# Get the specific learning rate and dropout rate
lr=${LearningRates[$lr_index]}
dropout=${Dropouts[$dropout_index]}

OUTPUT_PREFIX=./sim_out/${POP1}-${POP2}_${SPECIES}_${DATA}/${DEMO}/${lr}/${dropout}/${SGE_TASK_ID}/

if [ "${DEMO}" == "dadi_joint" ];
then
   PARAM="NI,TG,NF,TS,NI1,NI2,NF1,NF2"
else
   PARAM="NI,TG,NF,TS,NI1,NI2,NF1,NF2,MG"
fi

# Ensure the output directory exists
if [ ! -d "${OUTPUT_PREFIX}" ]; then
    mkdir -p "${OUTPUT_PREFIX}"
    echo "Directory created: ${OUTPUT_PREFIX}"
fi

# Load Python virtual environment
source ./tfenv_2.13/bin/activate

# Define specific output path for this configuration
OUTPUT=${OUTPUT_PREFIX}output.out
CPROF=${OUTPUT_PREFIX}cprof.txt

# Update README with current config
echo "expanded params, proposal width = 15, fc2 = 320, fc1 = 128, pt/lr= ${lr}, pt/dp = ${dropout}, NUM_BATCH = 50, adamW" > ${OUTPUT_PREFIX}README
qstat | tail -n 1 | awk '{print $1}' >> ${OUTPUT_PREFIX}README

# Run the Python script with current learning rate and dropout
echo "python3 pg_gan.py -m ${DEMO} -p ${PARAM} -n ${POP1S},${POP2S} -d ${INPUT} -l ${lr} -k ${dropout} -toy > ${OUTPUT}" > ${OUTPUT_PREFIX}README
python3 pg_gan.py -m ${DEMO} -p ${PARAM} -n ${POP1S},${POP2S} -d ${INPUT} --learning_rate ${lr} --dropout ${dropout} > ${OUTPUT}
echo "job completed with lr=${lr}, dropout=${dropout}" >> ${OUTPUT_PREFIX}README

