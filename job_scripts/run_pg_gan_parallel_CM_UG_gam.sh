#!/bin/bash
#$ -pe smp 1
#$ -l h_vmem=30G
#$ -l h_rt=240:0:0
#$ -cwd
#$ -j y
#$ -t 1-9  # Run 5 parallel tasks with the same parameters

DATA=nsg
DEMO=dadi_joint_mig
POP1=CM
POP2=UG
POP1S=594
POP2S=224
SPECIES=gam
INPUT=${POP1}-${POP2}_${SPECIES}_${DATA}.h5

# Fixed learning rate and dropout rate
lr="1e-3"
dropout="0.25"

OUTPUT_PREFIX=./sim_out/${POP1}-${POP2}_${SPECIES}_${DATA}/${DEMO}/grid/${SGE_TASK_ID}/

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

# Parameters based on DEMO
if [ "${DEMO}" == "dadi_joint" ];
then
   PARAM="NI,TG,NF,TS,NI1,NI2,NF1,NF2"
else
   PARAM="NI,TG,NF,TS,NI1,NI2,NF1,NF2,MG"
fi

# Update README with current config
echo "Configuration: lr=${lr}, dropout=${dropout, augmented NF1 pretrain parameter}, run instance ${SGE_TASK_ID}" > ${OUTPUT_PREFIX}README
qstat | tail -n 1 | awk '{print $1}' >> ${OUTPUT_PREFIX}README

# Run the Python script with current learning rate and dropout
echo "python3 pg_gan.py -m ${DEMO} -p ${PARAM} -n ${POP1S},${POP2S} -d ${INPUT} --learning_rate ${lr} --dropout ${dropout} -toy > ${OUTPUT}" >> ${OUTPUT_PREFIX}README
python3 pg_gan.py -m ${DEMO} -p ${PARAM} -n ${POP1S},${POP2S} -d ${INPUT} --learning_rate ${lr} --dropout ${dropout} > ${OUTPUT}
echo "job completed with lr=${lr}, dropout=${dropout}, run instance ${SGE_TASK_ID}" >> ${OUTPUT_PREFIX}README
