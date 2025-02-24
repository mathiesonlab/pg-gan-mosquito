#!/bin/bash
#$ -pe smp 1
#$ -l h_vmem=30G
#$ -l h_rt=240:0:0
#$ -cwd
#$ -j y
#$ -t 1-4

#!/bin/bash

#!/bin/bash

DATA=nsg
DEMO=dadi_joint_mig
POP1=GN
POP2=BF
POP1S=88
POP2S=334
SPECIES=gam
PATH=/bigdata/smathieson/pg-gan/mosquito
INPUT=${PATH}/${POP1}-${POP2}_${SPECIES}_${DATA}.h5

# Toy mode
TOY=""

# Define phase (pt, sa, or full_training)
PHASE="pt"

# Check if phase is provided
if [ -z "$PHASE" ]; then
  echo "Phase parameter (pt, sa, or full_training) is required."
  exit 1
fi

# Define arrays of learning rates and dropout rates for pretraining and simulated annealing
pt_learning_rates=("1e-3" "1e-4")
pt_dropout_rates=("0.5" "0.25")
sa_learning_rates=("1e-7" "1e-6")
sa_dropout_rates=("0.8" "0.6")

# Calculate the total number of combinations for pt and sa phases
if [ "$PHASE" == "pt" ]; then
  total_combinations=$((${#pt_learning_rates[@]} * ${#pt_dropout_rates[@]}))
elif [ "$PHASE" == "sa" ]; then
  total_combinations=$((${#sa_learning_rates[@]} * ${#sa_dropout_rates[@]}))
else
  total_combinations=1  # full_training runs with only one set of parameters
fi

# Ensure that the number of tasks matches the number of combinations (for pt and sa)
if [ "$PHASE" != "full_training" ] && [ $SGE_TASK_ID -gt $total_combinations ]; then
    echo "Task ID exceeds the number of parameter combinations."
    exit 1
fi

# Get the specific learning rate and dropout rate for this task based on the phase
if [ "$PHASE" == "pt" ]; then
  index_lr=$((($SGE_TASK_ID - 1) / ${#pt_dropout_rates[@]}))
  index_dropout=$((($SGE_TASK_ID - 1) % ${#pt_dropout_rates[@]}))
  lr=${pt_learning_rates[$index_lr]}
  dropout=${pt_dropout_rates[$index_dropout]}
  OUTPUT_PREFIX=./sim_out/${POP1}-${POP2}_${SPECIES}_${DATA}/${DEMO}/${PHASE}/${lr}/${dropout}/


elif [ "$PHASE" == "sa" ]; then
  index_lr=$((($SGE_TASK_ID - 1) / ${#sa_dropout_rates[@]}))
  index_dropout=$((($SGE_TASK_ID - 1) % ${#sa_dropout_rates[@]}))
  lr=${sa_learning_rates[$index_lr]}
  dropout=${sa_dropout_rates[$index_dropout]}
  OUTPUT_PREFIX=./sim_out/${POP1}-${POP2}_${SPECIES}_${DATA}/${DEMO}/${PHASE}/${lr}/${dropout}/

elif [ "$PHASE" == "full_training" ]; then
  OUTPUT_PREFIX=./sim_out/${POP1}-${POP2}_${SPECIES}_${DATA}/${DEMO}/${PHASE}/${SGE_TASK_ID}/
fi


# Output prefix and ensure the output directory exists
if [ ! -d "${OUTPUT_PREFIX}" ]; then
    mkdir -p "${OUTPUT_PREFIX}"
    echo "Directory created: ${OUTPUT_PREFIX}"
else
    echo "Directory already exists: ${OUTPUT_PREFIX}"
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
echo "Configuration: phase=${PHASE} run instance ${SGE_TASK_ID}" > ${OUTPUT_PREFIX}README
qstat | tail -n 1 | awk '{print $1}' >> ${OUTPUT_PREFIX}README

# Handle different phases
if [ "$PHASE" == "pt" ]; then
    echo "python3 pg_gan.py -m ${DEMO} -p ${PARAM} -n ${POP1S},${POP2S} -d ${INPUT} --pt_lr ${lr} --pt_dropout ${dropout} --pre_trained_dir ${OUTPUT_PREFIX} --save_pm --phase ${PHASE} ${TOY} > ${OUTPUT}" >> ${OUTPUT_PREFIX}README
    python3 pg_gan.py -m ${DEMO} -p ${PARAM} -n ${POP1S},${POP2S} -d ${INPUT} --pt_lr ${lr} --pt_dropout ${dropout} --pre_trained_dir ${OUTPUT_PREFIX} --save_pm --phase ${PHASE} ${TOY} > ${OUTPUT}

elif [ "$PHASE" == "sa" ]; then
    # Set pre-trained directory for simulated annealing
    pretrained_lr="5e-4"
    pretrained_dropout="0.5"
    pretrained_dir="./sim_out/${POP1}-${POP2}_${SPECIES}_${DATA}/${DEMO}/pt/${pretrained_lr}/${pretrained_dropout}/"
    pretrained_discriminator="${pretrained_dir}discriminator.data-00000-of-00001"
    if [ ! -f "${pretrained_discriminator}" ]; then
      echo "Pre-trained discriminator not found: ${pretrained_discriminator}"
      exit 1
    fi

    echo "python3 pg_gan.py -m ${DEMO} -p ${PARAM} -n ${POP1S},${POP2S} -d ${INPUT} --sa_lr ${lr} --sa_dropout ${dropout} --pre_trained_dir ${pretrained_dir} --phase ${PHASE} ${TOY} > ${OUTPUT}" >> ${OUTPUT_PREFIX}README
    python3 pg_gan.py -m ${DEMO} -p ${PARAM} -n ${POP1S},${POP2S} -d ${INPUT} --sa_lr ${lr} --sa_dropout ${dropout} --pre_trained_dir ${pretrained_dir} --phase ${PHASE} ${TOY} > ${OUTPUT}

elif [ "$PHASE" == "full_training" ]; then
    pt_dropout="0.5"
    pt_lr="5e-4"
    sa_dropout="0.8"
    sa_lr="25e-6"
    echo "python3 pg_gan.py -m ${DEMO} -p ${PARAM} -n ${POP1S},${POP2S} -d ${INPUT} --pt_lr ${pt_lr} --pt_dropout ${pt_dropout} --sa_lr ${sa_lr} --sa_dropout ${sa_dropout} --phase ${PHASE} ${TOY}" >> ${OUTPUT_PREFIX}README
    python3 pg_gan.py -m ${DEMO} -p ${PARAM} -n ${POP1S},${POP2S} -d ${INPUT} --pt_lr ${pt_lr} --pt_dropout ${pt_dropout} --sa_lr ${sa_lr} --sa_dropout ${sa_dropout} --phase ${PHASE} ${TOY} > ${OUTPUT}
fi
