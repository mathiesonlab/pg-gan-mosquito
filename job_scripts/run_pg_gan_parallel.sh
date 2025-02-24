#!/bin/bash
#$ -pe smp 1
#$ -l h_vmem=30G
#$ -l h_rt=240:0:0
#$ -cwd
#$ -j y
#$ -t 1-5

DATA=nsg
DEMO=dadi_joint
POP1=ANG
POP2=BFA
SPECIES=col
INPUT=${POP1}-${POP2}_${SPECIES}_${DATA}.h5
OUTPUT_PREFIX=./sim_out/${POP1}-${POP2}_${SPECIES}_${DATA}/${DEMO}/${SGE_TASK_ID}/
OUTPUT=${OUTPUT_PREFIX}${POP1}-${POP2}_${SPECIES}_${DATA}.out
CPROF=${OUTPUT_PREFIX}/cprof.txt

if [ "${DEMO}" == "dadi_joint" ];
then
   PARAM="NI,TG,NF,TS,NI1,NI2,NF1,NF2"
else
   PARAM="NI,TG,NF,TS,NI1,NI2,NF1,NF2,MG"
fi

if [ ! -d "${OUTPUT_PREFIX}" ]; then
    # The directory does not exist, create it
    mkdir -p "${OUTPUT_PREFIX}"
    echo "Directory created: ${OUTPUT_PREFIX}"
fi


source ./tfenv_2.13/bin/activate
echo "proposal width = 15, fc2 = 160,pt/lr= 1e-3 25e-6, pt/dp = 0.5 0.8, NUM_BATCH = 50, adamW" > ${OUTPUT_PREFIX}README
qstat | tail -n 1 | awk '{print $1}' >> ${OUTPUT_PREFIX}README
python3 -m cProfile -o ${CPROF} pg_gan.py -m ${DEMO} -p ${PARAM} -n 150,156 -d ${INPUT} > ${OUTPUT}
echo "job completed" >> ${OUTPUT_PREFIX}README

