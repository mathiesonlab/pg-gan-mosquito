#!/bin/bash
#$ -pe smp 1
#$ -l h_vmem=36G
#$ -l h_rt=240:0:0
#$ -cwd
#$ -j y

DATA=nsg 
DEMO=dadi_joint
POP1=CM
POP2=UG
POP1S=594
POP2S=224
SPECIES=gam
INPUT=${POP1}-${POP2}_${SPECIES}_${DATA}.h5


OUTPUT_PREFIX=./sim_out/${POP1}-${POP2}_${SPECIES}_${DATA}/${DEMO}/3/
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
echo "expanded params, proposal width = 15, fc2 = 320, fc1 = 128, pt/lr= 1e-3 25e-6, pt/dp = 0.25 0.8, NUM_BATCH = 50, adamW" > ${OUTPUT_PREFIX}README
qstat | tail -n 1 | awk '{print $1}' >> ${OUTPUT_PREFIX}README
python3 -m cProfile -o ${CPROF} pg_gan.py -m ${DEMO} -p ${PARAM} -n ${POP1S},${POP2S} -d ${INPUT} > ${OUTPUT}
echo "job completed" >> ${OUTPUT_PREFIX}README
