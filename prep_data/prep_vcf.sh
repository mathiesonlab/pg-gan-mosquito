#!/bin/bash

# example VCF:
# /bigdata/smathieson/pg-gan/mosquito/ag3_gamb_colu_3R_phased.vcf.gz

PREFIX=/bigdata/smathieson/pg-gan/mosquito/ag3_gamb_colu
SUFFIX=phased.vcf.gz

#POP=BFA-GNB-UG_gamb
POP=GNB-BFA_gamb

# for each chromosome
for CHROM in 3L 3R
do
  echo "bcftools view -S gamb_samples.txt --min-ac 1:minor -m2 -M2 -v snps -Oz -o ${POP}_${CHROM}_${SUFFIX} ${PREFIX}_${CHROM}_${SUFFIX}"
  bcftools view -S gamb_samples.txt --min-ac 1:minor -m2 -M2 -v snps -Oz -o ${POP}_${CHROM}_${SUFFIX} ${PREFIX}_${CHROM}_${SUFFIX}
done

# then merge into one vcf
echo "bcftools concat -f filelist.txt -Oz -o ${POP}_${SUFFIX}"
bcftools concat -f filelist.txt -Oz -o ${POP}_${SUFFIX}
