# pg-gan-mosquito

This software is an implementation of `pg-gan` for non-model species, specifically mosquitoes from the *Anopheles gambiae* complex. It simulates data that replicates real population genetic data. It implements a GAN-based algorithm (Generative Adversarial Network) described in the paper [Automatic inference of demographic parameters using Generative Adversarial Networks](https://onlinelibrary.wiley.com/doi/10.1111/1755-0998.13386). The mosquito results are described in the preprint [On the use of generative models for evolutionary inference of malaria vectors from genomic data](https://www.biorxiv.org/content/10.1101/2025.06.26.661760v1).

## Requirements

The package versions (as of August 2025) we are using can be found in `requirements.txt` and are listed below:

```
demes==0.2.3
demesdraw==0.4.0
ghapi==1.0.6
h5py==3.13.0
matplotlib==3.10.5
msprime==1.3.3
nox==2025.5.1
numpy==2.3.2
pandas==2.3.1
pytest==8.4.1
rich==14.1.0
scikit_allel==1.3.13
scikit_learn==1.7.1
scipy==1.16.1
seaborn==0.13.2
setuptools==59.6.0
tensorflow==2.19.0
```

## Demographic models

There are currently two demographic models implemented in `pg-gan-mosquito`, which specify potential demographic models between pairs of populations, e.g. GN (Guinea) and BF (Burkina Faso). One model has no migration post-split (`dadi-joint`), and the other includes migration (`dadi-joint-mig`). The parameter spaces for these two demographic models are based on `dadi` estimates from [Genetic diversity of the African malaria vector Anopheles gambiae](https://www.nature.com/articles/nature24995#MOESM1). (see [the pg-gan repo](https://github.com/mathiesonlab/pg-gan) for information about adding your own model). Use the `-m` flag to specify the model (required parameter).

## Schematic diagram for demographic inference and model selection

![pg_gan_mosquito schematic diagram](https://github.com/mathiesonlab/pg-gan-mosquito/blob/main/supp/pg_gan_mosquito_schem.png)

## Workflow command lines

Example commands are shown below to perform the deep learning portion of the pipeline. First a `.h5` needs to be created from a VCF file (see [the pg-gan repo](https://github.com/mathiesonlab/pg-gan)) for more information about this step.

1. Filter non-segregating sites if needed:

```
python3 prep_data/filter_nonseg.py -i ${H5_orig} -o ${H5}
```

2. Run `pg-gan-mosquito` with the no-migration demographic model:

```
python3 pg_gan.py -m dadi_joint -p NI,TG,NF,TS,NI1,NI2,NF1,NF2 -n ${SAMPLE_SIZES} -d ${H5} --pt_lr 1e-3 --pt_dropout 0.25 --phase full_training -s ${SEED} > ${OUT_FILE1}
```

3. Run `pg-gan-mosquito` with the migration demographic model:

```
python3 pg_gan.py -m dadi_joint_mig -p NI,TG,NF,TS,NI1,NI2,NF1,NF2,MG -n ${SAMPLE_SIZES} -d ${H5} --pt_lr 1e-3 --pt_dropout 0.25 --phase full_training -s ${SEED} > ${OUT_FILE2}
```

4. Demographic model selection (see `jobs_scripts/run_demo_sel.sh` for more information):

```
python3 demographic_selection_oop.py -d ${INPUT} -b ${OUTPUT_PREFIX} -s > ${OUTPUT}
```

5. Model selection loss figure:

```
python3 plotting/plot_demo_sel.py -o ${OUTPUT} -i ${LOSS_FIG}
```

## Visualizing summary statistics

`summary_stats_multi.py` was developed to compare the summary statistics distribution of the real data with the simulated data under the derived posterior as well as the baseline parameters, in a quantitative manner captured by Wasserstein distance. For example, to produce something similar to the below (for both models), run the following commands:

```
python3 summary_stats_multi.py ${OUT_FILE1} ${FIG_FILE1} dadi_joint
python3 summary_stats_multi.py ${OUT_FILE2} ${FIG_FILE2} dadi_joint_mig
```

![summary_stats_multi.py example](https://github.com/mathiesonlab/pg-gan-mosquito/blob/main/supp/ss_multi_readme.png)

See `job_scripts/run_ss_multi.sh` for more information. You can also create visualizations of the inferred demographic histories. See `job_scripts/run_deme_draw.sh` for more details.

## General notes

`pg-gan-mosquito` is under active development. Please reach out to Sara Mathieson (smathi [at] sas [dot] upenn [dot] edu) with any questions.



