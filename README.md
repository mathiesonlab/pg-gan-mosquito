# pg-gan-mosquito

This software is an implementation of pg_gan on non-model species, to simulate data that replicate real population anopheles genetic data. It implements a GAN-based algorithm (Generative Adversarial Network) described in the paper [Automatic inference of demographic parameters using Generative Adversarial Networks](https://onlinelibrary.wiley.com/doi/10.1111/1755-0998.13386). 

## Environments
The created environments can be found in the environments folder. ```tf_env_requirements.txt``` is used for deep learning related processes such as ```pg_gan.py``` and ```demographic_selection_oop.py```. ```pg-gan_env.yml``` is used for creating summary statistics plots, ABC rejection analysis and other utility functions.

## Demographic models
There are currently two demographic models implemented in pg_gan_mosquito, which models the potential demographic model between the GNB-BFA_gamb populations. The parameter spaces for respective demographic parameters are based off dadi models from [Genetic diversity of the African malaria vector Anopheles gambiae](https://www.nature.com/articles/nature24995#MOESM1). (see https://github.com/mathiesonlab/pg-gan for information about adding your own model). Use the -m flag to specify the model (required parameter).

## Schematic diagram for demographic inference and model selection
See below for the workflow of implementing demographic inference and model selection using pg_gan_mosquito. Example commands are shown below to perform the deep learning portion of the pipeline

```
#activate environment
source ~/{PATH}/tfenv_2.13/bin/activate
#demographic inference on real data {H5} with demographic model {DEMO}, inferring demographic parameters {PARAM}, with sample population size {SAMPLE_SIZE}. The posterior outfile is written in {OUTPUT}.
python3 pg_gan.py -m ${DEMO} -p ${PARAM} -n {SAMPLE_SIZE} -d ${H5} > ${OUTPUT}
#demographic model selection on real data {H5}, between derived posteriors stored in input_posteriors.txt, which contains path to the respective posterior outfiles. The model selection outfile is written in {OUTPUT}.
python3 demographic_selection_oop.py -d ${H5} -b ${INPUT_DIR} -s -t > ${OUTPUT}
```


![pg_gan_mosquito schematic diagram](https://github.com/mathiesonlab/pg-gan-mosquito/blob/main/supp/pg_gan_mosquito_schem.png)
