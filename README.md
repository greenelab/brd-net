# brd-net
Transfer learning for uncovering the biology underlying rare disease

[![Build Status](https://travis-ci.com/greenelab/brd-net.svg?branch=master)](https://travis-ci.org/ben-heil/brd-net)

The goal of this project is to 
1. train a model to differentiate between the gene expression of healthy individuals and individuals with a disease
2. apply the model to rare diseases and find how gene expression differs in the disease state

## Installation
Most of the dependencies (for both R and python) are included in the file `environment.yml`.
Upon [installing Anaconda](https://docs.anaconda.com/anaconda/install/), the dependencies can be installed and loaded with the following command:

```sh
conda env create --file environment.yml
conda activate brdnet
```

The one package that must be installed manually is PLIER, due to an issue in compatibility between Conda's tools for
building packages from Github and the Bioconductor package repository.
Fortunately, it can be installed easily by activating the brdnet conda environment, starting R, and running the commands below.

```R
library('devtools')
install_github('wgmao/PLIER')
```


Once everything is installed, if you want to create a Jupyter Notebook kernel from the environment, you can do so with

```sh
conda activate brdnet
python -m ipykernel install --user --name brdnet --display-name "brdnet"
```
## Run Order 
Because some scripts depend on the output of others, running them in order is important when starting from scratch.
The recommended running order is as follows:

1. Run `find_studies.py` to label samples from studies which contain adult gene expression that is clearly healthy or unhealthy
2. Run `download_categorized_data.ipynb` to download the expression data for the samples output by find\_studies.py
3. If you want to filter your results based on ontology terms, run `subset_studies.py`.
4. Run `model_evaluation_pipeline.sh`, which runs PLIER with different k values, then calls `evaluate_models.py` on the results


### Note
The environment file explicitly references the channel for each dependency.
While this makes it easier to follow, older versions of Anaconda may not support this format.
Development was done with conda version 4.7.5, so any version newer than that should work fine.
