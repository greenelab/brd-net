# brd-net
Transfer learning for uncovering the biology underlying rare disease

[![Build Status](https://travis-ci.com/greenelab/brd-net.svg?branch=master)](https://travis-ci.org/ben-heil/brd-net)

The goal of this project is to 
1. train a model to differentiate between the gene expression of healthy individuals and individuals with a disease
2. apply the model to rare diseases and find how gene expression differs in the disease state

## Method Overview
The current plan is to use [PLIER](https://github.com/wgmao/PLIER) as a dimensionality reduction method to help the training of a model that
predicts whether gene expression is healthy or unhealthy. 
This classifier will then be used to make predictions about diseases, hopefully classifying their gene expression as unhealthy.
Finally, some form of model interpretation will determine which of the biological networks used by PLIER most strongly indicate to the classifier that something is wrong.

Some patients don't ever receive a diagnosis explaining what their rare disease is, and a method like brd-net would help guide doctors and researchers to find the right diagnosis.

## Data Generation
As there is not a database (that we know of) that contains labels denoting whether samples in the [Sequence Read Archive](https://www.ncbi.nlm.nih.gov/sra)
correspond to healthy or unhealthy gene expression, we generated such labels by hand.
This hand labeling was done using the script `find_studies.py`, which queries [MetaSRA](http://metasra.biostat.wisc.edu/) to retrieve
gene expression from all tissue samples in the SRA that have associated metadata.
`find_studies.py` then provides an interface to write string matching rules to sort samples into 
the categories 'healthy', 'disease', and 'unknown' based on their titles. 
The expression data for the set of assigned samples from `find_studies` can be downloaded by running the notebook `download_categorized_data.ipynb`.


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
