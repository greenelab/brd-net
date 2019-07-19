# brd-net
Transfer learning for uncovering the biology underlying rare disease

[![Build Status](https://travis-ci.org/ben-heil/brd-net.svg?branch=master)](https://travis-ci.org/ben-heil/brd-net)

The goal of this project is to 
1. train a model to differentiate between the gene expression of healthy individuals and individuals with a disease
2. apply the model to rare diseases and find how gene expression differs in the disease state

## Installation
---
All the dependencies (for both R and python) are included in the file `environment.yml`.
Upon [installing Anaconda](https://docs.anaconda.com/anaconda/install/), the dependencies can be installed and loaded with the following command:

```sh
conda env create --file environment.yml
conda activate brdnet
```

Once everything is installed, if you want to create a Jupyter Notebook kernel from the environment, you can do so with

```sh
conda activate brdnet
python -m ipykernel install --user --name brdnet --display-name "brdnet"
```

### Note
The environment file explicitly references the channel for each dependency.
While this makes it easier to follow, older versions of Anaconda may not support this format.
Development was done with conda version 4.7.5, so any version newer than that should work fine.
