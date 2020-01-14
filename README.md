# Installation

Create conda environment with the requirements from environment.yml file.

Then install `tf_codage` and `load_balancer`:

`pip install -e .`
`pip install load_balancer==0.1`

TODO: create package for load_balancer.

## Running unit tests

The library `tf_codage` comes with a few unit tests. To run them use:

`pytest tests`

# Models

## Finetuning Camembert model with language model

## Classfication of discharge from surgery ("Compte rendu operatoire", CRO)

### Data

You should be able do download data with `dvc`:

`dvc checkout`

If you need to regenerate the data call:

`cd data && ./make_data.sh`

### Training models

To train models launch this command from the CLI:

`./batch_train.sh`

It will create workers on the available GPUs (to set the list see the script) and
schedule training all models on the workers

### Evaluating models


## Classification of hospital discharge ("Compte rendu hospitalization", CRH)

# Web service