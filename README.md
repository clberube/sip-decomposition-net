# sip-decomposition-net
Repository for the complex-valued variational autoencoder for Debye/Warburg decomposition of spectral induced polarization data

## Reproducibility
The repository currently contains all code necessary to load data, train the model and predict relaxation time distributions across the dataset and is ready for manuscript peer-review. 

Additional work to integrate all plotting functions necessary to reproduce all 14 figures of the manuscript is ongoing.

## Usage 
Run the train_model.py script to reproduce results from the paper.

## Repository structure
- train_model.py is the main script 
- plotlib.py contains the plotting functions
- utilities.py contains utility functions, including training and predicting functions
- models.py contains the complex-valued conditional variational autoencoder
