# GPSat

This repository contains code for the paper
"[Scalable interpolation of satellite altimetry data with probabilistic machine learning](https://www.nature.com/articles/s41467-024-51900-x)"
by William Gregory, Ronald MacEachern, So Takao, Isobel Lawrence, Carmen Nab, Marc Peter Deisenroth and Tsamados, Michel.

### Quick Start

Run an example notebook in colab:

<a target="_blank" href="https://colab.research.google.com/github/CPOMUCL/GPSat/blob/gh-pages/notebooks/inline_example.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>

### Documentation

can be found at: [GPSat Documentation](https://cpomucl.github.io/GPSat/ "Visit GPSat Documentation")

# Environment setup

From the top level directory, e.g. the one containing this file, create a virtual environment named `venv`

`python -m venv venv`

activate virtual environment with 

`source venv/bin/activate`

It's recommend to use a recent version of python, e.g. >= 3.10

# Install requirements with

`python -m pip install -r requirements.txt`


## Inline Example

Simple example of running optimal interpolation, includes binning raw data, 
predicting at multiple locations using many local experts

python script: 
`examples/inline_example.py`
notebook: 
`notebooks/inline_example.ipynb`

## Read Raw Data and Store to HDF5

NOTE: Running python scripts must be done in the top directory of this repository

See (out of date): notebooks/read_raw_data_and_store.ipynb

or run 

`python -m GPSat.read_and_store <input_config.json>`

If `<input_config.json>` not supplied an example config (`configs/example_read_and_store_raw_data.json.json`) 
will be used, paths will be changed to the package location.
Will create `data/example/ABC.h5`


## Bin Data

`python -m GPSat.bin_data <input_config.json>`

If `<input_config.json>` not supplied an example config (`configs/example_bin_raw_data.json`) will be used, paths
will be changed to the package location.
Requires `data/example/ABC.h5` exists and will create `data/example/ABC_binned.h5`

see (currently out of date): notebooks/bin_raw_data.ipynb 

## (Optional) Plot Observations

It can be useful to visualise before processing it further. This can be done
with  

`python -m examples.plot_observations <input_config.json>`

If `<input_config.json>` not supplied an example config (`configs/example_plot_observations.json`) will be used, paths
will be changed to the package location. Requires `data/example/ABC.h5` exists.


## Run Local Expert OI

`python -m examples.local_expert_oi <input_config.json>`

If `<input_config.json>` not supplied an example config will be used  (`configs/example_local_expert_oi.json.json`). 
Requires `data/example/ABC_binned.h5` exists and will create `results/example/ABC_binned_example.h3`

NOTE: to use a GPU with TensorFlow `LD_LIBRARY_PATH` may need to specified in Environment Variables. 
An example of such a path is `/path/to/conda/envs/<env_name>/lib/`


A work in progress plotting script is available (`python -m examples.local_expert_plot_obs <input_config.json>`) 
that will plot the expert locations and observations used in OI. The input_config is the same used for local_expert_oi.

## Post-Process HyperParameters

Provide the results file to apply some post-processing of hyperparameters, e.g. smooth with a kernel

`python -m GPSat.postprocessing <input_config.json>`

if `<input_config.json>` not supplied an example config (`example_postprocessing.json`) will be used, which
requires `results/example/ABC_binned_example.h5` exists and the results will be written to the same file to table `_SMOOTHED`. 

Post-processing (smoothing) hyperparameters will write a config to file that can be used to generate predictions
using the newly smoothed hyperparameters via `examples.local_expert_oi`.

## Generate Predictions using Post-Processed Hyper Parameters

Run `local_expert_oi` again this time using the configuration file generate from the post-processing step, e.g.:

`python -m examples.local_expert_oi results/example/ABC_binned_example_SMOOTHED.json`

The post-processing step can produce a configuration file of `local_expert_oi` that will
load the smoothed hyper parameters, skip optimisation and make predictions

## Plot Results

Plot heat map of values from results tables by specifying plot template(s) in a configuration file and
utilising plot functions from `plot_utils.py`

`python -m examples.plot_from_results <input_config.json>`

If `<input_config.json>` not supplied an example config (`example_plot_from_results.json`).
In order the for example script to work the predictions made using the smoothed hyper parameters
must be present.

# Miscellaneous


### Review Observation Data

To generate plots of observations, and generate statistics run:

`python -m examples.plot_observations <input_config.json>`

if `<input_config.json>` not supplied an example config will be used. Requires `data/example/ABC.h5` 


### Generate Synthetic Data

Using observation data with some ground truth, create synthetic (noisy) observations

`python -m examples.sample_from_ground_truth <input_config.json>`

if `<input_config.json>` not supplied an example config will be used. Requires `data/example/ABC.h5` and
`data/MSS/CryosatMSS-arco-2yr-140821_with_geoid_h.csv` exists.


# Citation
If you found this useful, please consider citing
```
@article{gregory2024scalable,
  title={Scalable interpolation of satellite altimetry data with probabilistic machine learning},
  author={Gregory, William and MacEachern, Ronald and Takao, So and Lawrence, Isobel and Nab, Carmen and Deisenroth, Marc Peter and Tsamados, Michel},
  journal={Nature Communications},
  year={2024}
}
```


