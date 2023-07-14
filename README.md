# GPSat

## TODO (UPDATE THESE!):
- [ ] Update this README.md file, point to examples
- [ ] Separate out OI on individual grid from the full class
- [ ] Allowable output types. How to save and load hyperparameters/variational parameters (individual?). Best database?
- [ ] Examples: sea ice, ocean elevation, simulated data
- [ ] Complete unit testing (pytests).
- [ ] Specify which gpytorch version should be used.

# Environment setup

Because of the use of `cartopy` (for plotting) which is installed via conda it is recommended to use a conda environment

`conda create --name pysat_env python=3.9`

Python=3.9 is specified as that is the version the code base was originally developed with. 

# Install requirements with

from your desired conda or virtual environment, from the directory containing requirements.txt run: 

`python -m pip install requirements.txt`

### NOTE: plotting maps requires 'cartopy' package which can be install with: 

`conda install -c conda-forge cartopy=0.20.2`
 

## Inline Example

simple example of running optimal interpolation, includes binning raw data, 
predicting at multiple locations using many local experts

python script: 
`examples/inline_example.py`
notebook: 
`notebooks/inline_example.ipynb`

## Read Raw Data and Store to HDF5

NOTE: Running python scripts must be done in the top directory of this repository

see (out of date): notebooks/read_raw_data_and_store.ipynb

or run 

`python -m GPSat.read_and_store <input_config.json>`

if `<input_config.json>` not supplied an example config will be used. 
Will create `data/example/ABC.h5`


## Bin Data

`python -m GPSat.bin_data <input_config.json>`

if `<input_config.json>` not supplied an example config will be used. Requires `data/example/ABC.h5` exists 
and will create `data/example/ABC_binned.h5`

see (currently out of date): notebooks/bin_raw_data.ipynb 

## Run Local Expert OI

`python -m examples.local_expert_oi <input_config.json>`

if `<input_config.json>` not supplied an example config will be used. Requires `data/example/ABC_binned.h5` exists and
will create `results/example/ABC_binned_example.h3`

NOTE: to use a GPU with TensorFlow `LD_LIBRARY_PATH` may need to specified in Environment Variables. 
An example of such a path is `/path/to/conda/envs/<env_name>/lib/`


## Post-Process HyperParameters

Provide the results file to apply some post-processing of hyperparameters, e.g. smooth with a kernel

`python -m GPSat.postprocessing <input_config.json>`

if `<input_config.json>` not supplied an example config (`example_postprocessing.json`) will be used, which
requires `results/example/ABC_binned_example.h5` exists and the results will be written to the same file to table `_SMOOTHED`. 

Post-processing (smoothing) hyperparameters will write a config to file that can be used to generate predictions
using the newly smoothed hyperparameters via `examples.local_expert_oi`.

# Miscellaneous


### Review Observation Data

To generate plots of observations, and generate statistics run:

`python -m examples.plot_observations <input_config.json>`

if `<input_config.json>` not supplied an example config will be used. Requires `data/example/ABC.h5` 



### Generate Synthetic Data

using observation data with some ground truth, create synthetic (noisy) observations

`python -m examples.sample_from_ground_truth <input_config.json>`

if `<input_config.json>` not supplied an example config will be used. Requires `data/example/ABC.h5` and
`data/MSS/CryosatMSS-arco-2yr-140821_with_geoid_h.csv` exists.

