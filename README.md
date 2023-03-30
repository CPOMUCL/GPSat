# PyOptimalInterpolation

## TODO:
- Separate out OI on individual grid from the full class
- Create a class "gridOI" that does the data selection, train hyperparameters/variational parameters over gridpoints, post-process hyperparameters and predict
- Adding the other functionalities
- Allowable output types. How to save and load hyperparameters/variational parameters (individual?). Best database?
- Examples: sea ice, ocean elevation, simulated data
- Unit testing (pytests).


### NOTE: plotting maps requires 'cartopy' package which can be install with: conda install -c conda-forge cartopy=0.20.2
 
Running python scripts must be done in the top directory of this repository

## Read Raw Data and Store to HDF5

see: notebooks/read_raw_data_and_store.ipynb

or run 

`python -m PyOptimalInterpolation.read_and_store <input_config.json>`

if `<input_config.json>` not supplied an example config will be used. 
Will create `data/example/ABC.h5`


## Bin Data

`python -m examples.bin_raw_data_from_hdf5_by_batch <input_config.json>`

if `<input_config.json>` not supplied an example config will be used. Requires `data/example/ABC.h5` exists 
and will create `data/example/ABC_binned.h5`

see (currently out of date): notebooks/bin_raw_data.ipynb 

## Run Local Expert OI

`python -m examples.local_expert_oi <input_config.json>`

if `<input_config.json>` not supplied an example config will be used. Requires `data/example/ABC_binned.h5` exists.

NOTE: to use a GPU with TensorFlow `LD_LIBRARY_PATH` may need to specified in Environment Variables. 
An example of such a path is `/path/to/conda/envs/<env_name>/lib/`

## Review Raw Data (Stats and Plot)

see: notebooks/review_raw_data.ipynb

## Run OI

TODO: add notebook

## Post Process 

TODO: add notebook

## Load Data and Params -> make predictions

TODO: add notebook
