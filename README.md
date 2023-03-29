# PyOptimalInterpolation

## TODO:
- Separate out OI on individual grid from the full class
- Create a class "gridOI" that does the data selection, train hyperparameters/variational parameters over gridpoints, post-process hyperparameters and predict
- Adding the other functionalities
- Allowable output types. How to save and load hyperparameters/variational parameters (individual?). Best database?
- Examples: sea ice, ocean elevation, simulated data
- Unit testing (pytests).


### NOTE: plotting maps requires 'cartopy' package which can be install with: conda install -c conda-forge cartopy=0.20.2
 
## Read Raw Data and Store to HDF5

see: notebooks/read_raw_data_and_store.ipynb

or run 

`python -m PyOptimalInterpolation.read_and_store <input_config.json>`

if `<input_config.json>` not supplied an example config will be used


## Review Raw Data (Stats and Plot)

see: notebooks/review_raw_data.ipynb

## Pre-Process (bin) Data

see: notebooks/bin_raw_data.ipynb

## Run OI

TODO: add notebook

## Post Process 

TODO: add notebook

## Load Data and Params -> make predictions

TODO: add notebook
