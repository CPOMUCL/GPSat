#!/bin/bash

# run modules and example scripts using default inputs
# script expected to be run with virtual environment activated, with all required packages installed
# 

# read in flat files, store in h5 file
echo -----------------------
echo read data from flat files and store them in a single h5
echo running: python -m GPSat.read_and_store 
python -m GPSat.read_and_store 

# plot raw observations (e.g. those read in above)
echo -----------------------
echo plot raw observations 
echo running: python -m examples.plot_observations
python -m examples.plot_observations

# bin raw observations 
echo -----------------------
echo pre-process: remove outliers and bin data to reduce the total number of observations 
echo running: python -m GPSat.bin_data
python -m GPSat.bin_data

# run OI on binned obs
echo -----------------------
echo run OI on binnned observations
echo running: python -m examples.local_expert_o
python -m examples.local_expert_oi

# post process OI results: smooth hyper parameters
echo -----------------------
echo run post processing - smoothing of hyper parameters
echo running: python -m GPSat.postprocessing
python -m GPSat.postprocessing

# re-run OI using smoothed hyper parameters (e.g. load parameters, don't optimise) 
echo -----------------------
echo RE-RUN local_expert_oi using oi_config for smoothed hyper parameters
echo results/example/ABC_binned_example_SMOOTHED.json
echo running: python -m examples.local_expert_oi results/example/ABC_binned_example_SMOOTHED.json
python -m examples.local_expert_oi results/example/ABC_binned_example_SMOOTHED.json

# generate plots for the (smoothed) hyper parameters and predictions (f*)
echo -----------------------
echo generate plots of hyper parameters and predictions using data from tables ending in _SMOOTHED in the results file
echo with no configuration provided a default one will be used 
echo running: python -m examples.plot_from_results 
python -m examples.plot_from_results 


