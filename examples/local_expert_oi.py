# run LocalExpertOI using provided configuration

# if a config (json) file not provide as input argument a default/example config fill be used

# HOW TO: generate example input data
# - data/example/ABC.h - run: python -m PyOptimalInterpolation.read_and_store
# - data/example/ABC_binned.h5 - run: python -m examples.bin_raw_data_from_hdf5_by_batch

import os
import warnings
import time
import json

import numpy as np
import pandas as pd

import tensorflow as tf

from PyOptimalInterpolation import get_parent_path, get_data_path
from PyOptimalInterpolation.local_experts import LocalExpertOI
from PyOptimalInterpolation.utils import get_config_from_sysargv, nested_dict_literal_eval, grid_2d_flatten

# change tensorflow warning levels(?)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

# --
# helper functions
# --

print("GPUs:", tf.config.list_physical_devices('GPU'))

pd.set_option("display.max_columns", 200)

# ---
# config
# ---

config = get_config_from_sysargv()

# assert config is not None, f"config is None, issue reading it in? check argument provided to script"
# if config not read in - get default / example
if config is None:

    config_file = get_parent_path("configs", "example_local_expert_oi.json")
    warnings.warn(f"\n\nconfig is empty / not provided, will just use an example config:\n{config_file}\n\n")
    with open(config_file, "r") as f:
        config = nested_dict_literal_eval(json.load(f))

    # specify paths and input values
    config["results"]["dir"] = get_parent_path("results", "example")
    config["results"]["file"] = "ABC_binned_example.h5"
    config["locations"]["file"] = get_data_path("locations", "example_expert_locations_arctic.csv")
    config["data"]["data_source"] = get_data_path("example", "ABC_binned.h5")
    config["pred_loc"]["df_file"] = get_data_path("locations", "2d_xy_grid_5x5km.csv")


# ------
# (extract) parameters
# ------

# pop out and print "comment"
comment = config.pop("comment", None)
comment = "\n".join(comment) if isinstance(comment, list) else comment
print(f"\nconfig 'comment':\n\n{comment}\n\n")

results = config["results"]

# run_kwargs - previously named misc
run_kwargs = config.get("run_kwargs", config.get("misc", {}))

# legacy handling of skip_valid_checks_on being in config
if "skip_valid_checks_on" not in run_kwargs:
    skip_valid_checks_on = ["skip_valid_checks_on"] + config.get("skip_valid_checks_on", [])
    run_kwargs["skip_valid_checks_on"] = skip_valid_checks_on

# --------
# initialise LocalExpertOI object
# --------

locexp = LocalExpertOI(expert_loc_config=config['locations'],
                       data_config=config["data"],
                       model_config=config["model"],
                       pred_loc_config=config.get("pred_loc", None))

# ----------------
# Increment over the expert locations
# ----------------

store_path = os.path.join(results["dir"], results["file"])

locexp.run(store_path=store_path,
           **run_kwargs)

