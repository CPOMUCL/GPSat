# run LocalExpertOI using provided configuration

# if a config (json) file not provide as input argument a default/example config fill be used

# HOW TO: generate example input data
# - data/example/ABC.h - run: python -m GPSat.read_and_store
# - data/example/ABC_binned.h5 - run: python -m examples.bin_raw_data_from_hdf5_by_batch

import os
import re
import warnings
import logging
import time
import json

import numpy as np
import pandas as pd

# to prevent printing of log INFO to screen - needs to be set before tf import
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf

from GPSat import get_parent_path, get_data_path
from GPSat.local_experts import LocalExpertOI
from GPSat.utils import get_config_from_sysargv, json_serializable, nested_dict_literal_eval, cprint
from GPSat.models import get_model

# --
# helper functions
# --

def get_local_expert_oi_config():

    configs = get_config_from_sysargv()

    # assert config is not None, f"config is None, issue reading it in? check argument provided to script"
    # if config not read in - get default / example
    if configs is None:
        config_file = get_parent_path("configs", "example_local_expert_oi.json")
        warnings.warn(f"\n\nconfig is empty / not provided, will just use an example config:\n{config_file}\n\n")
        with open(config_file, "r") as f:
            configs = nested_dict_literal_eval(json.load(f))

        # specify paths and input values
        configs["results"]["dir"] = get_parent_path("results", "example")
        configs["results"]["file"] = "ABC_binned_example.h5"
        configs["locations"]["source"] = get_data_path("locations", "example_expert_locations_arctic_no_date.csv")
        configs["data"]["data_source"] = get_data_path("example", "ABC_binned.h5")
        configs["pred_loc"]["df_file"] = get_data_path("locations", "2d_xy_grid_5x5km.csv")

    # if configs are not a list, make them into one
    if not isinstance(configs, list):
        print("putting config into a list")
        configs = [configs]

    assert isinstance(configs, list), "configs expected to be list"

    return configs


if __name__ == "__main__":

    print("GPUs:", tf.config.list_physical_devices('GPU'))

    pd.set_option("display.max_columns", 200)

    # ---
    # read config
    # ---

    configs = get_local_expert_oi_config()

    # ----
    # increment over the list of configs
    # ----

    t1 = time.time()
    for config_count, config in enumerate(configs):

        cprint("-" * 30, c="BOLD")
        cprint("will attempt to use LocalExpertOI.run(...) using components from the following config:", c="OKCYAN")
        cprint(json.dumps(json_serializable(config), indent=4), c="HEADER")

        # ------
        # (extract) parameters
        # ------

        # check for run used kwargs, warn/error if any exists
        valid_keys = ["comment", "run_kwargs", "misc", "locations", "data", "model", "pred_loc", "results"]
        invalid_keys = [k for k in config if k not in valid_keys]
        assert len(invalid_keys) == 0, \
            f"the following invalid keys were provided in the config:\n{invalid_keys}\nremove and try again"

        cprint("*" * 50, "OKBLUE")
        cprint("*" * 50, "OKBLUE")
        cprint("*" * 50, "OKBLUE")
        print(f"config count: {config_count+1}/{len(configs)}")

        # print(json.dumps(json_serializable(config), indent=4))

        # ------
        # (extract) parameters
        # ------

        # pop out and print "comment"
        comment = config.pop("comment", None)
        comment = "\n".join(comment) if isinstance(comment, list) else comment
        print(f"\nconfig 'comment':\n\n{comment}\n\n")

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

        # legacy handling of how to get store_path
        if "results" in config:
            results = config["results"]
            store_path = os.path.join(results["dir"], results["file"])
        else:
            store_path = run_kwargs.pop("store_path")

        locexp.run(store_path=store_path,
                   **run_kwargs)

        cprint(f"results were written to:\n{store_path}", c="OKGREEN")

    t2 = time.time()
    cprint(f"Total run time: {t2 - t1:.2f} seconds", c="OKGREEN")
