# run LocalExpertOI using provided list of configurations
# for creating list of configs see: examples/create_cross_validation_config_locations.py

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
from PyOptimalInterpolation.utils import json_serializable

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

configs = get_config_from_sysargv()

# ----
# increment over each config
# ----

for config_count, config in enumerate(configs):

    print("*" * 50)
    print("*" * 50)
    print("*" * 50)
    print(f"config count: {config_count+1}/{len(configs)}")

    # print(json.dumps(json_serializable(config), indent=4))

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

    # ----`----
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


# TODO: store list of configs in oi_config table

