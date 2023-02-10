# simple example of using LocalExpertOI class using example data

# HOW TO: generate example input data
# - data/example/ABC.h - run: notebooks/read_raw_data_and_store.ipynb
# - data/example/ABC_binned.zarr - run: notebooks/bin_raw_data.ipynb

# BEFORE RUNNING: Double check the inline config below!

import os

import numpy as np
import pandas as pd

import tensorflow as tf

from PyOptimalInterpolation import get_parent_path, get_data_path
from PyOptimalInterpolation.local_experts import LocalExpertOI

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

# --
# helper functions
# --


print("GPUs:", tf.config.list_physical_devices('GPU'))

pd.set_option("display.max_columns", 200)

# ---
# config
# ---

# parameters for location selection (local_select)
days_ahead = 4
days_behind = 4
incl_rad = 300 * 1000


# REVIEW BELOW: namely results, input_data, local_expert_locations
# oi_config file
oi_config = {
    "results": {
        "dir": get_parent_path("results", "example"),
        "file": f"ABC_binned3.h5"
    },
    "locations": {
        # file path of expert locations
        "file": get_data_path("example", "locations.csv"),
        # columns shall be added or manipulated as follows - are these needed?
        "col_funcs": {
            "date": {"func": "lambda x: x.astype('datetime64[D]')", "col_args": "date"},
            "t": {"func": "lambda x: x.astype('datetime64[D]').astype(int)", "col_args": "date"},
        },
        # keep only relevant columns - (could keep all?)
        "keep_cols": ["x", "y", "date", "t"],
        # select a subset of expert locations
        "row_select": [
            # select locations with dates in Dec 2018
            {"col": "date", "comp": ">=", "val": "2020-03-05"},
            {"col": "date", "comp": "<=", "val": "2020-03-06"},
            {"col": "lat", "comp": ">=", "val": 65},
            {"col": "s", "comp": ">=", "val": 0.15}
        ],
        "sort_by": "date"
    },
    "data": {
        "data_source": get_data_path("example", f"ABC_binned.zarr"),
        "col_funcs": {
            "date": {"func": "lambda x: x.astype('datetime64[D]')", "col_args": "date"},
            "t": {"func": "lambda x: x.astype('datetime64[D]').astype(int)", "col_args": "date"}
        },
        "obs_col": "obs",
        "coords_col": ['x', 'y', 't'],
        "local_select": [
            {"col": "t", "comp": "<=", "val": days_ahead},
            {"col": "t", "comp": ">=", "val": -days_behind},
            {"col": ["x", "y"], "comp": "<", "val": incl_rad}
        ],
        "global_select": [
            {"col": "lat", "comp": ">=", "val": 60},
            {"loc_col": "t", "src_col": "date", "func": "lambda x,y: np.datetime64(pd.to_datetime(x+y, unit='D'))"}
        ]

    },
    "model": {
        # "model": "PyOptimalInterpolation.models.GPflowGPRModel",
        "oi_model": "GPflowGPRModel",
        "init_params": {
            "coords_scale": [50000, 50000, 1],
            "obs_mean": None
        },
        "constraints": {
            "lengthscales": {
                "low": [0, 0, 0],
                "high": [2 * incl_rad, 2 * incl_rad, days_ahead + days_behind + 1]
            }
        }
    },
    # DEBUGGING: shouldn't skip model params - only skip misc (?)
    "skip_valid_checks_on": [],
    "misc": {
        "store_every": 10,
    }
}

# -----
# (extract) parameters
# ------

results = oi_config['results']

# # TODO: all "skip_valid_checks_on" in config just to be a str -> convert to list
skip_valid_checks_on = ["skip_valid_checks_on"] + oi_config.get("skip_valid_checks_on", [])

# misc
misc = oi_config.get("misc", {})
store_every = misc.get("store_every", 10)

# --------
# initialise LocalExpertOI object
# --------

locexp = LocalExpertOI(locations=oi_config['locations'],
                       data=oi_config['data'],
                       model=oi_config['model'])

# ----------------
# Increment over the expert locations
# ----------------

store_path = os.path.join(results['dir'], results['file'])

locexp.run(store_path=store_path,
           store_every=store_every,
           check_config_compatible=True,
           skip_valid_checks_on=skip_valid_checks_on)

