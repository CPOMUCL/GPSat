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
# TODO: deal with running out of memory?
# os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'

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
        "file": f"ABC_raw_SGPR2.h5"
    },
    "locations": {
        # file path of expert locations
        "file": get_data_path("example", "locations.csv"),
        # columns shall be added or manipulated as follows - are these needed?
        "col_funcs": {
            "date": {"func": "lambda x: x.astype('datetime64[D]')", "col_args": "date"},
            "t": {"func": "lambda x: x.astype('datetime64[D]').astype(int)", "col_args": "date"},
        },
        # (optional) keep only relevant columns - (could keep all?)
        # - should contain coord_col (see data), if more will be added to separate table 'coordinates'
        "keep_cols": ["x", "y", "t", "date", "lon", "lat"],
        # select a subset of expert locations
        "row_select": [
            # select locations with dates in Dec 2018
            {"col": "date", "comp": "==", "val": "2020-03-05"},
            # {"col": "date", "comp": "==", "val": "2020-03-05"},
            # {"col": "date", "comp": "<=", "val": "2020-03-06"},
            {"col": "lat", "comp": ">=", "val": 65},
            {"col": "s", "comp": ">=", "val": 0.15}
        ],
        # (optional) - sort locations by some column
        "sort_by": "date"
    },
    "data": {
        "data_source": get_data_path("example", f"ABC.h5"),
        "table": "data",
        "col_funcs": {
            "x": {
                "source": "PyOptimalInterpolation.utils",
                "func": "WGS84toEASE2_New",
                "col_kwargs": {"lon": "lon", "lat": "lat"},
                "kwargs": {"return_vals": "x"}
            },
            "y": {
                "source": "PyOptimalInterpolation.utils",
                "func": "WGS84toEASE2_New",
                "col_kwargs": {"lon": "lon", "lat": "lat"},
                "kwargs": {"return_vals": "y"}
            },
            "t": {"func": "lambda x: x.astype('datetime64[s]').astype(float) / (24 * 60 * 60)", "col_args": "datetime"},
        },
        "obs_col": "obs",
        "coords_col": ['x', 'y', 't'],
        "local_select": [
            {"col": "t", "comp": "<=", "val": days_ahead},
            {"col": "t", "comp": ">=", "val": -days_behind},
            {"col": ["x", "y"], "comp": "<", "val": incl_rad}
        ],
        # (optional) - read in a subset of data from data_source (rather than reading all into memory)
        # "global_select": [
        #     {"col": "lat", "comp": ">=", "val": 60},
        #     {"loc_col": "t", "src_col": "date", "func": "lambda x,y: np.datetime64(pd.to_datetime(x+y, unit='D'))"}
        # ]

    },
    "model": {
        # "model": "PyOptimalInterpolation.models.GPflowGPRModel",
        # "oi_model": "GPflowGPRModel",
        "oi_model": "GPflowSGPRModel",
        # (optional) extract parameters to provide when initialising oi_model
        "init_params": {
            "coords_scale": [50000, 50000, 1],
            "obs_mean": None,
            "num_inducing_points": 1000
        },
        # (optional) load/set parameters - either specify directly or read from file
        "load_params": {
            "previous": True,
            # read from results file? or could be another
            # "file": get_parent_path("results", "example", f"ABC_binned5.h5"),
            # parameters from the reference location will be fetched
            # - index_adjust allows for a shift
            # "index_adjust": {"t": {"func": "lambda x: x-1"}}
        },
        "constraints": {
            "lengthscales": {
                "low": [0, 0, 0],
                "high": [2 * incl_rad, 2 * incl_rad, days_ahead + days_behind + 1]
            }
        }
    },
    # DEBUGGING: shouldn't skip model params - only skip misc (?)
    "skip_valid_checks_on": ['model', 'locations', 'data'],
    "misc": {
        "store_every": 1,
    }
}

# -----
# (extract) parameters
# ------

results = oi_config['results']

# run_kwargs - previously named misc
run_kwargs = oi_config.get("run_kwargs", oi_config.get("misc", {}))

# legacy handling of skip_valid_checks_on being in config
if "skip_valid_checks_on" not in run_kwargs:
    skip_valid_checks_on = ["skip_valid_checks_on"] + oi_config.get("skip_valid_checks_on", [])
    run_kwargs["skip_valid_checks_on"] = skip_valid_checks_on

# --------
# initialise LocalExpertOI object
# --------

locexp = LocalExpertOI(data_config=oi_config['data'], model_config=oi_config['model'])

# ----------------
# Increment over the expert locations
# ----------------

store_path = os.path.join(results['dir'], results['file'])

locexp.run(store_path=store_path,
           **run_kwargs)

