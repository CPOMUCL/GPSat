# local expert OI on observations sampled from known ground truth (mss - geoid)


import os

import numpy as np
import pandas as pd
import time

import tensorflow as tf

from PyOptimalInterpolation import get_parent_path, get_data_path
from PyOptimalInterpolation.local_experts import LocalExpertOI
from PyOptimalInterpolation.utils import grid_2d_flatten

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

# prediction locations (on EASE2.0 grid)
xy_range = [-4500000.0, 4500000.0]
X = grid_2d_flatten(xy_range, xy_range, step_size=5 * 1000)
df = pd.DataFrame(X, columns=['y', 'x'])

# REVIEW BELOW: namely results, input_data, local_expert_locations
# oi_config file
oi_config = {
    "results": {
        "dir": get_parent_path("results", "example"),
        "file": f"ground_truth_w_noise.h5"
    },
    "locations": {
        # file path of expert locations
        "file": get_data_path("locations", "expert_locations_200km_subset.csv"),
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
            {"col": "lat", "comp": ">=", "val": 60}
        ]
    },
    "data": {
        "data_source": get_data_path("example", f"along_track_sample_from_mss_ground_truth.csv"),
        "obs_col": "obs_w_noise",
        "coords_col": ['x', 'y', 't'],
        "local_select": [
            {"col": "t", "comp": "<=", "val": days_ahead},
            {"col": "t", "comp": ">=", "val": -days_behind},
            {"col": ["x", "y"], "comp": "<", "val": incl_rad}
        ]
    },
    "model": {
        "oi_model": "GPflowSGPRModel",
        # (optional) extract parameters to provide when initialising oi_model
        "init_params": {
            "coords_scale": [50000, 50000, 1],
            "obs_mean": None,
            "num_inducing_points": 1000
        },
        "constraints": {
            "lengthscales": {
                "low": [0, 0, 0],
                "high": [2 * incl_rad, 2 * incl_rad, days_ahead + days_behind + 1]
            }
        }
    },
    # prediction location - optional - if not specified / provided will default to expert location
    "pred_loc": {
        "method": "from_dataframe",
        "df": df,
        "max_dist": 200 * 1000
    },
    # DEBUGGING: shouldn't skip model params - only skip misc (?)
    "skip_valid_checks_on": ['misc'],
    "misc": {
        "store_every": 1,
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

locexp = LocalExpertOI(data_config=oi_config['data'], model_config=oi_config['model'],
                       pred_loc_config=oi_config.get("pred_loc", None))

# ----------------
# Increment over the expert locations
# ----------------

store_path = os.path.join(results['dir'], results['file'])

start = time.time()

locexp.run(store_path=store_path,
           store_every=store_every,
           check_config_compatible=True,
           skip_valid_checks_on=skip_valid_checks_on)

end = time.time()

print("--"*10)
print(f"Total run time: {end-start} seconds")
print("--"*10)

