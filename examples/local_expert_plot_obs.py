# local expert OI on observations sampled from known ground truth (mss - geoid)
# NOTE!: this script requires global_select to be specified in config - plots get generated when global data changes
# - e.g.     "global_select":        {
#                 "loc_col": "t",
#                 "src_col": "date",
#                 "func": "lambda x,y: np.datetime64(pd.to_datetime(x+y, unit='D'))"
#             }

import os
import re
import json
import time

import numpy as np
import pandas as pd

import tensorflow as tf

from PyOptimalInterpolation import get_parent_path, get_data_path
from PyOptimalInterpolation.local_experts import LocalExpertOI
from PyOptimalInterpolation.utils import grid_2d_flatten

from PyOptimalInterpolation.utils import json_serializable,  get_config_from_sysargv

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

# --
# helper functions
# --


print("GPUs:", tf.config.list_physical_devices('GPU'))

pd.set_option("display.max_columns", 200)

# ---
# config
# ---


oi_config = get_config_from_sysargv(argv_num=1)

assert oi_config is not None, f"config is empty / not provided, must specify path to config (json file) as argument"


# -----
# (extract) parameters
# ------

# don't need  the prediction locations
oi_config.pop("pred_loc", None)

projection = "north"

results = oi_config['results']

# # TODO: all "skip_valid_checks_on" in config just to be a str -> convert to list
# skip_valid_checks_on = ["skip_valid_checks_on"] + oi_config.get("skip_valid_checks_on", [])

store_path = os.path.join(results['dir'], results['file'])
image_file = os.path.join(os.path.dirname(store_path), re.sub("\..*$", ".pdf", os.path.basename(store_path)))

# to add lon, lat columns in expert locations - from x,y
# exprt_lon_lat = {
#     ("lon", "lat"): {
#         "source": "PyOptimalInterpolation.utils",
#         "func": "EASE2toWGS84_New",
#         "col_kwargs": {
#             "x": "x",
#             "y": "y"
#         },
#         "kwargs": {
#             "lat_0": 90
#         }
#     }
# }
exprt_lon_lat = None


# --------
# initialise LocalExpertOI object
# --------

locexp = LocalExpertOI(data_config=oi_config['data'],
                       model_config=oi_config['model'],
                       pred_loc_config=oi_config.get("pred_loc", None))

# ----------------
# Increment over the expert locations
# ----------------

# os.makedirs(results['dir'], exist_ok=True)

start = time.time()

locexp.plot_locations_and_obs(image_file=image_file,
                              projection=projection,
                              xrpt_loc_col_funcs=exprt_lon_lat,
                              s=2,
                              s_exprt_loc=100)


end = time.time()


print("--"*10)
print(f"Total run time: {end-start: .2f} seconds")
print("--"*10)


print(f"output file:\n{image_file}")
