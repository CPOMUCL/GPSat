# local expert OI on observations
# TODO: this script needs cleaning up, to be made more robust
#  - should work with any oi_config.json
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

from GPSat import get_parent_path, get_data_path
from GPSat.local_experts import LocalExpertOI
from GPSat.utils import grid_2d_flatten
from GPSat.utils import json_serializable,  get_config_from_sysargv, cprint
from examples.local_expert_oi import get_local_expert_oi_config

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

# --
# helper functions
# --


print("GPUs:", tf.config.list_physical_devices('GPU'))

pd.set_option("display.max_columns", 200)

# ---
# config
# ---

# oi_config = get_config_from_sysargv(argv_num=1)
oi_config = get_local_expert_oi_config()

assert oi_config is not None, f"config is empty / not provided, must specify path to config (json file) as argument"

if isinstance(oi_config, list):
    cprint("oi_config provided is a list, currently can only handle dict, taking first entry", c="WARNING")
    oi_config = oi_config[0]

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
image_file = os.path.join(os.path.dirname(store_path), re.sub("\..*$", "_OBS_AND_EXPERT_LOCATIONS.pdf", os.path.basename(store_path)))

# to add lon, lat columns in expert locations - from x,y
# exprt_lon_lat = {
#     ("lon", "lat"): {
#         "source": "GPSat.utils",
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
                       pred_loc_config=oi_config.get("pred_loc", None),
                       expert_loc_config=oi_config['locations'])

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


cprint("--"*10, c="OKBLUE")
cprint(f"Total run time: {end-start: .2f} seconds", c="OKBLUE")
cprint("--"*10, c="OKBLUE")


cprint(f"output file:\n{image_file}", c="OKGREEN")
