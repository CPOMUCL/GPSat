# using some reference config and specifying which data should be held out
# create multiple oi_configs, which can be store in a list to a json file
# - specify a single xval_row_select, which can take in 'date' via '%s'
# - multiple dates can be increment over, each will create a config
# - above row_select will be used to create prediction locations (held out data)
# - - prediction locations are written (appended) to file
# - - pred_loc_config is create
# - the negative / inverse of the row_select will be add to the data_config row_select to ensure observations are held out
# - create a list oi_configs, save them to file: read in and run with examples/local_expert_oi_xval.py

# NOTE: in line modification is currently required

import os
import warnings
import time
import json

import numpy as np
import pandas as pd

from PyOptimalInterpolation.dataloader import DataLoader
from PyOptimalInterpolation import get_parent_path, get_data_path
from PyOptimalInterpolation.local_experts import LocalExpertOI, LocalExpertData
from PyOptimalInterpolation.utils import get_config_from_sysargv, nested_dict_literal_eval, grid_2d_flatten
from PyOptimalInterpolation.utils import json_serializable

pd.set_option("display.max_columns", 200)

# ---
# helper functions
# ---

def get_prediction_data_config_for_xval(data_config,
                                        xval_row_select,
                                        max_dist,
                                        pred_loc_load_kwargs=None):

    # TODO: add location directories, with file names for expert and predictions
    # TODO: replace / drop columns from expert locations

    assert isinstance(xval_row_select, dict), f"xval_row_select must be a dict, got: {type(xval_row_select)}"

    # -----
    # 1) extract the hold out data - to be used as the prediction locations
    # -----

    dc_copy = data_config.copy()

    # add to the data_config row_select the xval_row_select - so can get just the hold_out data
    dc_copy['row_select'] = dc_copy.get("row_select", []) + [xval_row_select]

    loi = LocalExpertOI(data_config=dc_copy)

    # load the prediction locations / hold out data
    pred_loc = loi.data.load(reset_index=False)

    # ----
    # 2) exclude the prediction locations / hold out data when reading in data
    # ----

    # negate the xval_row_select and add it to the data row_select
    # - negating will invert (apply NOT or ~) to selection bool array - i.e. True -> False and vice versa
    # - this will prevent the held out data from being read in
    neg_xval_row_select = xval_row_select.copy()
    neg_xval_row_select['negate'] = True

    dc_copy = data_config.copy()
    dc_copy['row_select'] = dc_copy.get("row_select", []) + [neg_xval_row_select]

    # ----
    # 3) create pred_loc config
    # ----

    if pred_loc_load_kwargs is None:
        pred_loc_load_kwargs = {}

    pred_loc_config = {
        "method": "from_source",
        "load_kwargs": {
            **pred_loc_load_kwargs
        },
        "max_dist": max_dist
    }

    # return
    # - prediction location config
    # - data config
    # - prediciton locations
    return pred_loc_config, dc_copy, pred_loc


# ---
# reference LocalExpertOI config - to be used as basis for config
# ---

# TODO: specify reference config as input parameter (file path) from config
# NOTE: reference config will be used for data, model and location
ref_config = get_config_from_sysargv()

# assert ref_config is not None, f"ref_config is None, issue reading it in? check argument provided to script"
# if ref_config not read in - get default / example

config_file = ""
if ref_config is None:

    config_file = get_parent_path("configs", "example_local_expert_oi_reference_for_xval.json")
    warnings.warn(f"\n\nref_config is empty / not provided, will just use an example ref_config:\n{config_file}\n\n")
    with open(config_file, "r") as f:
        ref_config = nested_dict_literal_eval(json.load(f))

    # specify paths and input values
    ref_config["results"]["dir"] = get_parent_path("results", "example")
    ref_config["results"]["file"] = "ABC_binned_example_xval.h5"
    ref_config["locations"]["source"] = get_data_path("locations", "example_expert_locations_arctic_no_date.csv")
    ref_config["data"]["data_source"] = get_data_path("example", "ABC_binned.h5")


# # alternatively could read in reference config and then modify
# config_file = "/path/to/ref/config.json"
# with open(config_file, "r") as f:
#     ref_config = json.load(f)


# ----
# parameters: CHANGE AS NEEDED
# ----

# "run_kwargs" to be added to every config
run_kwargs = {
    "store_every": 5,
    # NOTE: currently for xval - will iterate over a list of configs, so configs will mostly not be compatible
    "check_config_compatible": False
}

# "results" config to be added to every config
results_config = {
    "dir": get_parent_path("results", "example"),
    "file": "ABC_binned_example_xval.h5"
}

# *************
# xval row_select: a SINGLE row_select statement for selecting data to hold out
# - these held out values will become the prediction locations
# - and the negative of this row_select will be used exclude the data from import

# row select use to select hold out data, will be applied to data
xval_row_select = {
        "func": "lambda source, date: (source == 'B') & (date == np.datetime64('%s'))",
        "col_args": ["source", "date"]
    }
# *************

# date to increment over - these will be used in the above to 'fill in' '%s'
dates = ["2020-03-04", "2020-03-05"]

# maximum distance to perform predictions on
# - used in pred_locs_config
max_dist = 200 * 1000


# where should the list of configs be written
output_configs_dir = get_parent_path("configs")
os.makedirs(output_configs_dir, exist_ok=True)
list_of_config_file_name = "xval_list_of_oi_configs.json"


# where to write prediction locations
output_locations_dir = get_parent_path("data", "locations")
os.makedirs(output_locations_dir, exist_ok=True)

# prediction locations file name
pred_loc_file_name = "xval_prediction_locations.csv"
pred_loc_source = os.path.join(output_locations_dir, pred_loc_file_name)

# if prediction location exists remove
assert not os.path.exists(pred_loc_source), f"pred_loc_source:\n{pred_loc_source}\nexists, remove or change file name /location"
# somewhat dangerous to assume safe to delete, use above assert
# if os.path.exists(pred_loc_source):
#     print(f"pred_loc_source:\n{pred_loc_source}\nexists, removing")
#     os.remove(pred_loc_source)


# -----
# generate the oi_configs
# -----

data_config = ref_config['data'].copy()
expert_loc_config = ref_config['locations'].copy()
model_config = ref_config["model"].copy()

# store oi_configs to a list, to be written to file later
oi_config_list = []

for date in dates:

    print("------")
    print(f"creating xval config for date: {date}")

    # ----
    # data and pred_loc configs
    # ----

    xrs = xval_row_select.copy()
    xrs["func"] = xrs["func"] % date

    # prediction location load_kwargs
    pred_loc_load_kwargs = {
        "source": os.path.join(output_locations_dir, pred_loc_file_name),
        "col_funcs": {
            "date": {
                "func": "lambda x: x.astype('datetime64[D]')",
                "col_args": "date"
            }
        },
        "row_select": [{"col": "date", "comp": "==", "val": date}]
    }

    pred_loc_config, dc_copy, pred_loc = get_prediction_data_config_for_xval(data_config=data_config,
                                                                             xval_row_select=xrs,
                                                                             max_dist=max_dist,
                                                                             pred_loc_load_kwargs=pred_loc_load_kwargs)

    # ---
    # expert location config
    # ---

    # NOTE: expert_location should not have 'date' coorindates, these should be added in
    elc = expert_loc_config.copy()

    # add the current date for cross validation
    elc["add_data_to_col"] = {"date": [date]}

    # ---
    # create oi_config
    # ---

    oi_config = {
        "results": results_config,
        "data": dc_copy,
        "locations": elc,
        "model": model_config,
        "pred_locs": pred_loc_config,
        "run_kwargs": run_kwargs
    }

    if "comment" in ref_config:
        oi_config["comment"] = ref_config["comment"]

    oi_config_list.append(json_serializable(oi_config))

    # ---
    # write the prediction locations / hold out data to file
    # ---

    pred_loc_file = pred_loc_config['load_kwargs']['source']

    # append to existing file
    pred_loc.to_csv(pred_loc_file, index=False, mode='a', header=not os.path.exists(pred_loc_file))


# write list of configs to file
with open(os.path.join(output_configs_dir, list_of_config_file_name), "w") as f:
    json.dump(oi_config_list, f, indent=4)


# ----
# validation - check, for each config, there is no overlap between the prediction locations and the data
# ----


if config_file == get_parent_path("configs", "example_local_expert_oi_reference_for_xval.json"):

    # HARDCODED! this works with default example, but might not work with different datasets
    merge_col = ['x', 'y', 't', 'source']

    print("checking overlap of prediction locations and observation data for each config")
    for config in oi_config_list:

        loi = LocalExpertOI(data_config=config["data"])

        # NOTE: this will load all the observation data - becareful
        obs = loi.data.load()

        pred_locs = DataLoader.load(**config['pred_locs']['load_kwargs'])


        chk = obs.merge(pred_locs,
                        on=merge_col,
                        how='outer',
                        indicator=True)

        assert "both" not in chk['_merge'].unique()

