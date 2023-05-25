# THIS WAS LAZILY COPIED FROM create_xval_config_from_ref_config.py
# - aim is to modify to hold out specific tracks
# create a list of configs, each performing cross validation for a single "date"

import copy
# NOTE: in line modification is currently required

import os
import re
import sys
import warnings
import time
import json

import numpy as np
import pandas as pd

from PyOptimalInterpolation import get_parent_path, get_data_path
from PyOptimalInterpolation.dataloader import DataLoader
from PyOptimalInterpolation.utils import get_config_from_sysargv, cprint, json_serializable
import inspect

pd.set_option("display.max_columns", 200)

# ---
# helper functions
# ---

def return_as_list(x):
    if isinstance(x, list):
        return x
    else:
        return [x]

def remove_bad_table_char(x):
    # move bad characters - those that can be used in table name for HDF5
    # TODO: there are probably more
    return re.sub("-", "", x)


# ---
# reference LocalExpertOI config - to be used as basis for config
# ---

# TODO: specify reference config as input parameter (file path) from config
# NOTE: reference config will be used for data, model and location
ref_config = get_config_from_sysargv()

assert ref_config is not None, f"ref_config is None, issue reading it in? check argument provided to script"

# ----
# parameters: CHANGE AS NEEDED
# ----

# output_file = get_parent_path("configs", "GPOD_elev_from_lead_XVAL.json")
output_file = None

# prediction (data) locations will be selected by date, and then increment over
# unique values in second_hold_out_col
second_hold_out_col = "track_start"

# which column is the 'date' column
date_col = "date"
date_col = "start_date"

# dates: a config for xval will be created for each date
dates = ["2019-01-15", "2019-02-15", "2019-03-15", "2019-04-15"]
# dates = ["2019-12-01", "2020-01-01", "2020-02-01", "2020-03-01",
#          "2019-12-15", "2020-01-15", "2020-02-15", "2020-03-15"]
# dates = np.arange(np.datetime64("2020-02-01"), np.datetime64("2020-02-29")).astype('str').tolist()


# template for prediction location, change max_dist as needed, method should be 'from_source'
# - load_kwargs will be added later (derived from data config)

pred_loc = {
    "method": "from_source",
    "max_dist": 200 * 1000,
}
# add date hold out (row_select) to predicton locations?
hold_out_data_on_pred_loc = True

# generate a prediction on fine grid (not hold out positions)
# pred_loc = {
#     "method": "from_source",
#     "max_dist": 200 * 1000,
#     "load_kwargs": {
#         "source": get_data_path("locations", "2d_xy_grid_5x5km.csv")
#     },
# }
# # add date hold out (row_select) to prediction locations?
# hold_out_data_on_pred_loc = False




# xval row_select: a SINGLE row_select statement for selecting data to hold out
# NOTE: expected to have 'date' in func string, which will be populated via format
# - these held out values will become the prediction locations
# - and the negative of this row_select will be used exclude the data from import

# row select use to select hold out data, will be applied to data
# xval_row_select = {
#         "func": "lambda source, date: (source == 'B') & (date == np.datetime64('date'))",
#         "col_args": ["source", "date"]
#     }
# xval_row_select = {
#         "func": "lambda date, sat: (date == np.datetime64('{date}')) & (sat == 'S3A') ",
#         "col_args": ["date", "sat"]
#     }

# primary hold out select
xval_row_select = {
        "func": "lambda date: date == np.datetime64('{date}') ",
        "col_args": date_col
    }

# secondary hold out select
second_row_select = {
        # NOTE: add or remove ' around {other_col} as needed (depending on str/datetime or not)
        "func": "lambda date, other_col: (date == np.datetime64('{date}')) & (other_col == {other_col}) ",
        "col_args": [date_col, second_hold_out_col]
    }

# xval_row_select = {
#         "func": "lambda date: (date == np.datetime64('{date}')) & (start_track == 'start_track') ",
#         "col_args": [date_col, "start_track"]
#     }

# make a copy of the config (needed?)
org_config = copy.deepcopy(ref_config)

# ---
# specify output file if not provided
# ---

# if output file not specified inline, then get from the sys.argv
if output_file is None:
    output_file = re.sub("\.json", "_XVAL.json", sys.argv[1], re.IGNORECASE)

# -------
# prediction locations: get load_kwargs from data config (assumed to be loading from hdf5!)
# -------

# remove any existing prediction location
ref_config.pop('pred_loc', None)

# prep / handle kwargs for DataLoader.load
load_kwargs = copy.deepcopy(ref_config['data'])
load_kwargs['source'] = load_kwargs.pop("data_source")

load_params = [k for k in inspect.signature(DataLoader.load).parameters]
for k in list(load_kwargs.keys()):
    if k not in load_params:
        load_kwargs.pop(k)

# add kwargs that will used for xval, if they're not there already
load_kwargs["row_select"] = load_kwargs.get("row_select", [])
load_kwargs["where"] = load_kwargs.get("where", [])

load_kwargs['row_select'] = return_as_list(load_kwargs['row_select'])
load_kwargs['where'] = return_as_list(load_kwargs['where'])

if hold_out_data_on_pred_loc:
    pred_loc['load_kwargs'] = load_kwargs


# -----
# generate the oi_configs
# -----

data_config = ref_config['data'].copy()
expert_loc_config = ref_config['locations'].copy()
model_config = ref_config["model"].copy()
run_kwargs = ref_config['run_kwargs'].copy()

# store oi_configs to a list, to be written to file later
oi_config_list = []

for date in dates:

    print("------")
    print(f"creating xval config for date: {date}")

    pl = copy.deepcopy(pred_loc)
    dc = copy.deepcopy(data_config)

    # ----
    # pred_loc configs
    # ----

    xrs = xval_row_select.copy()
    # xrs["func"] = xrs["func"] % date
    xrs["func"] = xrs["func"].format(date=date)

    #
    if hold_out_data_on_pred_loc:
        # use where to select the current date
        # - using where so only read one date from hdf5 file (which can be large)
        pl['load_kwargs']['where'].append({"col": date_col, "comp": "==", "val": date})
        # add the xval row select
        pl['load_kwargs']['row_select'].append(xrs)

    # ---
    # helper function
    # ---

    def make_xval_config(dc, elc, pl, run_kwargs, model_config, xrs):

        # ---
        # data config
        # ---

        # add the negative of the xval row select to the data selection
        neg_xval_row_select = copy.deepcopy(xrs)
        neg_xval_row_select['negate'] = True
        dc['row_select'] = dc.get("row_select", []) + [neg_xval_row_select]

        # ---
        # expert location config
        # ---

        # NOTE: expert_location should not have 'date' coordinates, these should be added in
        # elc = copy.deepcopy(expert_loc_config)

        # add the current date for cross validation
        elc["add_data_to_col"] = {"date": [date]}

        # ---
        # create oi_config
        # ---

        oi_config = {
            "data": dc,
            "locations": elc,
            "model": model_config,
            "pred_loc": pl,
            "run_kwargs": run_kwargs
        }

        if "comment" in ref_config:
            oi_config["comment"] = ref_config["comment"]

        return oi_config

    # ---
    # load current prediction locations - before
    # ---

    if second_hold_out_col is not None:
        # load the prediction columns for current date
        pldf = DataLoader.load(**pl['load_kwargs'], reset_index=False)

        u_2nd_holdout = pldf[second_hold_out_col].unique()
        for u2h in u_2nd_holdout:
            srs = second_row_select.copy()
            srs['func'] = srs['func'].format(date=date, other_col=u2h)

            pl = copy.deepcopy(pred_loc)
            pl['load_kwargs']['where'].append({"col": date_col, "comp": "==", "val": date})
            pl['load_kwargs']['row_select'].append(srs)

            oi_config = make_xval_config(dc=dc,
                                         elc=copy.deepcopy(expert_loc_config),
                                         pl=pl,
                                         run_kwargs=run_kwargs,
                                         model_config=model_config,
                                         xrs=srs)

            # create a table suff
            table_suffix = f"_{date}_{u2h}"
            table_suffix = remove_bad_table_char(table_suffix)
            print(f"table_suffix: {table_suffix}")
            oi_config["run_kwargs"]["table_suffix"] = table_suffix

            oi_config_list.append(json_serializable(oi_config))

    else:
        print("NOT USING A SECONDARY COLUMN FOR HOLDOUTS")

        oi_config = make_xval_config(dc=dc,
                                     elc=copy.deepcopy(expert_loc_config),
                                     pl=pl,
                                     run_kwargs=run_kwargs,
                                     model_config=model_config,
                                     xrs=xrs)


        oi_config_list.append(json_serializable(oi_config))


# write list of configs to file
cprint(f"writing to file:\n{output_file}", "OKBLUE")

with open(output_file, "w") as f:
    json.dump(oi_config_list, f, indent=4)

