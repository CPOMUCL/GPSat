# simple example of using LocalExpertOI class using example data

# HOW TO: generate example input data
# - data/example/ABC.h - run: notebooks/read_raw_data_and_store.ipynb
# - data/example/ABC_binned.zarr - run: notebooks/bin_raw_data.ipynb

# BEFORE RUNNING: Double check the inline config below!

import os
import re
import time
import tables
import numpy as np
import pandas as pd

import tensorflow as tf

from PyOptimalInterpolation import get_parent_path, get_data_path
from PyOptimalInterpolation.utils import check_prev_oi_config, get_previous_oi_config
from PyOptimalInterpolation.models import GPflowGPRModel
from PyOptimalInterpolation.dataloader import DataLoader
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
        "file": f"ABC_binned2.h5"
    },
    "input_data": {
        "file_path": get_data_path("example", f"ABC_binned.zarr"),
        "obs_col": "obs",
        "coords_col": ['x', 'y', 't'],
        "col_funcs": {
            "date": {"func": "lambda x: x.astype('datetime64[D]')", "col_args": "date"},
            "t": {"func": "lambda x: x.astype('datetime64[D]').astype(int)", "col_args": "date"}
        }
    },
    "local_expert_locations": {
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
    # from either ncdf, zarr or ndf
    "global_select": [
        # static where - condition like this wi
        {"col": "lat", "comp": ">=", "val": 60},
        # dynamic where - become a function of expert location and local
        # - loc_col - column / coordinate of expert location
        # - src_col - column / coordinate in data source
        # - func - function to transform loc_col (from expert location)
        #     and val (from local_select)
        {"loc_col": "t", "src_col": "date", "func": "lambda x,y: np.datetime64(pd.to_datetime(x+y, unit='D'))"}
    ],
    # how to select data for local expert - i.e. within the vicinity
    "local_select": [
        {"col": "t", "comp": "<=", "val": days_ahead},
        {"col": "t", "comp": ">=", "val": -days_behind},
        {"col": ["x", "y"], "comp": "<", "val": incl_rad}
    ],
    "constraints": {
        "lengthscales": {
            "low": [0, 0, 0],
            "high": [2 * incl_rad, 2 * incl_rad, days_ahead + days_behind + 1]
        }
    },
    # DEBUGGING: shouldn't skip model params - only skip misc (?)
    # "skip_valid_checks_on": ["local_expert_locations", "misc", "results", "input_data"],
    "skip_valid_checks_on": ["local_expert_locations", "misc"],
    # parameters to provide to model (inherited from BaseGPRModel) when initialising
    "model_params": {
        "coords_scale": [50000, 50000, 1],
        "obs_mean": None
    },
    "misc": {
        "store_every": 10,
    }
}

# -----
# (extract) parameters
# ------

# local expert locations
local_expert_locations = oi_config["local_expert_locations"]

# store result in a directory
result_dir = oi_config['results']['dir']
result_file = oi_config['results']['file']
os.makedirs(result_dir, exist_ok=True)
store_path = os.path.join(result_dir, result_file)

# global selection criteria
global_select = oi_config.get("global_select", [])

# selection criteria for local expert
local_select = oi_config.get("local_select", [])

# TODO: all "skip_valid_checks_on" in config just to be a str -> convert to list
skip_valid_checks_on = ["skip_valid_checks_on"] + oi_config.get("skip_valid_checks_on", [])

# input data
# CURRENTLY ONLY HANDLES FILES FOR open_dataset
input_data_file = oi_config['input_data']['file_path']
input_col_funcs = oi_config['input_data'].get("col_funcs", {})

# columns containing observations and coordinates
obs_col = oi_config['input_data']['obs_col']
coords_col = oi_config['input_data']['coords_col']

# parameters for model
model_params = oi_config.get("model_params", {})

# misc
misc = oi_config.get("misc", {})
store_every = misc.get("store_every", 10)


# ---
# check previous oi config (if exists) is consistent with one provided
# ---

# TODO: review checking of previous configs
prev_oi_config, skip_valid_checks_on = get_previous_oi_config(store_path, oi_config,
                                                              skip_valid_checks_on=skip_valid_checks_on)

# check previous oi_config matches current - want / need them to be consistent (up to a point)
check_prev_oi_config(prev_oi_config, oi_config,
                     skip_valid_checks_on=skip_valid_checks_on)

# --------
# initialise LocalExpertOI object
# --------

locexp = LocalExpertOI()

# ---------
# set data_source - where data will be read from (either from filesystem or DataFrame)
# ---------

locexp.set_data_source(file=input_data_file)

# ---------
# expert locations
# ---------

locexp.local_expert_locations(**local_expert_locations)

# remove previously found local expert locations
# - determined by (multi-index of) 'run_details' table
xprt_locs = locexp._remove_previously_run_locations(store_path,
                                                    locexp.expert_locs.copy(True),
                                                    table="run_details")

# ----------------
# Increment over the expert locations
# ----------------

# create a dictionary to store result (DataFrame / tables)
store_dict = {}
count = 0
df, prev_where = None, None
for idx, rl in xprt_locs.iterrows():

    # TODO: use log_lines
    print("-" * 30)
    count += 1
    print(f"{count} / {len(xprt_locs)}")

    # start timer
    t0 = time.time()

    # ----------------------------
    # (update) global data - from data_source (if need be)
    # ----------------------------

    df, prev_where = locexp._update_global_data(df=df,
                                                global_select=global_select,
                                                local_select=local_select,
                                                ref_loc=rl,
                                                prev_where=prev_where,
                                                col_funcs=input_col_funcs)

    # ----------------------------
    # select local data - relative to expert's location - from global data
    # ----------------------------

    df_local = DataLoader.local_data_select(df,
                                            reference_location=rl,
                                            local_select=local_select,
                                            verbose=False)
    print(f"number obs: {len(df_local)}")

    # if there are too few observations store to 'run_details' (so can skip later) and continue
    if len(df_local) <= 2:
        save_dict = {
            "run_details": pd.DataFrame({
                "num_obs": len(df_local),
                "run_time": np.nan,
                "mll": np.nan,
                "optimise_success": False
            }, index=[0])
        }
        store_dict = locexp._append_to_store_dict_or_write_to_table(ref_loc=rl,
                                                                    save_dict=save_dict,
                                                                    store_dict=store_dict,
                                                                    store_path=store_path,
                                                                    store_every=store_every)
        continue

    # -----
    # build model - provide with data
    # -----

    # initialise model
    # TODO: needed to review the unpacking of model_params, when won't it work?
    gpr_model = GPflowGPRModel(data=df_local,
                               obs_col=obs_col,
                               coords_col=coords_col,
                               **model_params)

    # --
    # apply constraints
    # --

    # TODO: generalise this to apply any constraints - use apply_param_transform (may require more checks)
    #  - may need information from config, i.e. obj = model.kernel, specify the bijector, other parameters

    if "lengthscales" in oi_config['constraints']:
        print("applying lengthscales contraints")
        low = oi_config['constraints']['lengthscales'].get("low", np.zeros(len(coords_col)))
        high = oi_config['constraints']['lengthscales'].get("high", None)
        gpr_model.set_lengthscale_constraints(low=low, high=high, move_within_tol=True, tol=1e-8, scale=True)

    # --
    # optimise parameters
    # --

    opt_dets = gpr_model.optimise_hyperparameters()

    # get the hyper parameters - for storing
    hypes = gpr_model.get_hyperparameters()

    # --
    # make prediction - at the local expert location
    # --

    pred = gpr_model.predict(coords=rl)
    # - remove y to avoid conflict with coordinates
    # pop no longer needed?
    pred.pop('y')

    # remove * from names - causes issues when saving to hdf5 (?)
    # TODO: make this into a private method
    for k, v in pred.items():
        if re.search("\*", k):
            pred[re.sub("\*","s", k)] = pred.pop(k)

    t1 = time.time()

    # ----
    # store results in tables (keys) in hdf file
    # ----

    run_time = t1-t0

    device_name = gpr_model.cpu_name if gpr_model.gpu_name is None else gpr_model.gpu_name

    # run details / info - for reference
    run_details = {
        "num_obs": len(df_local),
        "run_time": run_time,
        # "device": device_name,
        "mll": opt_dets['marginal_loglikelihood'],
        "optimise_success":  opt_dets['optimise_success']
    }

    # store data to specified tables according to key
    # - will add mutli-index based on location
    pred_df = pd.DataFrame(pred, index=[0])
    pred_df.rename(columns={c: re.sub("\*", "s", c) for c in pred_df.columns}, inplace=True)
    save_dict = {
        "preds": pred_df,
        "run_details": pd.DataFrame(run_details, index=[0]),
        **hypes
    }

    # ---
    # 'store' results
    # ---

    # change index to multi index (using ref_loc)
    # - add to table in store_dict or append to table in store_path if above store_every
    store_dict = locexp._append_to_store_dict_or_write_to_table(ref_loc=rl,
                                                                save_dict=save_dict,
                                                                store_dict=store_dict,
                                                                store_path=store_path,
                                                                store_every=store_every)

    t2 = time.time()
    print(f"total run time : {t2-t0:.2f} seconds")


