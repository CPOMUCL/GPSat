# read binned data
# select local data - relative to some reference location
# provide to GP Model
# optimise hyper parameters
# make predictions
# extract hyper parameters

import os
import re
import time
import warnings

import numpy as np
import xarray as xr
import pandas as pd

from PyOptimalInterpolation import get_parent_path, get_data_path
from PyOptimalInterpolation.utils import check_prev_oi_config
from PyOptimalInterpolation.models import GPflowGPRModel
from PyOptimalInterpolation.dataloader import DataLoader


# silence INFO messages from tf
# In detail:- ref: https://stackoverflow.com/questions/70429982/how-to-disable-all-tensorflow-warnings
# 0 = all messages are logged (default behavior) ,
# 1 = INFO messages are not printed ,
# 2 = INFO and WARNING messages are not printed ,
# 3 = INFO, WARNING, and ERROR messages are not printed
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

import tensorflow as tf

gpus = tf.config.list_physical_devices('GPU')
print("GPUS")
print(gpus)

# TODO: move sections of this file into methods of a new class (GridOI?)
# TODO: want to specify select in config that can work for netcdf or ndf5
# TODO: improve / add ref location functionality
# TODO: add run attributes to the output data - local select, coord scale
#  - namely so can be build model from scratch


pd.set_option("display.max_columns", 200)

# TODO: silence tensorflow NUMA warning
# TODO: specify an OI config, store in an output table (run_details?)
#  - if table exists check config is compatible (identical?)
# ---
# parameters
# ---

# parameters for location selection
days_ahead = 4
days_behind = 4
incl_rad = 300 * 1000

# dates to perform oi on - used in local_expert_locations
oi_dates = [
    # "2020-03-01", "2020-03-02", "2020-03-03", "2020-03-04",
    # "2020-03-06",
    # "2020-03-07", "2020-03-08", "2020-03-09",
    # "2020-03-10", "2020-03-11", "2020-03-12",
    # "2020-03-13", "2020-03-14", "2020-03-15",
    # "2020-03-16", "2020-03-17", "2020-03-18",
    # "2020-03-19", "2020-03-20", "2020-03-21"
"2020-03-01", "2020-03-02", "2020-03-03", "2020-03-04",
    "2020-03-22",
    "2020-03-23", "2020-03-24", "2020-03-25",
    "2020-03-26", "2020-03-27", "2020-03-28",
    "2020-03-29", "2020-03-30", "2020-03-31"
]

# oi_config file
oi_config = {
    "results": {
        # "dir":  get_parent_path("results", "sats_ra_cry_processed_arco"),
        "dir": get_parent_path("results", "gpod_lead_25km_INVST"),
        "file": f"oi_bin_{days_ahead}_{int(incl_rad / 1000)}.h5"
    },
    "input_data": {
        # "file_path": get_data_path("binned", "sats_ra_cry_processed_arco.zarr"),
        "file_path": get_data_path("binned", "gpod_lead_25km.zarr"),
        "obs_col": "elev_mss",
        "coords_col": ['x', 'y', 't']
    },
    # from either ncdf, zarr or ndf
    "global_select": [
        # {"col": "lat", "comp": ">=", "val": 60}
    ],
    # how to select data for local expert
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
    "local_expert_locations": {
        "loc_dims": {
            "x": "x",
            "y": "y",
            "date": oi_dates
        },
        "masks": ["had_obs", {"grid_space": 2, "dims": ['x', 'y']}],
        "col_func_dict": {
            "date": {"func": "lambda x: x.astype('datetime64[D]')", "col_args": "date"},
            "t": {"func": "lambda x: x.astype('datetime64[D]').astype(int)", "col_args": "date"}
        },
        "row_select": [{'func': '>=', 'col_args': 'lat', 'args': 60}],
        "keep_cols": ["x", "y", "date", "t"],
        "sort_by": ["date"]
    },
    # DEBUGGING: shouldn't skip model params
    "skip_valid_checks_on": ["local_expert_locations", "misc"],
    # parameters to provide to model (inherited from BaseGPRModel) when initialising
    "model_params": {
        "coords_scale": [50000, 50000, 1]
    },
    "misc": {
        "store_every": 10
    }
}

# local expert locations
# HARDCODED: dates for local expert locations

local_expert_locations = oi_config["local_expert_locations"]

# store result in a directory
result_dir = oi_config['results']['dir']
result_file = oi_config['results']['file']
os.makedirs(result_dir, exist_ok=True)
store_path = os.path.join(result_dir, result_file)

# selection criteria for local expert
local_select = oi_config.get("local_select", [])

# TODO: all "skip_valid_checks_on" in config just to be a str -> convert to list
skip_valid_checks_on = ["skip_valid_checks_on"] + oi_config.get("skip_valid_checks_on", [])

# input data
# CURRENTLY ONLY HANDLES FILES FOR open_dataset
input_data_file = oi_config['input_data']['file_path']

# columns containing observations and coordinates
obs_col = oi_config['input_data']['obs_col']
coords_col = oi_config['input_data']['coords_col']

# parameters for model
model_params = oi_config.get("model_params", {})

# misc
misc = oi_config.get("misc", {})
store_every = misc.get("store_every", 10)

# ---
# read data
# ---

# TODO: allow data to be from ncdf, ndf5, or zarr
# TODO: add for selection of data here - using global_select

# connect to Dataset
ds = xr.open_dataset(input_data_file)

# get the configuration(s) use to generate dataset
raw_data_config = ds.attrs['raw_data_config']
input_data_config = ds.attrs['bin_config']

# ---
# prep data
# ---

# TODO: generalise this - want the output to dataframe, with columns added if need but
# convert to a DataFrame - dropping missing
df = ds.to_dataframe().dropna().reset_index()

# add columns that will be used as coordinates
# - this could be done with DataLoader.add_col
# - convert date (units D, not ns) to int - since 1970-01-01
df['date'] = df['date'].values.astype('datetime64[D]')
df['t'] = df['date'].values.astype('datetime64[D]').astype(int)
# df['t'] = df['date'].astype(int)

# ---------
# expert locations
# ---------

# TODO: allow local experts to be calculated based off of
#  - input files (dataframes with bools)
#  - datasets / arrays
#  - using cartopy - would require development

expert_locations = local_expert_locations

# reference data for dimensions
# TODO: need to allow for DataFrame to be used
ref_data = ds

# dimensions for the local expert
# - more (columns) can be added with col_func_dict
loc_dims = expert_locations['loc_dims']

# expert location masks
# TODO: needs work
el_masks = expert_locations.get("masks", [])
masks = DataLoader.get_masks_for_expert_loc(ref_data=ds, el_masks=el_masks, obs_col=obs_col)


cfunc = expert_locations.get("col_func_dict", None)
rsel = expert_locations.get("row_select", None)
keep_cols = expert_locations.get("keep_cols", None)
sort_by = expert_locations.get("sort_by", None)

# get the local expert locations
# - this will be a DataFrame which will be used to create a multi-index
# - for each expert values will be stored to an hdf5 using an element (row) from above multi-index
xprt_locs = DataLoader.generate_local_expert_locations(loc_dims,
                                                       ref_data=ds,
                                                       masks=masks,
                                                       row_select=rsel,
                                                       col_func_dict=cfunc,
                                                       keep_cols=keep_cols,
                                                       sort_by=sort_by)


# TODO: review if using ref_locs.index is the best way
# set multi index of ref_locs
# - this is a bit messy, done so can use index.isin(...) when reading in previous result
tmp_index = xprt_locs.set_index(xprt_locs.columns.values.tolist())
xprt_locs.index = tmp_index.index

# ------------
# remove previously found local expert locations
# ------------

# TODO: get / generate the reference location more systematically

# read existing / previous results
try:
    with pd.HDFStore(store_path, mode='r') as store:
        # get index from previous results
        prev_res = store.select('run_details', columns=[])
        keep_bool = ~xprt_locs.index.isin(prev_res.index)
        print(f"using: {keep_bool.sum()} / {len(keep_bool)} reference locations - some were already found")
        xprt_locs = xprt_locs.loc[~xprt_locs.index.isin(prev_res.index)]
except OSError as e:
    print(e)
except KeyError as e:
    print(e)


# ---
# check previous oi config is consistent
# ---


# if the file exists - it is expected to contain a dummy table (oi_config) with oi_config as attr
if os.path.exists(store_path):
    with pd.HDFStore(store_path, mode='r') as store:
        prev_oi_config = store.get_storer("oi_config").attrs['oi_config']
else:
    with pd.HDFStore(store_path, mode='a') as store:
        _ = pd.DataFrame({"oi_config": ["use get_storer('oi_config').attrs['oi_config'] to get oi_config"]},
                         index=[0])
        # TODO: change key to configs / config_info
        store.append(key="oi_config", value=_)
        store.get_storer("oi_config").attrs['oi_config'] = oi_config
        # store.get_storer("raw_data_config").attrs["raw_data_config"] = raw_data_config
        store.get_storer("oi_config").attrs['input_data_config'] = input_data_config
        prev_oi_config = oi_config

# check previous oi_config matches current - want / need them to be consistent (up to a point)
check_prev_oi_config(prev_oi_config, oi_config, skip_valid_checks_on=skip_valid_checks_on)

# ----
# increment over the reference locations
# ----

count = 0

store_dict = {}
for idx, rl in xprt_locs.iterrows():

    print("-" * 30)
    count += 1
    print(f"{count} / {len(xprt_locs)}")

    # start timer
    t0 = time.time()

    # select local data
    df_local = DataLoader.local_data_select(df,
                                            reference_location=rl,
                                            local_select=local_select,
                                            verbose=False)
    print(f"number obs: {len(df_local)}")

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

    opt_dets = gpr_model.optimise_parameters()

    # get the hyper parameters - for storing
    hypes = gpr_model.get_parameters()

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
        "device": device_name,
        "mll": opt_dets['marginal_loglikelihood'],
        "optimise_success":  opt_dets['optimise_success']
    }

    # store data to specified tables according to key
    # - will add mutli-index based on location
    save_dict = {
        "preds": pd.DataFrame(pred, index=[0]),
        "run_details": pd.DataFrame(run_details, index=[0]),
        **hypes
    }

    # ---
    # append results or write to file
    # ---

    tmp = DataLoader.make_multiindex_df(idx_dict=rl, **save_dict)

    if len(store_dict) == 0:
        store_dict = {k: [v] for k, v in tmp.items()}
        num_store = 1
    else:
        for k, v in tmp.items():
            store_dict[k] += [v]
        num_store += 1

    if num_store >= store_every:
        print("SAVING RESULTS")
        for k, v in store_dict.items():
            print(k)
            df_tmp = pd.concat(v, axis=0)
            try:
                with pd.HDFStore(store_path, mode='a') as store:
                    store.append(key=k, value=df_tmp, data_columns=True)
            except Exception as e:
                cc = 1
        store_dict = {}

    t2 = time.time()
    print(f"total run time (including saving): {t2-t0:.2f} seconds")


    # if count > 3:
    #     break
