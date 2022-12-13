# read binned data
# select local data - relative to some reference location
# provide to GP Model
# optimise hyper parameters
# make predictions
# extract hyper parameters

import os
import re
import time

import numpy as np
import xarray as xr
import pandas as pd

from PyOptimalInterpolation import get_parent_path
from PyOptimalInterpolation.utils import WGS84toEASE2_New
from PyOptimalInterpolation.models import GPflowGPRModel
from PyOptimalInterpolation.dataloader import DataLoader

import tensorflow as tf

gpus = tf.config.list_physical_devices('GPU')
print("GPUS")
print(gpus)


# TODO: add run attributes to the output data - local select, coord scale
#  - namely so can be build model from scratch

pd.set_option("display.max_columns", 200)

# ---
# parameters
# ---

# store result in a directory
result_dir = get_parent_path("results", "sats_ra_cry_processed_arco")
result_file = "oi_bin.ndf"
os.makedirs(result_dir, exist_ok=True)

# netCDF containing binned observations
# nc_file = get_parent_path("data", "binned", "gpod_202003.nc")
# TODO: change to zarr file - or confirm open_dataset can determine engine from file type
nc_file = get_parent_path("data", "binned", "sats_ra_cry_processed_arco.nc")

# columns containing observations and coordinates
obs_col = "elev_mss"
coords_col = ['x', 'y', 't']

# dates to interpolate
# oi_dates = ["2020-03-10", "2020-03-11", "2020-03-12", "2020-03-13", "2020-03-14", "2020-03-15"]
oi_dates = ["2020-03-11", "2020-03-12", "2020-03-13"]


# minimum latitude - just to reduce the number of points
min_lat = 60.0

# parameters for location selection
days_ahead = 4
days_behind = 4
incl_rad = 300 * 1000

# local selection criteria
# - points within the vals of the reference location will be selected
# TODO: local_select should be store in ndf file (per table)
local_select = [
    {"col": "t", "comp": "<=", "val": days_ahead},
    {"col": "t", "comp": ">=", "val": -days_behind},
    {"col": ["x", "y"], "comp": "<", "val": incl_rad}
]

coords_scale = [50000, 50000, 1]

verbose = False

store_every = 10

# result_file = f"daysahead{days_behind}_daysbehind{days_behind}_inclrad{incl_rad}.h5"
store_path = os.path.join(result_dir, result_file)

# ---
# read data
# ---

# connect to Dataset
ds = xr.open_dataset(nc_file)

# ---
# prep data
# ---

# convert to a DataFrame - dropping missing
df = ds.to_dataframe().dropna().reset_index()

# add columns that will be used as coordinates
# - this could be done with DataLoader.add_col
df['t'] = df['date'].values.astype('datetime64[D]').astype(int)

# ---
# determine the reference locations for a local expert
# ---

# for now just take any location
ref_locs = df.loc[df['lat'] > min_lat, ['x', 'y']].drop_duplicates()
tmp = []
for oid in oi_dates:
    _ = ref_locs.copy()
    _['date'] = np.datetime64(oid)
    tmp += [_]
ref_locs = pd.concat(tmp)

# convert column(s) to be in appropriate coordinate space
# - again could use add_col here instead
ref_locs['t'] = ref_locs['date'].values.astype('datetime64[D]').astype(int)
# ref_locs['x'], ref_locs['y'] = WGS84toEASE2_New(ref_locs['lon'], ref_locs['lat'])


# set multi index of ref_locs
# - this is a bit messy, done so can use index.isin(...) when reading in previous result
tmp_index = ref_locs.set_index(ref_locs.columns.values.tolist())
ref_locs.index = tmp_index.index

# ---
# for given reference location
# ---

# prev_mindx = pd.DataFrame(columns=ref_locs.columns)

# read existing / previous results
try:
    with pd.HDFStore(store_path, mode='r') as store:
        # print(store.keys())
        # levels = store.get_storer("run_details").levels
        # prev_res = store.select_column('run_details', column=levels[0])
        try:
            prev_res = store.select('run_details', columns=[])
            keep_bool = ~ref_locs.index.isin(prev_res.index)
            print(f"using: {keep_bool.sum()} / {len(keep_bool)} reference locations - some were already found")
            ref_locs = ref_locs.loc[~ref_locs.index.isin(prev_res.index)]
        except Exception as e:
            print(e)
except Exception as e:
    print(e)

count = 0

store_dict = {}
for idx, rl in ref_locs.iterrows():


    # pd.MultiIndex.from_frame(rl.to_frame().T)

    t0 = time.time()

    # select local data
    df_local = DataLoader.local_data_select(df,
                                            reference_location=rl,
                                            local_select=local_select,
                                            verbose=False)

    # if len(df_local) < 400:
    #     continue
    print("-" * 30)
    print(f"number obs: {len(df_local)}")

    # -----
    # build model - provide with data
    # -----

    # initialise model
    gpr_model = GPflowGPRModel(data=df_local,
                               obs_col=obs_col,
                               coords_col=coords_col,
                               coords_scale=coords_scale)

    # --
    # set length scale constraints
    # --

    low = np.zeros(len(coords_col))
    high = np.array([2 * incl_rad, 2 * incl_rad, days_ahead + days_behind + 1])

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
    # - remove y to avoid conflict with coordindates
    pred.pop('y')

    # remove * from names - causes issues when saving to hdf5 (?)
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

    # TODO: it's maybe a bit slow to save each time - do differently

    # store to tables - defined by keys in save_dict
    # - n-d arrays will have (default) dimensions added
    # DataLoader.store_to_hdf_table_w_multiindex(idx_dict=rl,
    #                                            out_path=store_path,
    #                                            **save_dict)

    tmp = DataLoader.make_multiindex_df(idx_dict=rl, **save_dict)

    if len(store_dict) == 0:
        store_dict = {k: [v] for k, v in tmp.items()}
        num_store = 1
    else:
        for k, v in tmp.items():
            store_dict[k] += [v]
        num_store = len(store_dict[k])

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

    # count += 1
    # if count > 3:
    #     break
