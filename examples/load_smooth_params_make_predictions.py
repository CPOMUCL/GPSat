# THIS WAS BASICALLY JUST COPIED FROM: load_previous_parameters_make_predictions.py

# given results file containing an oi_config
# - read in input data
# - read in hyper-parameters, store as DataArray (?)
# - for a given (local) expert location
# - - select local data
# - - set previously generated hyper parameters
# - validate previous predictions against current


import os
import re

import numpy as np
import xarray as xr
import pandas as pd

from PyOptimalInterpolation import get_parent_path
from PyOptimalInterpolation.models import GPflowGPRModel
from PyOptimalInterpolation.dataloader import DataLoader

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
pd.set_option("display.max_columns", 200)

# TODO: confirm if using a different conda environment (say without a GPU) will get the same results
# TODO: run a full validation / regression to ensure all results are recovered
# TODO:

# ---
# parameters
# ---

hyper_param_names = ['lengthscales', 'kernel_variance', 'likelihood_variance']

results_dir = get_parent_path("results", "gpod_lead_25km")
file = f"oi_bin_4_300_post_proc.ndf"

store_path = os.path.join(results_dir, file)

pred_date = "2020-03-05"

# ---
# read in previous config
# ---

with pd.HDFStore(store_path, mode='r') as store:
    oi_config = store.get_storer("oi_config").attrs['oi_config']

# extract needed components
local_select = oi_config['local_select']
obs_col = oi_config['input_data']['obs_col']
coords_col = oi_config['input_data']['coords_col']
model_params = oi_config.get("model_params", {})

# ---
# read in previous (global) data
# ---


# TODO: reading in input_data should be done by a method in GridOI and handle many different input file types

input_data_file = oi_config['input_data']['file_path']

# connect to Dataset
ds = xr.open_dataset(input_data_file)

# get the configuration(s) use to generate dataset
# raw_data_config = ds.attrs['raw_data_config']
# bin_config = ds.attrs['bin_config']

# ---
# prep data
# ---

# TODO: how input_data is prepped should be specified in config and be done by a method

df = ds.to_dataframe().dropna().reset_index()
df['date'] = df['date'].values.astype('datetime64[D]')
df['t'] = df['date'].values.astype('datetime64[D]').astype(int)

# --
# load smoothed hyper parameter
# ---

# TODO: the following could be done in one step
hyps_df = {}
hyps = {}
# - here could use select with a where condition - need to convert location to compatible where
with pd.HDFStore(store_path, mode='r') as store:
    for k in hyper_param_names:
        hyps_df[k] = store.get(k)
        hyps[k] = DataLoader.mindex_df_to_mindex_dataarray(df=hyps_df[k].copy(True), data_name=k)

# ---
# get the locations to generate predictions
# ---

# HARDCODED BIT! - should be able to pick any, just drop duplicates?
midx = hyps_df['kernel_variance'].index
expt_locs = pd.DataFrame(index=midx).reset_index()
expt_locs.index = midx

# ---
# get local data (for local expert)
# ---

# select subset of locations - for one date
expt_locs = expt_locs.loc[expt_locs['date'] == pred_date, :]

for idx, rl in expt_locs.iterrows():

    # ---
    # select local data
    # ---
    df_local = DataLoader.local_data_select(df,
                                            reference_location=rl,
                                            local_select=local_select,
                                            verbose=False)
    print(f"number obs: {len(df_local)}")

    # ---
    # initialise GP model
    # ---

    gpr_model = GPflowGPRModel(data=df_local,
                               obs_col=obs_col,
                               coords_col=coords_col,
                               **model_params)

    # ---
    # set hyper parameters
    # ---

    # select for current location
    hyp_dict = {}
    for k, v in hyps.items():
        hyp_dict[k] = v.sel(index=idx).values

    # set hyper parameters
    gpr_model.set_hyperparameters(hyp_dict)

    # ----
    # make prediction
    # ----

    pred = gpr_model.predict(coords=rl)

    # rename prediction values - as done before...
    for k, v in pred.items():
        if re.search("\*", k):
            pred[re.sub("\*", "s", k)] = pred.pop(k)

    # ---
    # store results
    # ----
