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
import json
import time

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

# ---
# parameters
# ---

hyper_param_names = ['lengthscales', 'kernel_variance', 'likelihood_variance']

results_dir = get_parent_path("results", "gpod_lead_25km_INVST")
file = f"oi_bin_4_300_post_proc.h5"

store_path = os.path.join(results_dir, file)

# store results in to a 'new' file
# TODO: could just write to store_path - add predictions
# new_store_path = os.path.join(results_dir, re.sub("\.h5$", "_w_pred.h5", file))
new_store_path = store_path

# pred_date = "2020-03-10"

store_every = 10

# overwrite previously generated predictions?
overwrite = False

# ---
# read in previous config
# ---

with pd.HDFStore(store_path, mode='r') as store:
    oi_config = store.get_storer("oi_config").attrs['oi_config']

print(json.dumps(oi_config, indent=4))

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
# - not needed for predictions but info may be useful
raw_data_config = ds.attrs['raw_data_config']
bin_config = ds.attrs['bin_config']

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
# expt_locs = expt_locs.loc[expt_locs['date'] == pred_date, :]

# get previously made predictions - so can skip those
if not overwrite:
    try:
        print("getting previously calculated predictions - so can skip them")
        with pd.HDFStore(new_store_path, mode='r') as store:
            pdf = store.get("preds")

        # remove any (expert) locations which already have predictions
        expt_locs = expt_locs.loc[~expt_locs.index.isin(pdf.index)]
    except KeyError as e:
        print(e)

print("*" * 100)
print(f"expert location to make predictions on: {len(expt_locs)}")

# pre-calculate KDtree to speed up local data selection - this will return a list of len(local_select)
kdtree = DataLoader.kdt_tree_list_for_local_select(df, local_select)

print("there are: {len(expt_locs))} expert locations to generate predictions at")

# store results in a dict, after 'store_every' write to file
store_dict = {}
for idx, rl in expt_locs.iterrows():

    print("-" * 100)
    t0 = time.time()

    # ---
    # select local data
    # ---

    df_local = DataLoader.local_data_select(df,
                                            reference_location=rl,
                                            local_select=local_select,
                                            kdtree=kdtree,
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
    print("setting hyper parameters")
    gpr_model.set_parameters(hyp_dict)

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

    # TODO: The following should be wrapped into a method and stored as
    # run_time = time.time()-t0
    #
    # device_name = gpr_model.cpu_name if gpr_model.gpu_name is None else gpr_model.gpu_name

    # run details / info - for reference
    # run_details = {
    #     "num_obs": len(df_local),
    #     "run_time": run_time,
    #     "device": device_name,
    #     "mll": gpr_model.get_marginal_log_likelihood(),
    #     "optimise_success":  np.nan
    # }

    # store data to specified tables according to key
    # - will add mutli-index based on location
    # save_dict = {
    #     "preds": pd.DataFrame(pred, index=[0]),
    #     "run_details": pd.DataFrame(run_details, index=[0]),
    #     **gpr_model.get_hyperparameters()
    # }\

    # need to pop y because having y in (multi) index as well cause issues (Error)
    # - when reading back in
    pred.pop("y")
    save_dict = {
        "preds": pd.DataFrame(pred, index=[0])
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
                with pd.HDFStore(new_store_path, mode='a') as store:
                    store.append(key=k, value=df_tmp, data_columns=True)
            except Exception as e:
                print(f"Exception occurred storing data for key:{k}\n{e}")
        store_dict = {}

    # t2 = time.time()
    print(f"total run time (including saving): {time.time()-t0:.2f} seconds")


if len(store_dict) > 0:
    print("SAVING LAST RESULTS")
    for k, v in store_dict.items():
        print(k)
        df_tmp = pd.concat(v, axis=0)

        try:
            with pd.HDFStore(new_store_path, mode='a') as store:
                store.append(key=k, value=df_tmp, data_columns=True)
        except Exception as e:
            print(f"Exception occurred storing data for key:{k}\n{e}")
