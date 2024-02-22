# analysis the results for xval holding out a single track
# - use regex to find all prediction tables
# - - get weighted combination of the predictions made for each location
# - get the prediction locations by way of the config (identified)

import re
import json

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

from GPSat import get_parent_path
from GPSat.utils import nested_dict_literal_eval, get_weighted_values, nll
from GPSat.utils import EASE2toWGS84
from GPSat.dataloader import DataLoader
from GPSat.plot_utils import plot_hist, plot_pcolormesh, get_projection

pd.set_option("display.max_columns", 200)

# ---
# parameters
# ---

results_file = get_parent_path("results", "XVAL_gpod_freeboard_10x10km.h5")

# match strings like: _20191203_240
# regex = r'^_[0-9]{8}_[0-9]{1,5}$'
regex = r'^_GPR_SMOOTHED_[0-9]{8}_[0-9]{1,5}$'
# regex = r'^_SGPR_SMOOTHED_[0-9]{8}_[0-9]{1,5}$'

# weight function - for combining predictions
weighted_values_kwargs = {
        "ref_col": ["pred_loc_x", "pred_loc_y", "pred_loc_t"],
        "dist_to_col": ["x", "y", "t"],
        "val_cols": ["f*", "f*_var", "y_var", "config_id"],
        "weight_function": "gaussian",
        "lengthscale": 200_000
    }


# ---
# find the table suffixes for the xval results
# ---

with pd.HDFStore(results_file, mode='r') as store:
    all_tables = [re.sub("/", "", k) for k in store.keys()]

# get all the table suffixes
base_table = "run_details"
all_suffixes = [re.sub(f"^{base_table}", "", i) for i in all_tables if re.search(f"^{base_table}", i)]

xval_sufs = [i for i in all_suffixes if re.search(regex, i)]

print(f"found: {len(xval_sufs)} sets of results / tables ")

# --
# read in all the prediction tables
# --

all_xpreds = []

for suf in xval_sufs:

    print(f"suf: {suf}")

    with pd.HDFStore(results_file, mode='r') as store:
        preds = store.select(f"preds{suf}")
        rd = store.select(f"run_details{suf}")
        oic = store.select(f"oi_config{suf}")

    # get the config_ids for each local expert (x,y,t)
    expert_cols = list(rd.index.names)
    expert_cid = rd.reset_index()[expert_cols + ["config_id"]].drop_duplicates()

    # get the configs: key is id
    conf_dic = {}
    for row, _ in oic.iterrows():
        conf_dic[_['idx']] = oi_config = nested_dict_literal_eval(json.loads(_["config"]))

    # for each config id, load the prediction locations (from file)
    # NOTE: expects prediction locations come from_source and have load_kwargs
    pred_locs = {}
    for cid in expert_cid['config_id'].unique():

        c = conf_dic[cid]
        assert c['pred_loc']['method'] == "from_source", "method expected to be from_source"
        assert "load_kwargs" in c['pred_loc']

        pred_locs[cid] = DataLoader.load(**c['pred_loc']['load_kwargs'])

    # for each prediction location, merge on the config id
    preds.reset_index(inplace=True)
    preds = preds.merge(expert_cid, on=expert_cols, how='left')

    # get the weighted combination of predictions
    # NOTE: this expects each prediction location to be generated using the same config_id
    wpreds = get_weighted_values(df=preds, **weighted_values_kwargs)

    # ---
    # merge on the predictions with values at the prediction locations
    # ---

    pred_loc_cols = [f"pred_loc_{_}" for _ in expert_cols]

    tmp = []
    # NOTE: here config_id will be floats, will keys in pred_locs will be ints(?) - best not to mix them
    for wi in wpreds['config_id'].unique():
        pl = pred_locs[wi]

        _ = pl.merge(wpreds,
                     left_on=expert_cols,
                     right_on=pred_loc_cols,
                     how='inner')

        obs_col = conf_dic[wi]["data"]["obs_col"]

        _['diff'] = _[obs_col] - _["f*"]
        # normalise by the difference
        _['norm_diff'] = _['diff'] / np.sqrt(_['y_var'])
        # TODO: add a nll value
        _['nll'] = nll(y=_[obs_col], mu=_["f*"], sig=np.sqrt(_['y_var']), return_tot=False)

        tmp.append(_)

    xval_preds = pd.concat(tmp)

    all_xpreds.append(xval_preds)


xp = pd.concat(all_xpreds)

# ---
# visualise / analyse results
# ---

# HARDCODED: add lon, lat
xp['lon'], xp['lat'] = EASE2toWGS84(xp['pred_loc_x'], xp['pred_loc_y'])

df = xp.copy(True)

# quick plot

# plot_col = 'norm_diff'
plot_col = 'nll'

if plot_col == "norm_diff":
    qvmin, qvmax = None, None
    cmap = 'bwr'
    vmax = np.nanquantile(np.abs(df[plot_col]), q=0.99)
    vmin = -vmax
else:
    qvmin, qvmax = 0.05, 0.95
    vmin, vmax = None, None
    cmap = 'YlGnBu_r'


# df = df.loc[df[plot_col] < -1000]
# df = df.loc[df['track'] != 89]

projection = get_projection("north")

fig = plt.figure(figsize=(18, 9))
ax = fig.add_subplot(1, 2, 1, projection=projection)

plot_pcolormesh(ax=ax, lon=df['lon'], lat=df['lat'],
                plot_data=df[plot_col],
                fig=fig,
                s=3,
                qvmin=qvmin,
                qvmax=qvmax,
                vmin=vmin,
                vmax=vmax,
                title=f"regex: {regex}",
                scatter=True,
                cmap=cmap)

ax = fig.add_subplot(1, 2, 2)
plot_hist(ax,
          data=df[plot_col].values,
          q_vminmax=np.array([0.01, 0.99]),
          stats_values=['mean', 'std', 'skew', 'kurtosis', 'min', 'max', 'num obs'])

# heat map
# fig = plt.figure(figsize=(12, 12))
plt.tight_layout()
plt.show()
