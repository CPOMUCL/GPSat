#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import re

from datetime import datetime
from GPSat import get_parent_path, get_data_path
from GPSat.dataloader import DataLoader
from GPSat.utils import WGS84toEASE2, EASE2toWGS84, nll, rmse
from GPSat.postprocessing import glue_local_predictions


# Load xval results
pattern = r"/preds_\d+_\d+"
results_dir = get_parent_path("results")
# results_path = os.path.join(results_dir, "XVAL_ABC_50km_binned_by_tracks.h5")
# results_path = os.path.join(results_dir, "XVAL_ABC_25km_binned_by_tracks.h5")
results_path = os.path.join(results_dir, "XVAL_ABC_10km_binned_by_tracks.h5")
results_dict = {}
with pd.HDFStore(results_path, mode="r") as store:
    keys = [key for key in store.keys() if re.match(pattern, key)]
    for key in keys:
        results_dict[key] = store[key]

#%%
# Load ground truth data
data_dir = get_data_path("example")
# data_path = os.path.join(data_dir, "ABC_50km_binned_by_track.h5")
# data_path = os.path.join(data_dir, "ABC_25km_binned_by_track.h5")
data_path = os.path.join(data_dir, "ABC_10km_binned_by_track.h5")
with pd.HDFStore(data_path) as store:
    data_gt = store['data']

# Include 'pred_loc_x' and 'pred_loc_y' columns
lon, lat = data_gt['lon'], data_gt['lat']
pred_loc_x, pred_loc_y = WGS84toEASE2(lon, lat)
data_gt['pred_loc_x'] = pred_loc_x
data_gt['pred_loc_y'] = pred_loc_y

#%%
inference_radius = 200_000 # TODO: Retrieve properly from run_details.

scores = {}
for key, table in results_dict.items():
    sample = table.reset_index()
    results = glue_local_predictions(sample, inference_radius)
    date = pd.Timestamp(datetime.strptime(key.split("_")[1], "%Y%m%d"))
    track_id = int(key.rsplit("_", 1)[-1])

    data_on_track = data_gt[(data_gt['date'] == date) & (data_gt['track'] == track_id)].copy()
    # TODO: This is a hacky quick fix to get pred_loc coordinates consistent
    data_on_track.pred_loc_x = np.round(data_on_track.pred_loc_x)
    data_on_track.pred_loc_y = np.round(data_on_track.pred_loc_y)

    results_sorted = results.sort_values(['pred_loc_x', 'pred_loc_y'],  ignore_index=True)
    data_on_track = data_on_track.reset_index()
    data_on_track_sorted = data_on_track.sort_values(['pred_loc_x', 'pred_loc_y'],  ignore_index=True)

    results_sorted = pd.merge(results_sorted, data_on_track_sorted[['pred_loc_x', 'pred_loc_y', 'obs']], on=['pred_loc_x', 'pred_loc_y'])

    print("-"*30 + f"\n Track id: {track_id}")
    y = results_sorted['obs']
    mu = results_sorted['f*']
    sigma = np.sqrt(results_sorted['y_var'])
    RMSE = rmse(y, mu)
    NLL = nll(y, mu, sigma, return_tot=False).mean()
    print(f"RMSE: {RMSE}")
    print(f"NLL: {NLL}")

    scores[key] = {'RMSE': RMSE, 'NLL': NLL}


# %%
rmse_list = []
nll_list = []
for key, value in scores.items():
    rmse_list.append(value['RMSE'])
    nll_list.append(value['NLL'])

plt.hist(rmse_list)
print(f"Mean RMSE: {np.mean(rmse_list)}")

# %%
plt.hist(nll_list)
print(f"Mean NLL: {np.mean(nll_list)}")

# %%
