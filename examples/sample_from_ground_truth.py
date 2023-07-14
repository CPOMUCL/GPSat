# create a synthetic dataset where observations are taken from some ground truth
# - optionally with noise added

import os
import re
import json
import warnings

import numpy as np
import pandas as pd

from scipy.spatial import KDTree

from GPSat import get_data_path, get_parent_path
from GPSat.utils import json_serializable, nested_dict_literal_eval, get_config_from_sysargv
from GPSat.dataloader import DataLoader



# import matplotlib.pyplot as plt
# import cartopy.crs as ccrs
# from GPSat.plot_utils import plot_pcolormesh

# for land / ocean mask - requires
#  pip install global-land-mask
# from global_land_mask import globe

# TODO:


pd.set_option('display.max_columns', 200)

# -----
# config
# -----

# config = get_config_from_sysargv()
config = get_config_from_sysargv()

# assert config is not None, f"config not provide"
if config is None:
    config_file = get_parent_path("configs", "example_sample_from_ground_truth.json")

    warnings.warn(f"\nconfig is empty / not provided, will just use an example config:\n{config_file}")

    with open(config_file, "r") as f:
        config = nested_dict_literal_eval(json.load(f))

    # override the defaults
    config['ground_truth']['source'] = get_data_path("MSS", "CryosatMSS-arco-2yr-140821_with_geoid_h.csv")
    config['observations']['source'] = get_data_path("example", "ABC.h5")
    config['output']['dir'] = get_data_path("example", "synthetic")
    config['output']['file'] = "synthetic_data_from_ground_truth_ABC.h5"

    # check inputs source exists
    assert os.path.exists( config['ground_truth']['source']), \
        f" config['ground_truth']['source']:\n{ config['ground_truth']['source']}\ndoes not exists. "

    assert os.path.exists(config['observations']['source']), \
        f"config['observations']['source']:\n{config['observations']['source']}\ndoes not exists. " \
        f"to create run: python -m GPSat.read_and_store"

# -------
# output directory + file
# -------

# write sampled obs to:
out_dir = config["output"]["dir"]
out_file = os.path.join(out_dir, config["output"]["file"])
out_table = config.get("table", "data")

os.makedirs(out_dir, exist_ok=True)

# ---
# ground truth file
# ---

# ground truth file - not stored in package - expect the file to contain ['mss', 'h', 'lon', 'lat'] -
gt_config = config["ground_truth"]

# remove certain keys - review these
gt_obs_cols = gt_config.pop("obs_col")

gt = DataLoader.load(**gt_config)

# require 'x', 'y' (coordinate) columns exist
assert np.in1d(['x', 'y'], gt.columns).all(), f"gt.columns: {gt.columns} missing 'x' and/or 'y'"

# ---
# observations
# ---

obs_config = config['observations']

obs = DataLoader.load(**obs_config)

# require 'x', 'y' (coordinate) columns exist
assert np.in1d(['x', 'y'], obs.columns).all(), f"obs.columns: {obs.columns} missing 'x' and/or 'y'"

# ----
# create a KDTree using ground truth locations
# ----

# the nearest ground truth value will be used for an observations
kdt = KDTree(gt.loc[:, ['x', 'y']].values)

# ---
# find the nearest ground truth value for each observation
# ---

data_config = config['data_generation']

obs_col = data_config.pop("new_obs_col", "obs")
add_noise = data_config.get("add_noise", 0.0)

# let the observations be the nearest locations
dist, ind = kdt.query(obs[['x', 'y']].values, k=1)

# update the obs column to be gt_obs_cols value from the nearest location from the mss data
obs[obs_col] = gt.iloc[ind, :][gt_obs_cols].values

# mean of the observation
if data_config.get("add_mean_col", False):
    obs[f"{obs_col}_mean"] = obs[obs_col].mean()

# mean the observations? requires the mean be added
if data_config.get("demean_obs", False):
    assert f"{obs_col}_mean" in obs, f"'{obs_col}_mean' is missing , set in config set " \
                                     f"'add_mean_col'=True, or set demean_obs=False"
    obs[obs_col] = obs[obs_col] - obs[f"{obs_col}_mean"]

# noise observations
if add_noise:
    # print("asdf")
    obs[f"{obs_col}_w_noise"] = obs[obs_col] + np.random.normal(loc=0, scale=add_noise, size=len(obs))


# add any additional columns
DataLoader.add_cols(df=obs, col_func_dict=data_config['col_funcs'])

# select a subset of columns (optional)
col_select = data_config.get("col_select", None)
if col_select is not None:
    obs = obs.loc[:, col_select]
    # obs = DataLoader.data_select(obj=obs,
    #                              copy=False,
    #                              columns=data_config.get("col_select", None))

# -----
# write to file
# -----

print(f"writing to:\n{out_file}")
if re.search("\.csv", out_file, re.IGNORECASE):
    obs.to_csv(out_file,
                     index=False)
elif re.search('\.h5', out_file, re.IGNORECASE):
    with pd.HDFStore(out_file, mode='w') as store:
        store.append(key=out_table,
                     value=obs,
                     index=False,
                     append=False,
                     data_columns=True)
        storer = store.get_storer(out_table)
        storer.attrs['config'] = config
else:
    file_type = re.sub('^.*\.', '', out_file)
    raise ValueError(f"output file suffix not handled: {file_type}")

print('finished')


# ---
# create a gridded product - for plotting
# ---
#
# # from GPSat.utils import grid_2d_flatten
#
# x_range = [-4500000.0, 4500000.0]
# y_range = [-4500000.0, 4500000.0]
# grid_res = 12.5 * 1000
#
# x_min, x_max = x_range[0], x_range[1]
# y_min, y_max = y_range[0], y_range[1]
#
# # number of bin (edges)
# n_x = ((x_max - x_min) / grid_res) + 1
# n_y = ((y_max - y_min) / grid_res) + 1
# n_x, n_y = int(n_x), int(n_y)
#
# # NOTE: x will be dim 1, y will be dim 0
# x_edge = np.linspace(x_min, x_max, int(n_x))
# y_edge = np.linspace(y_min, y_max, int(n_y))
#
# x_grid, y_grid = np.meshgrid(x_edge, y_edge)
# lon_grid, lat_grid = EASE2toWGS84_New(x_grid, y_grid)
#
# X = np.concatenate([x_grid.flatten()[:, None], y_grid.flatten()[:, None]], axis=1)
#
# # get the nearest values
# nearest_dist, nearest_ind = kdt.query(X, k=1)
#
# # show how to recover 2d array from 1d array
# assert np.all(X[:, 0].reshape(x_grid.shape) == x_grid)
# assert np.all(X[:, 1].reshape(y_grid.shape) == y_grid)
#
# -----
# plot gridded product of ground truth
# -----
#
# plot_data = df.iloc[nearest_ind, :]['z'].values.reshape(x_grid.shape)
#
# # this just loads data
# is_in_ocean = globe.is_ocean(lat_grid, lon_grid)
# plot_data[~is_in_ocean] = np.nan
#
# figsize = (15, 15)
#
# fig, ax = plt.subplots(figsize=figsize,
#                        subplot_kw={'projection': ccrs.NorthPolarStereo()})
#
#
# # print(f"there are now: {len(od)} expert locations")
# vmin = np.nanquantile(plot_data, q=0.05)
# vmax = np.nanquantile(plot_data, q=0.95)
#
# # data needed for plotting
# lon, lat = lon_grid, lat_grid
#
# # TODO: remove points too far from original data - i.e. recover the pole hole
# print("plotting results")
# plot_pcolormesh(ax,
#                 lon=lon,
#                 lat=lat,
#                 vmin=vmin,
#                 vmax=vmax,
#                 plot_data=plot_data,
#                 scatter=False,
#                 s=200,
#                 fig=fig,
#                 cbar_label="Ground Truth",
#                 cmap='YlGnBu_r')
#
# plt.show()


