import os
import re

import numpy as np
import pandas as pd

from scipy.spatial import KDTree

from PyOptimalInterpolation import get_data_path
from PyOptimalInterpolation.utils import WGS84toEASE2_New, EASE2toWGS84_New
from PyOptimalInterpolation.dataloader import DataLoader


# import matplotlib.pyplot as plt
# import cartopy.crs as ccrs
# from PyOptimalInterpolation.plot_utils import plot_pcolormesh

# for land / ocean mask - requires
#  pip install global-land-mask
# from global_land_mask import globe

# TODO:


pd.set_option('display.max_columns', 200)

# -------
# Parameters
# -------

# write sampled obs to:
# out_file = get_data_path("ground_truth", "along_track_sample_from_mss_ground_GPOD.h5")
out_file = get_data_path("ground_truth", "along_track_sample_from_mss_ground_ABC.h5")

os.makedirs(os.path.dirname(out_file), exist_ok=True)

# noise to add to ground truth for noisy observations
gt_added_noise = 0.1

# ---
# ground truth file
# ---

# ground truth file - not stored in package - expect the file to contain ['mss', 'h', 'lon', 'lat'] -
mss_file = get_data_path("MSS", "CryosatMSS-arco-2yr-140821_with_geoid_h.csv")

# ----
# file containing along track observation data - i.e. the sample locations
# ----

# tracks from GPOD data (more observations than example)
# track_file = "/mnt/m2_red_1tb/Data/GPOD/gpod_all.h5"
# track_table = "data"
# # use fb for sea ice, elev for lean/ocean
# # track_obs_col = "elev"
# track_obs_col = "fb"
# track_where = [
#     {
#         "col": "datetime",
#         "comp": ">=",
#         "val": "2020-02-01"
#     },
#     {
#         "col": "datetime",
#         "comp": "<=",
#         "val": "2020-04-01"
#     },
#     {
#         "col": "type",
#         "comp": "=",
#         # "val": [1, 3] # lead and ocean
#         "val": [2] # sea ice
#     }
# ]

# # tracks from example data
track_file = get_data_path("example", "ABC.h5")
track_table = "data"
track_where = None
# track_where = {
#     "col": "source",
#     "comp": "==",
#     "val": "C"
# }
track_obs_col = "obs"


# create a config file
input_config = {
    "track": {
        "file": track_file,
        "table": track_table,
        "obs_col": track_obs_col,
        "where": track_where
    },
    "add_noise": gt_added_noise,
    "mss_file": mss_file,
    "generated_using_file": "sample_from_ground_truth.py"
}

# ----
# mss ground truth with geoid height for reference
# ----

df = pd.read_csv(mss_file)
# df = pd.read_csv(mss_file, header=None, sep="\s+", names=['lon', 'lat', 'z'])
df['z'] = df['mss'] - df['h']

# create a KDTree to all finding the nearest value (for interpolation)
df['x'], df['y'] = WGS84toEASE2_New(df['lon'], df['lat'])

kdt = KDTree(df.loc[:, ['x', 'y']].values)

# ----
# sample from data
# ----

# a long track locations
store = pd.HDFStore(track_file, mode='r')

# track_loc = store.get("data")
track_loc = DataLoader.data_select(store, table=track_table, where=track_where)
store.close()

# remove nan observations from track
track_loc = track_loc.loc[~np.isnan(track_loc[track_obs_col])]

track_loc['x'], track_loc['y'] = WGS84toEASE2_New(track_loc['lon'], track_loc['lat'])

# let the observations be the nearest locations
dist, ind = kdt.query(track_loc[['x', 'y']].values, k=1)

# update the obs column to be z value from the nearest location from the mss data
track_loc['obs'] = df.iloc[ind, :]["z"].values

# sub track the mean of the obseravtions
# - NOTE: only for above 60 lat
# z_mean = df.loc[df['lat'] > 60, "z"].mean()
z_mean = track_loc['obs'].mean()

track_loc['obs_mean'] = z_mean

# demean observations - to give a prior mean of 0
track_loc['obs'] = track_loc['obs'] - z_mean

# add noise here
track_loc['obs_w_noise'] = track_loc['obs'] + np.random.normal(loc=0, scale=gt_added_noise, size=len(track_loc))

# add time column - get number of seconds since epoch, divide by seconds in day
track_loc['t'] = track_loc['datetime'].values.astype("datetime64[s]").astype(float) / (24 * 60 * 60)

track_loc['date'] = track_loc['datetime'].values.astype("datetime64[D]")

# write to file
print(f"writing to:\n{out_file}")
if re.search("\.csv", out_file, re.IGNORECASE):
    track_loc.to_csv(out_file,
                     index=False)
elif re.search('\.h5', out_file, re.IGNORECASE):
    with pd.HDFStore(out_file, mode='w') as store:
        store.append(key="data",
                     value=track_loc,
                     index=False,
                     append=False,
                     data_columns=True)
        storer = store.get_storer("data")
        storer.attrs['config'] = input_config
else:
    file_type = re.sub('^.*\.', '', out_file)
    raise ValueError(f"output file suffix not handled: {file_type}")

print('finished')


# ---
# create a gridded product - for plotting
# ---
#
# # from PyOptimalInterpolation.utils import grid_2d_flatten
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


