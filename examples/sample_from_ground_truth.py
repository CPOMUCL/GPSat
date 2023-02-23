
import numpy as np
import pandas as pd

from scipy.spatial import KDTree

from PyOptimalInterpolation import get_data_path
from PyOptimalInterpolation.utils import WGS84toEASE2_New, EASE2toWGS84_New

import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from PyOptimalInterpolation.plot_utils import plot_pcolormesh

# for land / ocean mask - requires
#  pip install global-land-mask
from global_land_mask import globe

pd.set_option('display.max_columns', 200)

# create ground truth function from gridded product using nearest point (with KDTree
mss_file = get_data_path("MSS", "CryosatMSS-arco-2yr-140821.txt")

df = pd.read_csv(mss_file, header=None, sep="\s+", names=['lon', 'lat', 'z'])
df['x'], df['y'] = WGS84toEASE2_New(df['lon'], df['lat'])

kdt = KDTree(df.loc[:, ['x', 'y']].values)

# ---
# create a gridded product - for plotting
# ---

x_range = [-4500000.0, 4500000.0]
y_range = [-4500000.0, 4500000.0]
grid_res = 12.5 * 1000

x_min, x_max = x_range[0], x_range[1]
y_min, y_max = y_range[0], y_range[1]

# number of bin (edges)
n_x = ((x_max - x_min) / grid_res) + 1
n_y = ((y_max - y_min) / grid_res) + 1
n_x, n_y = int(n_x), int(n_y)

# NOTE: x will be dim 1, y will be dim 0
x_edge = np.linspace(x_min, x_max, int(n_x))
y_edge = np.linspace(y_min, y_max, int(n_y))

x_grid, y_grid = np.meshgrid(x_edge, y_edge)
lon_grid, lat_grid = EASE2toWGS84_New(x_grid, y_grid)

X = np.concatenate([x_grid.flatten()[:, None], y_grid.flatten()[:, None]], axis=1)

# get the nearest values
nearest_dist, nearest_ind = kdt.query(X, k=1)

# show how to recover 2d array from 1d array
assert np.all(X[:, 0].reshape(x_grid.shape) == x_grid)
assert np.all(X[:, 1].reshape(y_grid.shape) == y_grid)

plot_data = df.iloc[nearest_ind, :]['z'].values.reshape(x_grid.shape)


# this just loads data
is_in_ocean = globe.is_ocean(lat_grid, lon_grid)
plot_data[~is_in_ocean] = np.nan

figsize = (15, 15)

fig, ax = plt.subplots(figsize=figsize,
                       subplot_kw={'projection': ccrs.NorthPolarStereo()})


# print(f"there are now: {len(od)} expert locations")

# data needed for plotting
lon, lat = lon_grid, lat_grid

plot_pcolormesh(ax,
                lon=lon,
                lat=lat,
                # vmin=vmin,
                # vmax=vmax,
                plot_data=plot_data,
                scatter=False,
                s=200,
                fig=fig,
                cbar_label="Num Obs within Local Select of Expert Location",
                cmap='YlGnBu_r')

plt.show()

# ----
# sample from data
# ----

# a long track locations
store = pd.HDFStore(get_data_path("example", "ABC.h5"), mode='r')
track_loc = store.get("data")

track_loc['x'], track_loc['y'] = WGS84toEASE2_New(track_loc['lon'], track_loc['lat'])

# let the observations be the nearest locations
dist, ind = kdt.query(track_loc[['x', 'y']].values, k=1)

# update the obs column to be z value from the nearest location from the mss data
track_loc['obs'] = df.iloc[ind, :]["z"].values

# add noise here
track_loc['obs_w_noise'] = track_loc['obs'] + np.random.normal(loc=0, scale=0.1, size=len(track_loc))

# write to file
track_loc.to_csv(get_data_path("example", "along_track_sample_from_mss_ground_truth.csv"),
                 index=False)






