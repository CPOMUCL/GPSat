
import os
import re

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs

from IPython.display import display
from PyOptimalInterpolation.dataloader import DataLoader
from PyOptimalInterpolation import get_data_path
from PyOptimalInterpolation.utils import WGS84toEASE2_New, EASE2toWGS84_New, stats_on_vals
from PyOptimalInterpolation.plot_utils import plot_pcolormesh, plot_hist


pd.set_option("display.max_columns", 200)

# ---
# parameters
# ---

# ndf file to read from
hdf_file = get_data_path("RAW", "sats_ra_cry_processed_arco.h5")
# netCDF file to write to (for binned data)
ncdf_file = get_data_path("binned", "sats_ra_cry_processed_arco.nc")

table = "data"
val_col = "elev_mss"
lon_col = "lon"
lat_col = "lat"

scatter_plot_size = 2

# -
# binning parameters
# -

# grid resolution - for binning - in km
grid_res = 50

# column to select data to bin by, e.g. ['sat', 'date']
bin_by = ['date']

# --
# read hdf5
# --

print("reading from hdf5 files")
# read by specifying file path
df = DataLoader.read_hdf(table=table, path=hdf_file)

# ---
# stats on data
# ---

print("*" * 20)
print("summary / stats table on metric (use for trimming)")

vals = df[val_col].values
stats_df = stats_on_vals(vals=vals, name=val_col,
                         qs=[0.01, 0.05] + np.arange(0.1, 1.0, 0.1).tolist() + [0.95, 0.99])

# print(stats_df)
display(stats_df)


# ----
# read / select data
# ----

plt_where = [
    {"col": "elev_mss", "comp": ">=", "val": -1},
    {"col": "elev_mss", "comp": "<=", "val": 1},
    # selecting tighter number of dates just for plotting - millions of points are slow to render
    # {"col": "datetime", "comp": ">=", "val": "2020-03-10"},
    # {"col": "datetime", "comp": "<=", "val": "2020-03-20"},
    # {"col": "type", "comp": "==", "val": 1},
]

plt_df = DataLoader.data_select(df, where=plt_where)

plt_stats_df = stats_on_vals(vals=plt_df[val_col].values, name=val_col,
                             qs=[0.01, 0.05] + np.arange(0.1, 1.0, 0.1).tolist() + [0.95, 0.99])

display(plt_stats_df)

# ---
# plot data
# ---

figsize = (10, 5)
fig = plt.figure(figsize=figsize)

# figure title
where_print = ", ".join([" ".join([str(v) for k, v in pw.items()]) for pw in plt_where])
# put data source in here?
sup_title = f"val_col: {val_col}\n" \
            f"min_datetime {str(plt_df['datetime'].min())}, " \
            f"max datetime: {str(plt_df['datetime'].max())} \n" \
            f"where conditions:" + where_print
fig.suptitle(sup_title, fontsize=10)

nrows, ncols = 1, 2

# first plot: heat map of observations
ax = fig.add_subplot(1, 2, 1,
                     projection=ccrs.NorthPolarStereo())

plot_pcolormesh(ax=ax,
                lon=plt_df[lon_col].values,
                lat=plt_df[lat_col].values,
                plot_data=plt_df[val_col].values,
                fig=fig,
                # title=plt_title,
                # vmin=vmin,
                # vmax=vmax,
                cmap='YlGnBu_r',
                # cbar_label=cbar_labels[midx],
                scatter=True,
                s=scatter_plot_size)

ax = fig.add_subplot(1, 2, 2)

plot_hist(ax=ax,
          data=plt_df[val_col].values,
          ylabel="",
          stats_values=['mean', 'std', 'skew', 'kurtosis', 'min', 'max', 'num obs'],
          title=f"{val_col}",
          xlabel=val_col,
          stats_loc=(0.2, 0.8))

plt.tight_layout()
plt.show()


# ---
# bin data
# ---

# convert 'datetime' to date
plt_df['date'] = plt_df['datetime'].values.astype('datetime64[D]')
plt_df['x'], plt_df['y'] = WGS84toEASE2_New(plt_df['lon'], plt_df['lat'])

# get a Dataset of binned data
ds_bin = DataLoader.bin_data_by(df=plt_df,
                                by_cols=bin_by,
                                val_col=val_col,
                                grid_res=grid_res * 1000,
                                x_range=[-4500000.0, 4500000.0],
                                y_range=[-4500000.0, 4500000.0])

# add lon,lat grid values to coords
x_grid, y_grid = np.meshgrid(ds_bin.coords['x'], ds_bin.coords['y'])
lon_grid, lat_grid = EASE2toWGS84_New(x_grid, y_grid)

ds_bin = ds_bin.assign_coords({"lon": (['y', 'x'], lon_grid),
                               "lat": (['y', 'x'], lat_grid)})

# write to file - mode = 'w' will overwrite file (?)
DataLoader.write_to_netcdf(ds=ds_bin, path=ncdf_file, mode="w")

# ---
# plot binned data
# ---

# plot each point as a scatter plot
# - extract to DataFrame
bin_df = ds_bin.to_dataframe().dropna().reset_index()

plt_df = bin_df

figsize = (10, 5)
fig = plt.figure(figsize=figsize)

# figure title
where_print = ", ".join([" ".join([str(v) for k, v in pw.items()]) for pw in plt_where])
# put data source in here?
sup_title = "Binned data\n"\
            f"val_col: {val_col}\n" \
            f"min_datetime {str(plt_df['date'].min())}, " \
            f"max datetime: {str(plt_df['date'].max())} \n" \
            # f"where conditions:" + where_print
fig.suptitle(sup_title, fontsize=10)

nrows, ncols = 1, 2

# first plot: heat map of observations
ax = fig.add_subplot(1, 2, 1,
                     projection=ccrs.NorthPolarStereo())

plot_pcolormesh(ax=ax,
                lon=plt_df[lon_col].values,
                lat=plt_df[lat_col].values,
                plot_data=plt_df[val_col].values,
                fig=fig,
                # title=plt_title,
                # vmin=vmin,
                # vmax=vmax,
                cmap='YlGnBu_r',
                # cbar_label=cbar_labels[midx],
                scatter=True,
                s=scatter_plot_size)

ax = fig.add_subplot(1, 2, 2)

plot_hist(ax=ax,
          data=plt_df[val_col].values,
          ylabel="",
          stats_values=['mean', 'std', 'skew', 'kurtosis', 'min', 'max', 'num obs'],
          title=f"{val_col}",
          xlabel=val_col,
          stats_loc=(0.2, 0.8))

plt.tight_layout()
plt.show()


# --
# alternative plot - using 2-d array
# --

plt_data = ds_bin[val_col].data
last_axis = [i for i in range(len(plt_data.shape)) if i > 1]
plt_data = np.nanmean(plt_data, axis=tuple(last_axis))

lat_grid = ds_bin.coords['lat'].data
lon_grid = ds_bin.coords['lon'].data

figsize = (10, 5)
fig = plt.figure(figsize=figsize)

# figure title
where_print = ", ".join([" ".join([str(v) for k, v in pw.items()]) for pw in plt_where])
# put data source in here?
sup_title = "Binned data - from 2-d array\n"\
            f"val_col: {val_col}\n" \
            f"min_datetime {ds_bin.coords['date'].min().data.astype('datetime64[D]')}, " \
            f"max datetime: {ds_bin.coords['date'].max().data.astype('datetime64[D]')} \n" \
            # f"where conditions:" + where_print
fig.suptitle(sup_title, fontsize=10)

nrows, ncols = 1, 2

# first plot: heat map of observations
ax = fig.add_subplot(1, 2, 1,
                     projection=ccrs.NorthPolarStereo())

plot_pcolormesh(ax=ax,
                lon=lon_grid,
                lat=lat_grid,
                plot_data=plt_data,
                fig=fig,
                # title=plt_title,
                # vmin=vmin,
                # vmax=vmax,
                cmap='YlGnBu_r',
                # cbar_label=cbar_labels[midx],
                scatter=False,
                s=scatter_plot_size)

ax = fig.add_subplot(1, 2, 2)

plot_hist(ax=ax,
          data=plt_data[~np.isnan(plt_data)],
          ylabel="",
          stats_values=['mean', 'std', 'skew', 'kurtosis', 'min', 'max', 'num obs'],
          title=f"{val_col}",
          xlabel=val_col,
          stats_loc=(0.2, 0.8))

plt.tight_layout()
plt.show()
