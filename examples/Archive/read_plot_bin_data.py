
import os
import re
import json

import pandas as pd
import numpy as np
import xarray as xr

import matplotlib.pyplot as plt
import cartopy.crs as ccrs

from IPython.display import display
from GPSat.dataloader import DataLoader
from GPSat import get_data_path, get_parent_path
from GPSat.utils import WGS84toEASE2, EASE2toWGS84, stats_on_vals, config_func
from GPSat.plot_utils import plot_pcolormesh, plot_hist


pd.set_option("display.max_columns", 200)

# TODO: be more explicit for bind data parameters - store as attribute

# --
# helper function
# --

def _quick_check(df, val_col, lon_col='lon', lat_col='lat', s=2):
    import matplotlib.pyplot as plt
    import cartopy.crs as ccrs

    figsize = (10, 5)
    fig = plt.figure(figsize=figsize)

    ax = fig.add_subplot(1, 2, 1,
                         projection=ccrs.NorthPolarStereo())

    plot_pcolormesh(ax=ax,
                    lon=df[lon_col].values,
                    lat=df[lat_col].values,
                    plot_data=df[val_col].values,
                    fig=fig,
                    # title=plt_title,
                    # vmin=vmin,
                    # vmax=vmax,
                    cmap='YlGnBu_r',
                    # cbar_label=cbar_labels[midx],
                    scatter=True,
                    s=s)

    plt.show()


# ---
# parameters
# ---

# TODO: change ndf_file to input_file - put in config
# ndf file to read from
# hdf_file = get_data_path("RAW", "sats_ra_cry_processed_arco.h5")
# hdf_file = get_data_path("RAW", "sats_ra_cry_processed_arco.h5")
# hdf_file = get_data_path("RAW", "gpod_ocean_lead.h5")
# zarr file (directory?)

# netCDF file to write to (for binned data)
# ncdf_file = get_data_path("RAW", "sats_ra_cry_processed_arco.nc")
# ncdf_file = get_data_path("binned", "gpod_ocean_lead.nc")


verbose = 3

# -
# plot parameters
# -

lon_col = "lon"
lat_col = "lat"
# val_col specified in binning parameters
scatter_plot_size = 2

image_dir = get_parent_path("images")
os.makedirs(image_dir, exist_ok=True)


# -
# binning parameters
# -

val_col = "elev_mss"

use_type = 1
type_name = {1: "lead", 2: "fb", 3: "ocean"}
grid_size = 25

bin_config = {
    "input_file":  get_data_path("RAW", "gpod_ocean_lead.h5"),
    "output_file": get_data_path("binned", f"gpod_{type_name[use_type]}_{val_col}_{grid_size}km_2018_2020.zarr"),
    "plot_file_prefix": f"{type_name[use_type]}_only_",
    "grid_res": grid_size * 1000,
    "bin_by": ['date'],
    "table": "data",
    "val_col": val_col,
    "select": [
        # {"col": val_col, "comp": ">=", "val": -20},
        # {"col": val_col, "comp": "<=", "val": 30},
        {"col": "elev_mss", "comp": ">=", "val": -2},
        {"col": "elev_mss", "comp": "<=", "val": 2},
        {"col": "type", "comp": "==", "val": use_type} # 1 - lead, 3 - ocean
    ],
    "x_col": "x",
    "y_col": "y",
    "x_range": [-4500000.0, 4500000.0],
    "y_range": [-4500000.0, 4500000.0],
    "col_funcs": {
        "date": {
            "func": "lambda x: x.astype('datetime64[D]')",
            "col_args": "datetime"
        },
        "x": {
            "source": "GPSat.utils",
            "func": "WGS84toEASE2_New",
            "col_kwargs": {"lon": "lon", "lat": "lat"},
            "kwargs": {"return_vals": "x"}
        },
        "y": {
            "source": "GPSat.utils",
            "func": "WGS84toEASE2_New",
            "col_kwargs": {"lon": "lon", "lat": "lat"},
            "kwargs": {"return_vals": "y"}
        }
    }
}


input_file = bin_config['input_file']
plot_file_prefix = bin_config.get("plot_file_prefix", "")
base_plot_name = plot_file_prefix + re.sub("\.", "", os.path.basename(input_file))

output_file = bin_config['output_file']

# convert 'datetime' to date
# plt_df['date'] = plt_df['datetime'].values.astype('datetime64[D]')
# plt_df['x'], plt_df['y'] = WGS84toEASE2_New(plt_df['lon'], plt_df['lat'])

table = bin_config['table']
val_col = bin_config['val_col']

# data to select for plotting (raw data) - use same data for binning
plt_where = bin_config['select']

# grid resolution - for binning - in km
grid_res = bin_config['grid_res']

# column to select data to bin by, e.g. ['sat', 'date']
bin_by = bin_config['bin_by']

x_col = bin_config['x_col']
y_col = bin_config['y_col']

x_range = bin_config['x_range']
y_range = bin_config['y_range']

col_funcs = bin_config.get("col_funcs", {})



# plt_where = [
#     {"col": "elev_mss", "comp": ">=", "val": -1},
#     {"col": "elev_mss", "comp": "<=", "val": 1},
#     # selecting tighter number of dates just for plotting - millions of points are slow to render
#     # {"col": "datetime", "comp": ">=", "val": "2020-03-10"},
#     # {"col": "datetime", "comp": "<=", "val": "2020-03-20"},
#     # {"col": "type", "comp": "==", "val": 1},
# ]

# --
# read hdf5
# --

print("reading from hdf5 files")
# read by specifying file path
# where = ["lat<=70.0", "lat>65"]
where = None
with pd.HDFStore(input_file, mode='r') as store:
    df = store.select(table, where=where)

# quick check - plot sub-set
# _quick_check(df.head(100000), val_col)


# get the configuration
try:
    with pd.HDFStore(input_file, mode='r') as store:
        raw_data_config = store.get_storer(table).attrs['config']
    print(json.dumps(raw_data_config, indent=4))
except Exception as e:
    print(e)
    print("issue getting raw_data_config? it should exists in attrs")
    raw_data_config = None

# ---
# stats on all data
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


plt_df = DataLoader.data_select(df, where=plt_where)

plt_stats_df = stats_on_vals(vals=plt_df[val_col].values, name=val_col,
                             qs=[0.01, 0.05] + np.arange(0.1, 1.0, 0.1).tolist() + [0.95, 0.99])

display(plt_stats_df)

# ---
# plot raw data
# ---

figsize = (10, 5)
fig = plt.figure(figsize=figsize)

# randomly select a subset
# WILL THIS SAVE TIME ON PLOTTING?
if len(plt_df) > 1e6:
    len_df = len(plt_df)
    p = 1e6/len_df
    # print(p)
    # b = np.random.binomial(len_df, p=p)
    b = np.random.uniform(0, 1, len_df)
    b = b <= p
    print(f"there were too many points {len(df)}\n"
          f"selecting {100 * b.mean():.2f}% ({b.sum()}) points at random for raw data plot")
    _ = plt_df.loc[b, :]
else:
    _ = plt_df


# figure title
where_print = ", ".join([" ".join([str(v) for k, v in pw.items()]) for pw in plt_where])
# put data source in here?
sup_title = f"val_col: {val_col} - type: {type_name[use_type]}\n" \
            f"min datetime {str(plt_df['datetime'].min())}, " \
            f"max datetime: {str(plt_df['datetime'].max())} \n" \
            f"where conditions:\n" + where_print
fig.suptitle(sup_title, fontsize=10)

nrows, ncols = 1, 2

print("plotting pcolormesh...")
# first plot: heat map of observations
ax = fig.add_subplot(1, 2, 1,
                     projection=ccrs.NorthPolarStereo())

plot_pcolormesh(ax=ax,
                lon=_[lon_col].values,
                lat=_[lat_col].values,
                plot_data=_[val_col].values,
                fig=fig,
                # title=plt_title,
                # vmin=vmin,
                # vmax=vmax,
                cmap='YlGnBu_r',
                # cbar_label=cbar_labels[midx],
                scatter=True,
                s=scatter_plot_size)

ax = fig.add_subplot(1, 2, 2)

print("plotting hist (using all data)...")
plot_hist(ax=ax,
          data=_[val_col].values,#plt_df[val_col].values,
          ylabel="",
          stats_values=['mean', 'std', 'skew', 'kurtosis', 'min', 'max', 'num obs'],
          title=f"{val_col}",
          xlabel=val_col,
          stats_loc=(0.2, 0.8))

plt.tight_layout()

plt_file = os.path.join(image_dir, f"{base_plot_name}_raw_data.pdf")
print(f"saving plot to file:\n{plt_file}")
plt.savefig(plt_file)
plt.show()

cc = 1
# ---
# bin data
# ---

# add columns
for new_col, col_fun in col_funcs.items():

    # add new column
    if verbose >= 3:
        print(f"adding new_col: {new_col}")
    plt_df[new_col] = config_func(df=plt_df,
                                  **col_fun)


# get a Dataset of binned data
ds_bin = DataLoader.bin_data_by(df=plt_df,
                                by_cols=bin_by,
                                val_col=val_col,
                                x_col=x_col,
                                y_col=y_col,
                                grid_res=grid_res,
                                x_range=x_range,
                                y_range=y_range)

# add lon,lat grid values to coords
x_grid, y_grid = np.meshgrid(ds_bin.coords[x_col], ds_bin.coords[y_col])
lon_grid, lat_grid = EASE2toWGS84(x_grid, y_grid)

ds_bin = ds_bin.assign_coords({"lon": ([y_col, x_col], lon_grid),
                               "lat": ([y_col, x_col], lat_grid)})

# add attributes - so know how data was created
# NOTE: can't save netcdf file with nested dict as attributes...
ds_bin.attrs['raw_data_config'] = raw_data_config
ds_bin.attrs['bin_config'] = bin_config

# write to file - mode = 'w' will overwrite file (?)
# DataLoader.write_to_netcdf(ds=ds_bin, path=ncdf_file, mode="w")


if re.search("\.zarr$", output_file, re.IGNORECASE):
    ds_bin.to_zarr(output_file, mode="w")
else:
    assert False, f"output_file: {output_file}\n NOT HANDLED"

# ds_bin = xr.open_zarr(output_file)

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
            f"min datetime {str(plt_df['date'].min())}, " \
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

plt_file = os.path.join(image_dir, f"{base_plot_name}_binned.pdf")
print(f"saving plot to file:\n{plt_file}")
plt.savefig(plt_file)
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

plt_file = os.path.join(image_dir, f"{base_plot_name}_binned_time_ave_grid.pdf")
print(f"saving plot to file:\n{plt_file}")
plt.savefig(plt_file)
plt.show()
