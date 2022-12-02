# read in raw data, apply selection criteria, plot results / generate summary table

import os
import re

import pandas as pd
import numpy as np

import scipy.stats as scst
import matplotlib.pyplot as plt
import cartopy.crs as ccrs

from functools import reduce
from PyOptimalInterpolation import get_parent_path
from PyOptimalInterpolation.utils import config_func, stats_on_vals, match, \
    WGS84toEASE2_New, EASE2toWGS84_New, bin_obs_by_date
from PyOptimalInterpolation.plot_utils import plot_pcolormesh, plot_hist

pd.set_option('display.max_columns', 200)

# ----
# helper functions - move to utils?
# ----


# ---
# parameters / configuration
# ---

# for each 'source' specify one or more files to read in

base_dir = get_parent_path("data", "ocean_elev_gpod_raw_tsv")

file_map = {
    "CS2": [
        os.path.join(base_dir, "CS2_SAR.tsv"),
        os.path.join(base_dir, "CS2_SARIN.tsv")
    ],
    "S3A": os.path.join(base_dir, "S3A.tsv"),
    "S3B": os.path.join(base_dir, "S3B.tsv")
}

# keyword arguments to provide to pd.read_csv
read_csv_kwargs = {
        "sep": "\t"
    }

verbose = 3

# ----
# read in data
# ----

res = []
for source, files in file_map.items():

    if isinstance(files, str):
        files = [files]

    assert isinstance(files, (list, tuple, np.array)), "files expected to be list or tuple"

    for file in files:
        assert os.path.exists(file), f"source: {source}, file: {file} does not exist"
        print(f"{'-' * 10}\nsource: {source}\nfile:{file}")
        _ = pd.read_csv(file, **read_csv_kwargs)
        # HARDCODED: adding a 'source' column
        _['source'] = source

    res += [_]

# concat all results
df = pd.concat(res)

# ---
# add / transform columns
# ---

col_funcs = {
    # convert lon, lat to x,y with EASE2.0 projection
    # "x,y": {
    #     # func can be defined function
    #     "func": WGS84toEASE2_New,
    #     "col_args": ["lon", "lat"],
    #     "out_cols": ["x", "y"]
    # },
    # get date in YYYYMMDD format from 'datetime' column
    "date": {
        # or it can be a string - which eval will act on
        "func": "date_from_datetime",
        "col_kwargs": {"dt": "datetime"}
    },
    # convert datetime to column to datetime64
    "datetime": {
        "func": lambda x: x.astype('datetime64'),
        "col_args": "datetime"
    },
    # elevation minus mean sea surface
    "elev-mss": {
        "func": "-",
        "col_args": ["elev", "mss"]
    }
}

print("*" * 20)

# apply column functions
for new_col, col_fun in col_funcs.items():

    # including out_cols to allow for multiple column output
    # - an alternative could be to parse new_col to get multiple columns
    # - e.g. "x,y" -> ["x", "y"]
    # - NOTE: after popping 'out_cols' will no longer exist in dict!
    new_col = col_fun.pop('out_cols', new_col)

    # add new column
    if verbose >= 3:
        print(f"adding new_col: {new_col}")
        print(f"using: {col_fun}")

    _ = config_func(df=df,
                    **col_fun)
    # allow for multiple column assignment
    if isinstance(new_col, (list, tuple)):
        for nc_idx, nc in enumerate(new_col):
            df[nc] = _[nc_idx]
    else:
        df[new_col] = _


# ---
# (row) selection criteria
# ---

row_select = [
    # dates after "2020-03-01"
    {"func": ">=", "col_args": "datetime", "args": np.datetime64("2020-03-01")},
    # dates before "2020-04-01"
    {"func": "<", "col_args": "datetime", "args": np.datetime64("2020-04-01")}
    # keep only: data["elev-mss"] <= 75
    # {"func": "<=", "col_args": "elev-mss", "args": 75}
]

select = np.ones(len(df), dtype=bool)

print("selecting rows:")
for sl in row_select:
    # print(sl)
    if verbose >= 3:
        print(sl)
    select &= config_func(df=df, **sl)

# select subset of data
if verbose >= 3:
    print(f"selecting {select.sum()}/{len(select)} rows")
df = df.loc[select, :]

# ---
# summary table
# ---

print("*" * 20)
print("summary / stats table on metric (use for trimming)")

val_col = "elev-mss"
vals = df[val_col].values
# vals = vals[(vals < 5) & (vals > -5) ]
stats_df = stats_on_vals(vals=vals, name=val_col,
                         qs=[0.01, 0.05] + np.arange(0.1, 1.0, 0.1).tolist() + [0.95, 0.99])

print(stats_df.T)

# ---
# plot (selected) values
# ---

print("*" * 20)
print("visualise values")

plot_select = [
    # keep only: data["elev-mss"] <= 75
    # {"func": "==", "col_args": "source", "args": "CS2"},
    {"func": "<=", "col_args": "elev-mss", "args": 5},
    {"func": ">=", "col_args": "elev-mss", "args": -5}
]
plt_title = "all sats"
sup_title = "March 2020 Raw Obs"
plot_col = val_col
lon_col, lat_col = 'lon', 'lat'
scatter_plot_size = 2

vmin, vmax = None, None

select = np.ones(len(df), dtype=bool)

print("selecting rows (for plotting)")
for sl in plot_select:
    # print(sl)
    if verbose >= 3:
        print(sl)
    select &= config_func(df=df, **sl)

# select subset of data
if verbose >= 3:
    print(f"selecting {select.sum()}/{len(select)} rows for plot")
pltdf = df.loc[select, :].copy()

# str_date = date.astype('datetime64[D]').astype(str)
# plt_title = str_date

# ax = fig.add_subplot(nrows, ncols, 1, projection=ccrs.NorthPolarStereo())

figsize = (10, 5)
fig = plt.figure(figsize=figsize)
fig.suptitle(sup_title)

nrows, ncols = 1, 2

# first plot: heat map of observations
ax = fig.add_subplot(1, 2, 1,
                     projection=ccrs.NorthPolarStereo())

plot_pcolormesh(ax=ax,
                lon=pltdf[lon_col].values,
                lat=pltdf[lat_col].values,
                plot_data=pltdf[plot_col].values,
                fig=fig,
                title=plt_title,
                vmin=vmin,
                vmax=vmax,
                cmap='YlGnBu_r',
                # cbar_label=cbar_labels[midx],
                scatter=True,
                s=scatter_plot_size)

ax = fig.add_subplot(1, 2, 2)

plot_hist(ax=ax,
          data=pltdf[plot_col].values,
          ylabel="",
          stats_values=['mean', 'std', 'skew', 'kurtosis', 'min', 'max', 'num obs'],
          title=f"elev - mss",
          xlabel="elev - mss (m)",
          stats_loc=(0.2, 0.85))

plt.tight_layout()
plt.show()


# ----
# bin data
# ----

# grid resolution (in km)
grid_res = 50

# HARDCODED
if grid_res == 50:
    bin_shape = (180, 180)
else:
    bin_shape = (360, 360)

# apply binning - using a binning function (add to utils?)
pltdf['x'], pltdf['y'] = WGS84toEASE2_New(pltdf['lon'].values,
                                          pltdf['lat'].values)

bin_src = {}
for src in pltdf['source'].unique():
    print("*" * 20)
    print(src)
    # bin the data - that was plotted
    bin_src[src] = bin_obs_by_date(df=pltdf.loc[pltdf['source'] == src, :].copy(),
                                   val_col='elev-mss',
                                   x_col='x',
                                   y_col='y',
                                   grid_res=grid_res,
                                   date_col='date')

# -----
# combine binned data into a single nd-array
# -----

# get common dates

# {k:len(list(v[0].keys())) for k,v in bin_src.items()}
src_dates = [np.array(list(v[0].keys())) for k, v in bin_src.items()]

# common_dates = reduce(lambda x,y: np.intersect1d(x,y), src_dates)
union_dates = reduce(lambda x, y: np.union1d(x, y), src_dates)

obs_shape = bin_shape + (len(union_dates), len(bin_src))
obs = np.full(obs_shape, np.nan)

for src, src_dates in bin_src.items():
    for date, vals in src_dates[0].items():
        date_loc = match(date, union_dates)[0]
        src_loc = match(src, list(bin_src.keys()))[0]
        obs[:, :, date_loc, src_loc] = vals


# get the average obs
ave_obs = np.nanmean(obs, axis=(2, 3))

# get the x,y values at the center of grid
src = list(bin_src.keys())[0]
x_edge, y_edge = bin_src[src][1], bin_src[src][2]

# get the centers for edges
x_cntr, y_cntr = x_edge[:-1] + np.diff(x_edge) / 2, y_edge[:-1] + np.diff(y_edge) / 2

# TODO: need to use mesh grid
x_grid, y_grid = np.meshgrid(x_cntr, y_cntr)

# convert to lon, lat
lon_grid, lat_grid = EASE2toWGS84_New(x_grid, y_grid)

figsize = (10, 5)
fig = plt.figure(figsize=figsize)
fig.suptitle(sup_title)

nrows, ncols = 1, 2

# first plot: heat map of observations
ax = fig.add_subplot(nrows, ncols, 1,
                     projection=ccrs.NorthPolarStereo())

plot_pcolormesh(ax=ax,
                lon=lon_grid,
                lat=lat_grid,
                plot_data=ave_obs,
                fig=fig,
                title="(averaged) binned obs",
                vmin=vmin,
                vmax=vmax,
                cmap='YlGnBu_r',
                # cbar_label=cbar_labels[midx],
                scatter=False)

ax = fig.add_subplot(nrows, ncols, 2)

plot_hist(ax=ax,
          data=ave_obs[~np.isnan(ave_obs)],
          ylabel="",
          stats_values=['mean', 'std', 'skew', 'kurtosis', 'min', 'max', 'num obs'],
          title=f"elev - mss",
          xlabel="elev - mss (m)",
          stats_loc=(0.2, 0.85))

plt.tight_layout()
plt.show()


plt.show()

#
# figsize = (10, 5)
# fig = plt.figure(figsize=figsize)
# fig.suptitle(sup_title)
#
# nrows, ncols = 1, 1
#
# # first plot: heat map of observations
# ax = fig.add_subplot(nrows, ncols, 1,
#                      projection=ccrs.NorthPolarStereo())
#
# fake_obs = np.arange(np.prod(ave_obs.shape)).reshape(ave_obs.shape)
#
# plot_pcolormesh(ax=ax,
#                 lon=np.round(lon_grid, 2),
#                 lat=np.round(lat_grid, 2),
#                 plot_data=fake_obs,
#                 fig=fig,
#                 title="ave obs",
#                 # vmin=vmin,
#                 # vmax=vmax,
#                 cmap='YlGnBu_r',
#                 # cbar_label=cbar_labels[midx],
#                 scatter=False)
# #
# # ax = fig.add_subplot(1, 2, 2)
# #
# # ax.imshow(ave_obs, interpolation='None')
#
# # plt.tight_layout()
# plt.show()
#
#
#
