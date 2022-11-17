# read in raw data, apply selection criteria, plot results / generate summary table

import os

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import cartopy.crs as ccrs

from PyOptimalInterpolation import get_parent_path
from PyOptimalInterpolation.utils import config_func, stats_on_vals #, WGS84toEASE2_New, date_from_datetime
from PyOptimalInterpolation.plot_utils import plot_pcolormesh, plot_hist

pd.set_option('display.max_columns', 200)

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
pltdf = df.loc[select, :]

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
