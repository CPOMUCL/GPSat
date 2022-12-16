# "post" process oi results
# - given a result file, generate a post processed file with smoothed values
# - assumes data is on regularly spaced grid points

import os
import re

import numpy as np
import xarray as xr
import pandas as pd
import itertools

import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from PyOptimalInterpolation.plot_utils import plot_pcolormesh

from functools import reduce
from astropy.convolution import convolve, Gaussian2DKernel
from scipy.spatial import KDTree

from PyOptimalInterpolation.utils import EASE2toWGS84_New
from PyOptimalInterpolation import get_parent_path
from PyOptimalInterpolation.models import GPflowGPRModel
from PyOptimalInterpolation.dataloader import DataLoader
from PyOptimalInterpolation.utils import match


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
pd.set_option("display.max_columns", 200)

# TODO: clipping option should be applied - or handled by constraints in optimisation
# TODO: clean up post_process_oi_irregular.py - it's very messy
# TODO: elements of this should be extract into methods in a GridOI class
#  - namely the

# ---
# helper functions
# ---


# ---
# parameters
# ---

input_dir = get_parent_path("results", "gpod_lead_25km")
input_file = f"oi_bin_4_300.ndf"

output_dir = input_dir
output_file = re.sub("\.ndf$", "_post_proc.ndf", input_file)

# prevent accidental over writing
assert output_file != input_file, f"output and input files can't be the same"


input_path = os.path.join(input_dir, input_file)
output_path = os.path.join(output_dir, output_file)

# std - should be equal to grid_res
# - get from config if not specified
# - should be same units as distance columns
# std = 2 * 25000
std = 2

# distance columns
dist_col = ['x', 'y']

# tables to smooth
smooth_tables = ['lengthscales', 'kernel_variance', 'likelihood_variance']


# ---
# read in previous config
# ---

with pd.HDFStore(input_path, mode='r') as store:
    oi_config = store.get_storer("oi_config").attrs['oi_config']


# extract needed components
# - review these
local_select = oi_config['local_select']
obs_col = oi_config['input_data']['obs_col']
coords_col = oi_config['input_data']['coords_col']
model_params = oi_config.get("model_params", {})

# ---
# read in previous (global) data
# ---

# NOTE: this is not needed
# TODO: reading in input_data should be done by a method in GridOI and handle many different input file types

input_data_file = oi_config['input_data']['file_path']

# connect to Dataset
ds = xr.open_dataset(input_data_file)

# get the configuration(s) use to generate dataset
raw_data_config = ds.attrs['raw_data_config']
input_data_config = ds.attrs['bin_config']


replace_dims = {k: ds.coords[k].values for k in ['x', 'y']}


# ---
# locations for predictions / to have smoothed hyper parameters (masks)
# ---

# TODO: the following section should be a methods
# expert_locations = oi_config["local_expert_locations"]
# HARDCODED: should get values from config
expert_locations = {
    "loc_dims": {
        "x": "x",
        "y": "y",
        "date": ["2020-03-05"]
    },
    "masks": ["had_obs", {"grid_space": 2, "dims": ['x', 'y']}]
}

# expert location masks
el_masks = expert_locations.get("masks", [])
# - remove masks that introduce spacing
masks = []
for m in el_masks:
    if isinstance(m, dict):
        if "grid_space" in m:
            continue
    print(m)
    masks += [m]

# dimensions for the local expert
# - more (columns) can be added with col_func_dict
loc_dims = expert_locations['loc_dims']

masks = DataLoader.get_masks_for_expert_loc(ref_data=ds, el_masks=masks, obs_col=obs_col)

# combine masks only
mask = reduce(lambda x, y: x&y, masks)
mask.name = "include"


# ---
# read in tables to smooth
# ---

dim_col = ['y', 'x', 'date']

org_data = {}
where = None
with pd.HDFStore(input_path, mode='r') as store:
    for k in smooth_tables:
        print(f"reading table: {k}")
        org_data[k] = store.select(k, where=where)


# ---
# store in DataArray
# ---

raw_arrays = {}
for k, v in org_data.items():
    print(f"{k}: converting to DataArray")
    _ = v.copy(True)
    extra_dim_col = [c for c in _.columns if c != k]
    _.reset_index(inplace=True)

    # TODO: make the following a method in DataLoader
    # --
    # store in DataArray
    # --

    # is there a cleaner way of doing this?

    # unique columns for the dimension values
    ucol = {c: np.sort(_[c].unique()) for c in dim_col + extra_dim_col}

    # replace some dimension values - namely to expand
    for kk, vv in replace_dims.items():
        ucol[kk] = vv

    # get the location of each element
    idx_loc = {c: match(_[c].values, ucol[c]) for c in ucol.keys()}

    # create an empty array
    new_shape = tuple([len(v) for k, v in ucol.items()])
    data = np.full(new_shape, np.nan)

    # populate the array
    tmp = tuple([v for k, v in idx_loc.items()])
    data[tmp] = _[k].values

    # store as DataArray
    raw_arrays[k] = xr.DataArray(data, coords=ucol, dims=list(ucol.keys()))

# raw_arrays["lengthscales"].sel(date="2020-03-05").max()

# ---
# apply kernel smoothing across x,y
# ---

# select each 2-d array in data

smooth_dims = ['y', 'x']

smooth_arrays = {}
for k, v in raw_arrays.items():
    print("-" * 10)
    print(f"{k}: applying smoothing")
    # get the coordinate values
    other_coords = {kk: vv.values for kk, vv in v.coords.items() if kk not in smooth_dims}

    # get all the combinations of other coordinate values
    all_combs = list(itertools.product(*[vv for kk, vv in other_coords.items()]))
    all_combs = pd.DataFrame(all_combs, columns=other_coords.keys())

    #
    res = []
    for idx, row in all_combs.iterrows():
        # select data
        row_dict = row.to_dict()
        tmp = v.sel(row_dict)

        # apply convolution using a (2d) gaussian kernel
        _ = convolve(tmp.data, Gaussian2DKernel(x_stddev=std, y_stddev=std),
                     fill_value=np.nan)
        da = xr.DataArray(_, coords=tmp.coords, dims=tmp.dims, name=k)

        # convert single value coordindates to dims
        res += [da.expand_dims(list(row_dict.keys()))]

    smooth_arrays[k] = xr.merge(res)[k]

    smooth_arrays[k] = smooth_arrays[k].where(mask, np.nan)


# ----
# store smooth data in DataFrame
# ----

# want to have the same multi index (names) as input data
df_tables = {}

add_cols = {
    "t": {"func": "lambda x: x.astype('datetime64[D]').astype(int)", "col_args": "date"}
}

for k, v in smooth_arrays.items():

    df = v.to_dataframe().reset_index()
    df.dropna(axis=0, inplace=True)

    DataLoader.add_cols(df, add_cols)

    # get the index names from the original data
    idx_names = org_data[k].index.names
    df.set_index(idx_names, inplace=True)

    # keep only the original columns
    org_cols = org_data[k].columns

    with pd.HDFStore(output_path, mode="a") as store:
        store.put(key=k, value=df[org_cols], append=False)

# ---
# store config information
# ---

# TODO: include "smooth" config
with pd.HDFStore(output_path, mode='a') as store:
    _ = pd.DataFrame({"oi_config": ["use get_storer('oi_config').attrs['oi_config'] to get oi_config"]},
                     index=[0])
    store.put(key="oi_config", value=_)
    store.get_storer("oi_config").attrs['oi_config'] = oi_config
    # store.get_storer("oi_config").attrs["raw_data_config"] = raw_data_config
    store.get_storer("oi_config").attrs['input_data_config'] = input_data_config


# ----
# plot results to sense check
# ----

date = chk_date = "2020-03-05"

for k, v in smooth_arrays.items():

    v_raw = raw_arrays[k]

    x, y = v.coords['x'].values, v.coords['y'].values
    x_grid, y_grid = np.meshgrid(x, y)
    lon_grid, lat_grid = EASE2toWGS84_New(x_grid, y_grid)

    plt_smth = v.sel(date=date).data
    plt_raw = v_raw.sel(date=date).data

    if len(plt_smth.shape) > 2:
        print(f"skipping:  {k}")
        continue

    fig = plt.figure(figsize=(20, 10))
    fig.suptitle(k, fontsize=10, y=0.99)

    # --
    # raw values
    # --

    # first plot: heat map of observations
    ax = fig.add_subplot(1, 2, 1,
                         projection=ccrs.NorthPolarStereo())

    plot_pcolormesh(ax=ax,
                    lon=lon_grid,
                    lat=lat_grid,
                    plot_data=plt_raw,
                    fig=fig,
                    title=k,
                    # vmin=vmin,
                    # vmax=vmax,
                    cmap='YlGnBu_r',
                    # cbar_label=cbar_labels[midx],
                    scatter=False,
                    extent=[-180, 180, 50, 90])

    # ---
    # smoothed values
    # ---

    # first plot: heat map of observations
    ax = fig.add_subplot(1, 2, 2,
                         projection=ccrs.NorthPolarStereo())

    plot_pcolormesh(ax=ax,
                    lon=lon_grid,
                    lat=lat_grid,
                    plot_data=plt_smth,
                    fig=fig,
                    title=k,
                    # vmin=vmin,
                    # vmax=vmax,
                    cmap='YlGnBu_r',
                    # cbar_label=cbar_labels[midx],
                    scatter=False,
                    extent=[-180, 180, 50, 90])

    plt.tight_layout()
    plt.show()



# --
# investigate
# --



# check orginal data
k = 'kernel_variance'
df = org_data[k].copy(True).reset_index()
df = df.loc[df['date'] == chk_date]
df['lon'], df['lat'] = EASE2toWGS84_New(df['x'], df['y'])

fig = plt.figure(figsize=(10, 10))
fig.suptitle(k, fontsize=10, y=0.99)

# first plot: heat map of observations
ax = fig.add_subplot(1, 1, 1,
                     projection=ccrs.NorthPolarStereo())

plot_pcolormesh(ax=ax,
                lon=df['lon'],
                lat=df['lat'],
                plot_data=df['kernel_variance'],
                fig=fig,
                title='input obs.',
                vmin=0,
                vmax=0.1,
                cmap='YlGnBu_r',
                # cbar_label=cbar_labels[midx],
                scatter=True,
                extent=[-180, 180, 50, 90],
                s=5)

plt.show()

# --
# view raw arrays - RAW DATA NEEDS TO BE TRANSPOSED!?
# --

da = raw_arrays[k].sel(date=chk_date)

x, y = da.coords['x'].values, da.coords['y'].values

x_grid, y_grid = np.meshgrid(x, y)
lon_grid, lat_grid = EASE2toWGS84_New(x_grid, y_grid)

# plt.imshow(da.data.T, interpolation="nearest")
# plt.show()


fig = plt.figure(figsize=(10, 10))
fig.suptitle(k, fontsize=10, y=0.99)

# first plot: heat map of observations
ax = fig.add_subplot(1, 1, 1,
                     projection=ccrs.NorthPolarStereo())

plot_pcolormesh(ax=ax,
                lon=lon_grid,
                lat=lat_grid,
                plot_data=da.data,
                fig=fig,
                title=k,
                vmin=0,
                vmax=0.1,
                cmap='YlGnBu_r',
                # cbar_label=cbar_labels[midx],
                scatter=False,
                extent=[-180, 180, 50, 90],
                s=10)

plt.show()
