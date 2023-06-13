# simple inline example of OI
# NOTE: there is no smoothing of hyper parameters

import os
import re

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from PyOptimalInterpolation import get_data_path, get_parent_path
from PyOptimalInterpolation.dataprepper import DataPrep
from PyOptimalInterpolation.utils import WGS84toEASE2_New, EASE2toWGS84_New, cprint, grid_2d_flatten, get_weighted_values
from PyOptimalInterpolation.local_experts import LocalExpertOI, get_results_from_h5file
from PyOptimalInterpolation.plot_utils import plot_pcolormesh, get_projection, plot_pcolormesh_from_results_data

# ----
# read in raw data
# ----

# read in all the *_RAW.csv files in data/example

raw_files = [get_data_path("example", i)
             for i in os.listdir(get_data_path("example")) if re.search("_RAW\.csv$", i)]

# read in, add source
tmp = []
for rw in raw_files:
    source = re.sub("_RAW\.csv$", "", os.path.basename(rw))
    _ = pd.read_csv(rw)
    _['source'] = source
    tmp.append(_)
df = pd.concat(tmp)


# convert lon, lat, datetime to x, y, t - to be used as the coordinate space
df['x'], df['y'] = WGS84toEASE2_New(lon=df['lon'], lat=df['lat'], lat_0=90, lon_0=0)
df['t'] = df['datetime'].values.astype("datetime64[D]").astype(float)

# ----
# bin raw data
# ----

# bin by date, source
# - returns a DataSet
bin_ds = DataPrep.bin_data_by(df=df.loc[(df['z'] > -0.35) & (df['z'] < 0.65)],
                              by_cols=['t', 'source'],
                              val_col='z',
                              x_col='x',
                              y_col='y',
                              grid_res=50_000,
                              x_range=[-4500000.0, 4500000.0],
                              y_range=[-4500000.0, 4500000.0])

# convert bin data to DataFrame
# - removing all the nans that would be added at grid locations away from data
bin_df = bin_ds.to_dataframe().dropna().reset_index()

# --
# plot binned data
# --

# this will plot all observations, some on top of each other
bin_df['lon'], bin_df['lat'] = EASE2toWGS84_New(bin_df['x'], bin_df['y'])

fig = plt.figure(figsize=(12, 12))
ax = fig.add_subplot(1, 1, 1, projection=get_projection('north'))

plot_pcolormesh(ax=ax,
                lon=bin_df['lon'],
                lat=bin_df['lat'],
                plot_data=bin_df['z'],
                title="example: binned obs",
                scatter=True,
                s=20,
                fig=fig,
                # vmin=[-]
                extent=[-180, 180, 60, 90])

plt.tight_layout()
plt.show()

# ----
# expert locations - on evenly spaced grid
# ----

xy_grid = grid_2d_flatten(x_range=[-500000.0, 500000.0],
                          y_range=[-500000.0, 500000.0],
                          step_size=200_000)

# store in dataframe
eloc = pd.DataFrame(xy_grid, columns=['x', 'y'])

# add a time coordinate
eloc['t'] = np.floor(df['t'].mean())

# ---
# plot expert locations
# ---

eloc['lon'], eloc['lat'] = EASE2toWGS84_New(eloc['x'], eloc['y'])


fig = plt.figure(figsize=(12, 12))
ax = fig.add_subplot(1, 1, 1, projection=get_projection('north'))

plot_pcolormesh(ax=ax,
                lon=eloc['lon'],
                lat=eloc['lat'],
                plot_data=eloc['t'],
                title="expert locations",
                scatter=True,
                s=20,
                fig=fig,
                # vmin=[-]
                extent=[-180, 180, 60, 90])

plt.tight_layout()
plt.show()


# ----
# prediction locations
# ----

xy_grid = grid_2d_flatten(x_range=[-500000.0, 500000.0],
                          y_range=[-500000.0, 500000.0],
                          step_size=5_000)

# store in dataframe
# NOTE: the missing 't' coordinate will be determine by the expert location
# - alternatively the prediction location can be specified
ploc = pd.DataFrame(xy_grid, columns=['x', 'y'])


# --
# prediction locations
# --


ploc['lon'], ploc['lat'] = EASE2toWGS84_New(ploc['x'], ploc['y'])


fig = plt.figure(figsize=(12, 12))
ax = fig.add_subplot(1, 1, 1, projection=get_projection('north'))

plot_pcolormesh(ax=ax,
                lon=ploc['lon'],
                lat=ploc['lat'],
                plot_data=np.full(len(ploc), 1.0), #np.arange(len(ploc)),
                title="prediction locations",
                scatter=True,
                s=0.1,
                # fig=fig,
                extent=[-180, 180, 60, 90])

plt.tight_layout()
plt.show()


# ----
# configurations:
# ----

# observation data
data = {
    "data_source": bin_df,
    "obs_col": "z",
    "coords_col": ["x", "y", "t"],
    # selection criteria used for each local expert
    "local_select": [
        {
            "col": "t",
            "comp": "<=",
            "val": 4
        },
        {
            "col": "t",
            "comp": ">=",
            "val": -4
        },
        {
            "col": [
                "x",
                "y"
            ],
            "comp": "<",
            "val": 300_000
        }
    ]
}

# local expert locations
local_expert = {
    "source": eloc
}

# model
model = {
    "oi_model": "GPflowGPRModel",
    "init_params": {
        # scale (divide) coordinates
        "coords_scale": [50000, 50000, 1]
    },
    "constraints": {
        # lengthscales - same order coord_col (see data)
        "lengthscales": {
            "low": [1e-08, 1e-08, 1e-08],
            "high": [600000, 600000, 9]
        }
    }
}

# prediction locations
# -
pred_loc = {
    "method": "from_dataframe",
    "df": ploc,
    "max_dist": 200_000
}

# ----
# Local Expert OI
# ----


locexp = LocalExpertOI(expert_loc_config=local_expert,
                       data_config=data,
                       model_config=model,
                       pred_loc_config=pred_loc)

# run optimal interpolation
# - no predictions locations supplied
store_path = get_parent_path("results", "inline_example.h5")

# for the purposes of a simple example, if store_path exists: delete it
if os.path.exists(store_path):
    cprint(f"removing: {store_path}")
    os.remove(store_path)

# run optimal interpolation
locexp.run(store_path=store_path,
           optimise=True)

# ----
# results are store in hdf5
# ----

# extract, store in dict
dfs, _ = get_results_from_h5file(store_path)

print(f"tables in results file: {list(dfs.keys())}")

# ----
# plot the predictions
# ----

plt_data = dfs["preds"]

weighted_values_kwargs = {
        "ref_col": ["pred_loc_x", "pred_loc_y", "pred_loc_t"],
        "dist_to_col": ["x", "y", "t"],
        "val_cols": ["f*", "f*_var"],
        "weight_function": "gaussian",
        "lengthscale": 200_000
    }
plt_data = get_weighted_values(df=plt_data, **weighted_values_kwargs)

plt_data['lon'], plt_data['lat'] = EASE2toWGS84_New(plt_data['pred_loc_x'], plt_data['pred_loc_y'])


fig = plt.figure(figsize=(12, 12))
ax = fig.add_subplot(1, 1, 1, projection=get_projection('north'))
plot_pcolormesh_from_results_data(ax=ax,
                                  dfs={"preds": plt_data},
                                  table='preds',
                                  val_col="f*",
                                  scatter=False,
                                  x_col='pred_loc_x',
                                  y_col='pred_loc_y',
                                  fig=fig)
plt.tight_layout()
plt.show()

