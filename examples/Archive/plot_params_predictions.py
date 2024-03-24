# plot results from OI
# NOTE: the following is not working as desired...
# TODO: tidy up th follow

import os
import re

import pandas as pd
import numpy as np
import xarray as xr
import cartopy.crs as ccrs


from GPSat.utils import match
from GPSat.utils import WGS84toEASE2, EASE2toWGS84, stats_on_vals
from GPSat.plot_utils import plot_pcolormesh, plot_hist
from GPSat.dataloader import DataLoader
from GPSat import get_parent_path

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

pd.set_option("display.max_columns", 200)

# TODO: this needs to be re-factored, should be getting config information from output (h5) file

# ---
# helper functions
# ---

def get_plot_data(df, x, y, lon_grid, lat_grid,
                  val_col,
                  lon_col='lon',
                  lat_col='lat',
                  store_as_2d=False):

    if store_as_2d:
        lon = lon_grid
        lat = lat_grid
        plot_data = np.full(lon_grid.shape, np.nan)

        x_loc = match(df['x'].values, x)
        y_loc = match(df['y'].values, y)
        plot_data[y_loc, x_loc] = df[val_col].values
        scatter = False
    else:
        lon = df[lon_col].values
        lat = df[lat_col].values
        plot_data = df[val_col].values
        scatter = True

    return lon, lat, plot_data, scatter


# ----
# parameters
# ----

# column from table to plot
plot_col = "fs"
plot_table = "preds"

# results directory and file - to plot from
# result_dir = get_parent_path("results", "freeboard")
# result_file = "oi_bin_cs2s3cpom_4_300_freeboard_obs_50km_None.h5"
#
result_dir = get_parent_path("results", "example")
result_file = "ABC_binned_example.h5"

store_path = os.path.join(result_dir, result_file)

# where images will be saved
image_dir = get_parent_path("images", os.path.basename(result_dir))
print(f"storing images in:\n{image_dir}")
os.makedirs(image_dir, exist_ok=True)

# lon,lat column names
lon_col, lat_col = "lon", "lat"

# HARDCODED: vmin/vmax values for certain columns of data
vminmax = {
    "fs": [-0.5, 0.5],
    "likelihood_variance": [0, 0.02],
    "kernel_variance": [0, 0.15]
}

figsize = (20, 20)


# ------------
# read in previous config
# ------------

with pd.HDFStore(store_path, mode='r') as store:
    oi_config = store.get_storer("oi_config").attrs['oi_config']

# coordinate columns
coords_col = oi_config['input_data']['coords_col']

# get the input data
data_source = oi_config['input_data']['file_path']

# # HARDCODED: assumes data_source is zarr/netcdf
# # TODO: make get config function for any storage type (?)
ds = xr.open_dataset(data_source)

# get x,y & lon,lat of input data grid
x, y = ds.coords['x'].values, ds.coords['y'].values

x_grid, y_grid = np.meshgrid(x, y)
lon_grid, lat_grid = EASE2toWGS84(x_grid, y_grid)

# ---
# read in results data
# ---

# where = "date=='2020-03-12'"
where = None

# read in results, store in dict with table aa key
with pd.HDFStore(store_path, mode="r") as store:
    # TODO: determine if it's faster to use select_colum - does not have where condition?

    all_keys = store.keys()
    dfs = {re.sub("/", "", k): store.select(k, where=where).reset_index()
           for k in all_keys}

# modify the tables as needed
for k in dfs.keys():
    _ = dfs[k]
    try:
        _['lon'], _['lat'] = EASE2toWGS84(_['x'], _['y'])
    except KeyError:
        pass
    dfs[k] = _

#data_vars[data_name] = DataLoader.mindex_df_to_mindex_dataarray(df, data_name=data_name)


# HACK:
# TODO: handle nd parameters better - use a select condition?
ls_map = {i:c for i, c in enumerate(coords_col)}
dfs['lengthscales']['lengthscales_dim'] = dfs['lengthscales']['lengthscales_0'].map(ls_map)

ls_df = dfs['lengthscales'].copy(True)
for dim in dfs['lengthscales']['lengthscales_0'].unique():
    tmp = ls_df.loc[ls_df['lengthscales_0'] == dim, :].copy(True)
    tmp.rename(columns={'lengthscales': f"ls_{ls_map[dim]}"}, inplace=True)
    dfs[f"ls_{ls_map[dim]}"] = tmp

# --------------------
# create a map for plots - namely which table to get the data col from
# --------------------

# plt_list = ["fs", "y_var", "ls_y", "ls_x", "ls_t", "kernel_variance", "likelihood_variance"]

# TODO: could include a where or select condition (i.e. using a row_select)
plt_map = {
    "fs": {
        "df": "preds",
        "col": "fs"
    },
    "y_var": {
        "df": "preds",
        "col": "y_var"
    },
    "ls_x": {
        "df": "ls_x",
        # "col": "lengthscales"
        "col": "ls_x"
    },
    "ls_y": {
        "df": "ls_y",
        # "col": "lengthscales",
        "col": "ls_y"
    },
    "ls_t": {
        "df": "ls_t",
        # "col": "lengthscales"
        # "where": [],
        "col": "ls_t"
    },
    "kernel_variance": {
        "df": "kernel_variance",
        "col": "kernel_variance"
    },
    "likelihood_variance": {
        "df": "likelihood_variance",
        "col": "likelihood_variance"
    }
}

# --
# read in raw / binned observation data
# --

# convert to a DataFrame - dropping missing
# df = ds.to_dataframe().dropna().reset_index()
# add columns that will be used as coordinates
# - this could be done with DataLoader.add_col
# df['t'] = df['date'].values.astype('datetime64[D]').astype(int)
# dfs['raw_data'] = df

# ----
# convert multi index DataFrame to DataArrays
# ----

#
# # read in results, store in dict with table aa key
# dfs2 = {}
# with pd.HDFStore(store_path, mode="r") as store:
#     # TODO: determine if it's faster to use select_colum - does not have where condition?
#
#     all_keys = store.keys()
#     dfs2 = {re.sub("/", "", k): store.select(k, where=where)
#            for k in all_keys}
#
#
# # not used - but if prefer to work with DataArrays
# da_dict = {}
# for k, v in plt_map.items():
#     da_dict[k] = DataLoader.mindex_df_to_mindex_dataarray(dfs2[v['df']][[v['col']]].copy(True),
#                                                           data_name=v['col'])


# ---
# plot results
# ---

# dates available
dates = dfs[plot_table]['date'].unique()
print(f"number of dates: {len(dates)}")

# get the plotting details
plt_dict = plt_map[plot_col]


# ---
# plot images: write to pdf and show
# ---

image_file = os.path.join(image_dir, re.sub("\.h5$", f"_{plot_col}.pdf", result_file))

print(f"plotting to file: {os.path.basename(image_file)}")

with PdfPages(image_file) as pdf:
    for date in dates:

        print(f"date: {date.astype('datetime64[D]')}")

        fig, ax = plt.subplots(figsize=figsize,
                               subplot_kw={'projection': ccrs.NorthPolarStereo()})

        # --
        # select data - this could be neater
        # --

        plt_col = plt_dict['col']
        tmp_df = dfs[plt_dict['df']]
        tmp_df = tmp_df.loc[tmp_df['date'] == date]

        # tmp_df = plt_df.loc[plt_df['date'] == date]
        _ = tmp_df[plt_col].values

        # plot range
        vmin, vmax = np.nanquantile(_, [0.01, 0.99])
        # use fixed vmin/vmax is specified
        if plt_col in vminmax:
            vmin, vmax = vminmax[plt_col]

        lon, lat, plot_data, scatter = get_plot_data(tmp_df, x, y, lon_grid, lat_grid,
                                                     val_col=plt_col,
                                                     lon_col='lon',
                                                     lat_col='lat',
                                                     store_as_2d=True)

        # ---
        # draw heat map
        # ---

        plot_pcolormesh(ax,
                        lon=lon,
                        lat=lat,
                        plot_data=plot_data,
                        scatter=scatter,
                        s=5,
                        fig=fig,
                        cbar_label=plot_col,
                        cmap='YlGnBu_r')

        # add title
        stitle = ax.set_title(f"col: {plot_col}, table: {plot_table}\ndate: {date.astype('datetime64[D]')}")

        plt.show()
        pdf.savefig(fig)

        # plt.close()
