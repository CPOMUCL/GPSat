#%%
# smooth values from results table, store in separate files - to be used for generating predictions late
import json
import re

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from astropy.convolution import convolve, Gaussian2DKernel

from PyOptimalInterpolation.utils import EASE2toWGS84_New, dataframe_to_2d_array, nested_dict_literal_eval
from PyOptimalInterpolation.plot_utils import plot_pcolormesh, get_projection
from PyOptimalInterpolation import get_parent_path, get_data_path
from PyOptimalInterpolation.models.gpflow_models import GPflowGPRModel


# ---
# helper function
# ---

def smooth_2d(x, stddev=None, x_stddev=None, y_stddev=None,
              min=None, max=None):

    # TODO: make nan masking optional?
    if x_stddev is None:
        x_stddev = stddev
    if y_stddev is None:
        y_stddev = stddev
    assert (x_stddev is not None) & (y_stddev is not None), \
        f"x_stddev is: {x_stddev}, y_stddev is: {y_stddev} , they both can't be None"

    nan_mask = np.isnan(x)

    # apply clipping and trimming
    if min is not None:
        min_mask = x < min
        x[min_mask] = min
    if max is not None:
        max_mask = x > max
        x[max_mask] = max

    # convert 0s to NaNs
    x[x==0] = np.nan

    # smooth out lengthscales
    out = convolve(x, Gaussian2DKernel(x_stddev=x_stddev, y_stddev=y_stddev))
    out[nan_mask] = np.nan

    return out

# ----
# parameters
# ----

# results file
# store_path = get_parent_path("results", "synthetic", "ABC_baseline.h5")
store_path = get_parent_path("results", "xval", "cs2cpom_elev_lead_binned_xval_25x25km.h5")

# list of all hyper-parameters to fetch from results - all need not be smooth
all_hyper_params = ["lengthscales", "likelihood_variance", "kernel_variance"]

# store smoothed results in separate file
out_file = re.sub("\.h5$", "_SMOOTHED.h5", store_path)

# hyper parameters to smooth
# - specify the parameters to smooth with keys, and smoothing parameters as values
smooth_dict = {
    "lengthscales": {
        "stddev": 1
    },
    "likelihood_variance": {
        "stddev": 1
    },
    "kernel_variance": {
        "stddev": 1,
        "max": 0.1
    }
}

# determine the dimensions to smooth over, will be used to make a 2d array
x_col, y_col = 'x', 'y'

# plot values?
plot_values = True

# additional parameters to be passed to dataframe_to_2d_array
to_2d_array_params = {}

# used only for plotting, and if x_col, y_col = 'x', 'y'
EASE2toWGS84_New_params = {}
# projection only used for plotting, same conditions above
projection = 'north'

# ----
# read in all hyper parameters
# ----

# read in all the hyper parameter tables
where = None
with pd.HDFStore(store_path, mode="r") as store:
    # TODO: determine if it's faster to use select_colum - does not have where condition?

    all_keys = store.keys()
    dfs = {re.sub("/", "", k): store.select(k, where=where).reset_index()
           for k in all_hyper_params}

    # NOTE: in the future configs maybe store as row entries in the 'oi_config' table
    try:
        oi_config = store.get_storer("oi_config").attrs['oi_config']
    except KeyError as e:
        # this is a bit of a HACK, the "oi_config" table should have 'oi_config' as an attribute
        oi_configs = store.get("oi_config")
        config_df = oi_configs[['config']].drop_duplicates()
        oi_configs = [nested_dict_literal_eval(json.loads(c)) for c in config_df['config'].values]
        oi_config = oi_configs[-1]

# check all keys in smooth_dict are in all_hyper_params
for k in smooth_dict.keys():
    assert k in all_hyper_params

# coords_col: used as an multi-index in hyperparameter tables
coords_col = oi_config['data']['coords_col']

# -----
# (optionally) smooth hyper parameters
# -----
#%%
out = {}

for hp in all_hyper_params:
    # if current hyper parameter is specified in the smooth dict
    if hp in smooth_dict:
        df = dfs[hp].copy(True)
        df[hp] = df[hp].fillna(0) # Replace NaNs with zero
        df_org_col_order = df.columns.values.tolist()
        # smoothing params
        smooth_params = smooth_dict[hp]
        # get the other (None smoothing) dimensions, to iterate over
        other_dims = [c for c in coords_col if c not in [x_col, y_col]]
        # add the other "_dim_*" columns
        dim_cols = [c for c in df.columns if re.search("^_dim_\d", c)]
        other_dims += dim_cols
        # get the unique combinations of other_dims, used to select subset of data
        unique_odims = df[other_dims].drop_duplicates()

        # increment over the rows -want to get a DataFrame representation of each row
        smooth_list = []
        for idx, row in unique_odims.iterrows():
            # get the row as a DataFrame
            row_df = row.to_frame().T

            # and merge on the other dim columns
            _ = row_df.merge(df,
                             on=other_dims,
                             how='inner')

            # convert dataframe to 2d array - this expects x_col, y_cols to be regularly spaced!
            val2d, x_grid, y_grid = dataframe_to_2d_array(_, val_col=hp, x_col=x_col, y_col=y_col, **to_2d_array_params)

            # apply smoothing (includes nan masking - make optional?)
            smth_2d = smooth_2d(val2d, **smooth_params)

            # TODO: optionally plot here?
            #  - show the original and smoothed side by side, could show on a map
            if plot_values:

                fig = plt.figure(figsize=(15, 8))
                row_str = ", ".join([f"{k}: {v}" for k,v in row.items()])
                smooth_str = ", ".join([f"{k}: {v}" for k,v in smooth_params.items()])
                fig.suptitle(f"hyper-parameter: {hp}\nselecting: {row_str}\nsmooth_params: {smooth_str}")

                # TODO: could replace this with plot_pcolormesh
                if (x_col == 'x') & (y_col == 'y'):
                    lon, lat = EASE2toWGS84_New(x_grid, y_grid, **EASE2toWGS84_New_params)

                    ax = plt.subplot(1, 2, 1, projection=get_projection(projection))

                    # make sure both plots have same vmin/max
                    vmax = np.max([np.nanquantile(_, q=0.99) for _ in [smth_2d, val2d] ])
                    vmin = np.min([np.nanquantile(_, q=0.01) for _ in [smth_2d, val2d]])

                    plot_pcolormesh(ax=ax,
                                    lon=lon,
                                    lat=lat,
                                    plot_data=val2d,
                                    title="original",
                                    fig=fig,
                                    vmin=vmin,
                                    vmax=vmax)

                    ax = plt.subplot(1, 2, 2, projection=get_projection(projection))
                    plot_pcolormesh(ax=ax,
                                    lon=lon,
                                    lat=lat,
                                    plot_data=smth_2d,
                                    title="smoothed",
                                    fig=fig,
                                    vmin=vmin,
                                    vmax=vmax)

                else:
                    ax = plt.subplot(1, 2, 1)
                    ax.imshow(val2d)
                    ax.set_title("original")

                    ax = plt.subplot(1, 2, 2)
                    ax.imshow(smth_2d)
                    ax.set_title("smoothed")

                plt.tight_layout()
                plt.show()

            # put values back in dataframe
            tmp = pd.DataFrame({
                hp: smth_2d.flatten(),
                x_col: x_grid.flatten(),
                y_col: y_grid.flatten()})

            # drop nans
            tmp.dropna(inplace=True)

            # add in the 'other dimension' values
            for od in other_dims:
                tmp[od] = row[od]

            # re-order columns to previous order: strictly not needed
            tmp = tmp[df_org_col_order]

            smooth_list.append(tmp)

        smooth_df = pd.concat(smooth_list)

        # set index to be coordinates column
        smooth_df.set_index(coords_col, inplace=True)

        out[hp] = smooth_df

    # if not smoothing, just take values as is
    else:
        # set index to coords_col
        out[hp] = dfs[hp].set_index(coords_col)

# ---
# write results to table
# ---
# %%
# NOTE: this will overwrite an existing file!
print(f"writing (smoothed) hyper parameters to:\n{out_file}")
with pd.HDFStore(out_file, mode="w") as store:
    for k, v in out.items():
        store.append(k, v)

# TODO: here optionally write oi_config to file, with load_params specified
#  - first should add 'run_kwargs' back to config
# from PyOptimalInterpolation.utils import json_serializable
# oi_config["model"]["load_params"] = {"file": out_file}
# with open(get_parent_path("configs", "check.json"), "w") as f:
#     json.dump(json_serializable(oi_config), f, indent=4)


# %%
