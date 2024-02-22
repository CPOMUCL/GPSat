# smooth values from results table, store in separate files - to be used for generating predictions late
import json
import re

import pandas as pd
import numpy as np
import numba as nb

import matplotlib.pyplot as plt

from astropy.convolution import convolve, Gaussian2DKernel

from GPSat.local_experts import get_results_from_h5file

from GPSat.utils import json_serializable, cprint
from GPSat.utils import EASE2toWGS84, dataframe_to_2d_array, nested_dict_literal_eval
from GPSat.plot_utils import plot_pcolormesh, get_projection
from GPSat import get_parent_path, get_data_path
from GPSat.models.gpflow_models import GPflowGPRModel
from GPSat.decorators import timer

# TODO: tidy up this script!

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


# @timer
@nb.guvectorize([(nb.float64[:], nb.float64[:], nb.float64[:], nb.float64[:],
                  nb.float64[:], nb.float64[:], nb.float64[:], nb.float64[:]),
                 (nb.float32[:], nb.float32[:], nb.float32[:], nb.float32[:],
                  nb.float32[:], nb.float32[:], nb.float32[:], nb.float32[:])],
                '(), (), (n), (n), (), (), (n)->()',
                nopython=True, target='parallel')
def gaussian_2d_weight(x0, y0, x, y, l_x, l_y, vals, out):
    """weight functions of the form exp(-d^2), where d is the distance between reference position
    (x0, y0) and the others"""

    # calculate the squared distance from the reference equation (normalising dist in each dimension by a length_scale)
    # - can they be specified with defaults?
    d2 = ((x-x0)/l_x[0]) ** 2 + ((y - y0)/l_y[0])**2

    # get the weight function (un-normalised)
    w = np.exp(-d2/2)

    # get the weighted sum of vals, skipping vals which are nan
    w_sum = 0
    w_val = 0
    for i in range(len(vals)):
        if ~np.isnan(vals[i]):
            w_val += w[i] * vals[i]
            w_sum += w[i]

    # if all weights are zero, i.e. in the case all nan vals, return np.nan
    if w_sum == 0:
        out[0] = np.nan
    # otherwise return the normalised weighted value
    else:
        out[0] = w_val / w_sum

# ----
# parameters
# ----

# list of all hyper-parameters to fetch from results - all need not be smooth
all_hyper_params = ["lengthscales", "likelihood_variance", "kernel_variance"]

# hyper parameters to copy (no smoothing applied)
# TODO: ideally would like to get other parameters from model param_names properpty
# - but that would require knowing and initialising the model
# copy_hyper_params = ["inducing_points"]
copy_hyper_params = []

# results file
# store_path = get_parent_path("results", "synthetic", "ABC_baseline.h5")
# store_path = get_parent_path("results", "xval", "cs2cpom_elev_lead_binned_xval_25x25km.h5")
# store_path = get_parent_path("results", "GPFGPR_cs2s3cpom_2019-2020_25km.h5")
# store_path = get_parent_path("results", "elev", "GPOD_elev_lead_binned_25x25km_rerun_BKUP.h5")
# store_path = get_parent_path("results", "xval", "cs2cpom_lead_binned_date_2019_2020_25x25km.h5")
# store_path = get_parent_path("results", "SGPR_vs_GPR_cs2s3cpom_2019-2020_25km.h5")
# store_path = get_parent_path("results", "cs2s3cpom_spaced_local_experts.h5")
# store_path = get_parent_path("results", "SGPR_gpod_lead_elev_10x10km.h5")
# store_path = get_parent_path("results", "SGPR_gpod_freeboard_10x10km.h5")
# store_path = get_parent_path("results", "CSAO", "XVAL_SAR_A_binned_by_track_25x25.h5")
# store_path = get_parent_path("results", "XVAL_gpod_freeboard_10x10km.h5")
store_path = get_parent_path("results", "example", "ABC_50km_for_xval.h5")


# projection only used for plotting
pole = 'north'
# pole = 'south'

# used to identify tables to smooth
reference_table_suffix = "_GPR"
# reference_table_suffix = "_SGPR"

# new table suffix - will be contacted to reference_table_suffix
# table_suffix = "_SMOOTHED_GPR"
table_suffix = "_SMOOTHED"

# reference table_suffix
assert table_suffix != reference_table_suffix

all_hyper_params_w_suf = [f"{_}{reference_table_suffix}" for _ in all_hyper_params]
copy_hyper_params_w_suf = [f"{_}{reference_table_suffix}" for _ in copy_hyper_params]


# output config file
out_config = re.sub("\.h5$", f"{reference_table_suffix}{table_suffix}.json", store_path)

# new prediction locations? set to None if
new_pred_loc = None
# new_pred_loc = {
#     "method": "from_dataframe",
#     "df_file": get_data_path("locations", "2d_xy_grid_5x5km.csv"),
#     "max_dist": 200000
# }

# store smoothed results in separate file
# out_file = re.sub("\.h5$", "_SMOOTHED.h5", store_path)
out_file = store_path

# method / function to use
# use_method = "smooth_2d"
use_method = "gaussian_2d_weight"

# hyper parameters to smooth
# - specify the parameters to smooth with keys, and smoothing parameters as values
# smooth_dict = {
#     "lengthscales": {
#         "stddev": 1
#     },
#     "likelihood_variance": {
#         "stddev": 1
#     },
#     "kernel_variance": {
#         "stddev": 1,
#         "max": 0.1
#     }
# }

# smoothing dict to use with use_method = "gaussian_2d_weight"
smooth_dict = {
    "lengthscales": {
        "l_x": 200_000,
        "l_y": 200_000,
        "max": 12
    },
    "likelihood_variance": {
        "l_x": 200_000,
        "l_y": 200_000,
        "max": 0.3
    },
    "kernel_variance": {
        "l_x": 200_000,
        "l_y": 200_000,
        "max": 0.1
    }
}

# add the reference table suffix
smooth_dict = {f"{k}{reference_table_suffix}": v for k, v in smooth_dict.items()}

# determine the dimensions to smooth over, will be used to make a 2d array
x_col, y_col = 'x', 'y'

# plot values?
plot_values = True

# additional parameters to be passed to dataframe_to_2d_array
to_2d_array_params = {}


# ----
# read in all hyper parameters
# ----

for chp in copy_hyper_params:
    assert chp not in all_hyper_params

select_tables = all_hyper_params + ["expert_locs", "oi_config"] + copy_hyper_params

dfs, oi_configs = get_results_from_h5file(store_path,
                                          global_col_funcs=None,
                                          merge_on_expert_locations=False,
                                          select_tables=select_tables,
                                          table_suffix=reference_table_suffix,
                                          add_suffix_to_table=True)

# coords_col: used as an multi-index in hyperparameter tables
coords_col = oi_configs[-1]['data']['coords_col']

# -----
# (optionally) smooth hyper parameters
# -----

out = {}

for hp_idx, hp in enumerate(all_hyper_params_w_suf):
    # if current hyper parameter is specified in the smooth dict
    if hp in smooth_dict:

        df = dfs[hp].copy(True)
        if use_method == "smooth_2d":
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
            val_col = all_hyper_params[hp_idx]
            if use_method == "smooth_2d":
                # convert dataframe to 2d array - this expects x_col, y_cols to be regularly spaced!
                val2d, x_grid, y_grid = dataframe_to_2d_array(_,
                                                              val_col=val_col,
                                                              x_col=x_col,
                                                              y_col=y_col,
                                                              **to_2d_array_params)

                # apply smoothing (includes nan masking - make optional?)
                smth_2d = smooth_2d(val2d, **smooth_params)


                tmp = pd.DataFrame({
                    hp: smth_2d.flatten(),
                    x_col: x_grid.flatten(),
                    y_col: y_grid.flatten()})

            # experimental smoothing method
            elif use_method == "gaussian_2d_weight":

                # converting to 2d to be able to plot
                # val2d, x_grid, y_grid = dataframe_to_2d_array(_, val_col=hp, x_col=x_col, y_col=y_col,
                #                                               **to_2d_array_params)

                x0, y0 = [_[c].values for c in [x_col, y_col]]
                x, y = [_[c].values for c in [x_col, y_col]]
                vals = _[val_col].values

                if 'max' in smooth_params:
                    vals[vals > smooth_params['max']] = smooth_params['max']

                if 'min' in smooth_params:
                    vals[vals < smooth_params['min']] = smooth_params['min']

                l_x, l_y = smooth_params.get("l_x", 1), smooth_params.get("l_y", 1)

                tmp = gaussian_2d_weight(x0, y0, x, y, l_x, l_y, vals)
                # replace the val_col with the smoothed values
                # _[f"{val_col}_smooth"] = tmp
                _[val_col] = tmp

                # create a new tmp dataframe with just val, x, y cols - other dimes to be added
                tmp = _[[val_col, x_col, y_col]].copy(True)

                if plot_values:

                    figsize = (15, 15)

                    fig, ax = plt.subplots(figsize=figsize,
                                           subplot_kw={'projection': get_projection(pole)})
                    extent = [-180, 180, -90, -60] if pole == "south" else  [-180, 180, 60, 90]

                    # TODO: change this, shouldn't be so hard coded
                    if pole == "north":
                        _['lon'], _['lat'] = EASE2toWGS84(_[x_col], _[y_col])
                    else:
                        _['lon'], _['lat'] = EASE2toWGS84(_[x_col], _[y_col], lat_0=-90, lon_0=0)

                    plot_pcolormesh(ax,
                                    lon=_['lon'],
                                    lat=_['lat'],
                                    plot_data=_[val_col],
                                    extent=extent,
                                    scatter=True,
                                    s=200,
                                    fig=fig,
                                    cbar_label=f"{hp}_smooth\n{row_df}",
                                    cmap='YlGnBu_r')

                    plt.tight_layout()
                    plt.show()

            else:
                raise NotImplementedError(f"use_method: {use_method} is not implemented")

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

        out_table = f'{hp}{table_suffix}'
        cprint(f"adding smoothed table: {out_table}", c="OKCYAN")
        out[out_table] = smooth_df

        # being lazy: make a copy of smooth_params used for this out_table
        smooth_dict[out_table] = smooth_params

    # if not smoothing, just take values as is
    else:
        # set index to coords_col
        # out[hp] = dfs[hp].set_index(coords_col)
        pass

# ---
# copy certain hyper parameters
# ---

for chp in copy_hyper_params_w_suf:

    out_table = f'{chp}{table_suffix}'
    try:
        cprint(f"copying table: {chp} to {out_table}", c="OKCYAN")
        out[out_table] = dfs[chp].copy(True)
        out[out_table].set_index(coords_col, inplace=True)
    except KeyError as e:
        cprint(f"{e} not found, skipping", c="FAIL")

# ---
# write results to table
# ---


cprint(f"writing (smoothed) hyper parameters to:\n{out_file}\ntable_suffix:{reference_table_suffix}{table_suffix}", c="OKGREEN")
with pd.HDFStore(out_file, mode="a") as store:
    for k, v in out.items():
        # out_table = f"{k}{table_suffix}"
        cprint(f"writing: {k} to table", c="BOLD")
        # TODO: confirm this will overwrite existing table?
        store.put(k, v, format="table", append=False)

        # store_attrs = store.get_storer(k).attrs
        try:
            store.get_storer(k).attrs['smooth_config'] = smooth_dict[k]
        except KeyError as e:
            org_table = re.sub(f"{table_suffix}$", "", k)
            store.get_storer(k).attrs['smooth_config'] = {"comment": f"no smoothing, copied directly from {org_table}"}

# ---
# write the configs to file

tmp = []
for oic in oi_configs:
    # change, update the run kwargs to not optimise and use the table_suffix
    run_kwargs = oic.get("run_kwargs", {})
    run_kwargs["optimise"] = False
    run_kwargs["table_suffix"] = f"{reference_table_suffix}{table_suffix}"
    run_kwargs["store_path"] = out_file

    # add load_params - load from self
    model = oic["model"]
    model["load_params"] = {
        "file": out_file,
        "table_suffix": f"{reference_table_suffix}{table_suffix}"
    }

    oic["run_kwargs"] = run_kwargs
    oic["model"] = model

    if new_pred_loc is not None:
        oic["pred_loc"] = new_pred_loc

    tmp.append(json_serializable(oic))

cprint(f"writing config (to use to make predictions with smoothed values) to:\n{out_config}", c="OKBLUE")
with open(out_config, "w") as f:
    json.dump(tmp, f, indent=4)

# oi_configs
# # specifying table_suffix



# TODO: here optionally write oi_config to file, with load_params specified
#  - first should add 'run_kwargs' back to config
# from GPSat.utils import json_serializable
# oi_config["model"]["load_params"] = {"file": out_file}
# with open(get_parent_path("configs", "check.json"), "w") as f:
#     json.dump(json_serializable(oi_config), f, indent=4)


