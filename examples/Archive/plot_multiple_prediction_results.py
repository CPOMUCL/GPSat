# plot predictions from multiple local experts, with possibly overlapping predictions
# NOTE: this code initially copied from plot_params_predictions
# TODO: this script needs to be tidied up and refactored

import os
import re
import warnings

import pandas as pd
import numpy as np
import xarray as xr
import cartopy.crs as ccrs

from functools import reduce

from scipy.spatial import KDTree

# for land / ocean mask - requires
#  pip install global-land-mask
from global_land_mask import globe

from GPSat.utils import match
from GPSat.utils import WGS84toEASE2, EASE2toWGS84, stats_on_vals
from GPSat.plot_utils import plot_pcolormesh, plot_hist
from GPSat.dataloader import DataLoader
from GPSat import get_parent_path, get_data_path
from GPSat.local_experts import LocalExpertData


from GPSat.utils import WGS84toEASE2

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

pd.set_option("display.max_columns", 200)

# TODO: this needs to be re-factored, should be getting config information from output (h5) file
# TODO: review the plot_dict to make sure it's fit for purpose

# ----
# helper functions
# ----


def nll(y, mu, sig, return_tot=True):
    # negative log likelihood assuming independent normal observations (y)
    out = np.log(sig * np.sqrt(2 * np.pi)) + (y - mu)**2 / (2 * sig**2)
    if return_tot:
        return np.sum(out[~np.isnan(out)])
    else:
        return out


def plot_helper(fig, nrows, ncols, plt_idx, **v):
    # extract plot parameters

    plot_data = v['plot_data']
    title = v.get("title", "")
    plt_hist = v.get("plot_hist", False)

    if plt_hist:
        # extract parameters - specific to plot hist / stats
        stats_loc = v.get('stats_loc', (0.2, 0.85))
        q_vminmax = v.get('q_vminmax', None)

        ax = fig.add_subplot(nrows, ncols, plt_idx)
        plot_hist(ax=ax,
                  data=plot_data[~np.isnan(plot_data)],
                  ylabel="",
                  stats_values=['mean', 'std', 'skew', 'kurtosis', 'min', 'max', 'num obs'],
                  title=title,
                  stats_loc=stats_loc,
                  q_vminmax=q_vminmax)
    else:
        # extract parameters - specific to plot pcolor mesh
        lon, lat = v['lon'], v['lat']
        cbar = v.get('cbar_label', None)
        abs_vminmax = v.get("abs_vminmax", False)
        s = v.get("s", 1)
        vmin_max = v.get("vminmax", None)
        cmap = v.get("cmap", 'YlGnBu_r')

        q_vminmax = v.get('q_vminmax', [0.05, 0.95])
        if vmin_max is None:
            if not abs_vminmax:
                vmin, vmax = np.nanquantile(plot_data, q=q_vminmax)
            else:
                max_q = max(q_vminmax) if isinstance(q_vminmax, (tuple, list)) else q_vminmax

                vmax = np.nanquantile(np.abs(plot_data), q=max_q)
                vmin = -vmax
        else:
            vmin, vmax = vmin_max[0], vmin_max[1]

        # first plot: heat map of observations
        ax = fig.add_subplot(nrows, ncols, plt_idx,
                             projection=ccrs.NorthPolarStereo())

        scatter = len(plot_data.shape) == 1

        plot_pcolormesh(ax,
                        lon=lon,
                        lat=lat,
                        plot_data=plot_data,
                        scatter=scatter,
                        vmin=vmin,
                        vmax=vmax,
                        s=s,
                        fig=fig,
                        cbar_label=cbar,
                        cmap=cmap)

        plt.title(title)

    return fig


num_plots_row_col_size = {
    1: {
        "nrows": 1, "ncols": 1, "fig_size": (20, 20)
    },
    2: {
        "nrows": 1, "ncols": 2, "fig_size": (20, 10)
    },
    3: {
        "nrows": 2, "ncols": 2, "fig_size": (20, 20)
    },
    4: {
        "nrows": 2, "ncols": 2, "fig_size": (20, 20)
    },
}

# ----
# dataset to examine
# ----

result_set = {
    # "5x5_SGPR": {
    #     "result_file": "ground_truth_GPOD_binned_seaice_5x5.h5",
    #     "s": 5
    # },
    # "10x10": {
    #     "result_file": "ground_truth_GPOD_binned_seaice_10x10.h5",
    #     "s": 10
    # },
    # "25x25": {
    #     "result_file": "ground_truth_GPOD_binned_seaice_25x25.h5",
    #     "s": 25
    # },
    # "50x50": {
    #     "result_file": "ground_truth_GPOD_binned_seaice_50x50.h5",
    #     "s": 50
    # },
    # "RAW_SGPR": {
    #     "result_file": "ground_truth_w_noise.h5",
    #     "s": 1
    # }
    # "10x10 CS2 Only 17day Window": {
    #     "result_file": "ground_truth_GPOD_binned_seaice_10x10_cs2only.h5",
    #     "s": 10
    # }
    # "10x10 CS2 Only 9day Window": {
    #     "result_file": "ground_truth_GPOD_binned_seaice_10x10_cs2only_9days.h5",
    #     "s": 10
    # },
    # "VFF on RAW": {
    #     "result_file": "ground_truth_GPOD_raw_vff.h5",
    #     "s": 1
    # }
    "Example 25x25": {
        "result_file": "ABC_binned_example.h5",
        "s": 25
    },
}

# ---
# find common prediction locations - to allow for better comparison
# ---

# results directory and file - to plot from
result_dir = get_parent_path("results", "example")

all_pred_locs = []
for k, v in result_set.items():
    store_path = os.path.join(result_dir, v['result_file'])
    with pd.HDFStore(store_path, mode="r") as store:
        # TODO: determine if it's faster to use select_colum - does not have where condition?

        all_keys = store.keys()
        _ = store.select('preds').reset_index()
        _ = _.loc[~np.isnan(_["f*"])]
        # _['lon'], _['lat'] = EASE2toWGS84_New(_['x'], _['y'])
        all_pred_locs.append(_[['pred_loc_x', 'pred_loc_y']].drop_duplicates())

common_pred_locs = reduce(lambda x, y: x.merge(y, how='inner'), all_pred_locs)

# ----
# parameters
# ----

# TODO: review some of the parameters below.. not used


# use common prediction locations?
use_common_pred_locs = True

# column from table to plot
plot_col = "fs"
plot_table = "preds"

# use_var_col = "y_var"
use_var_col = "f*_var"

# image directory
image_dir = get_parent_path("images", os.path.basename(result_dir))

base_image_file = "GroundTruth_sampled_prediction_and_error"

# store simple stats in a dataframe
stat_list = []

for use_result_set in result_set.keys():

    print(f"{'-' * 50}\n{use_result_set}\n{'-' * 20}")

    # which data set to use?
    # use_result_set = "RAW_SGPR"

    # ----
    # set file paths
    # ----

    # a dictionary with different input (result) files, along with output image_file (pdf) + other params

    result_file = result_set[use_result_set]['result_file']
    tmp_image_file = f"{base_image_file}_{use_result_set}.pdf"

    # CHANGE: scatter size for observations raw / binned data
    s_obs = result_set[use_result_set].get("s", 2)

    # where images will be saved

    print(f"storing images in:\n{image_dir}")
    os.makedirs(image_dir, exist_ok=True)

    # where to write plots to
    image_file = os.path.join(image_dir, tmp_image_file)

    # original ground truth file
    gt_file = get_data_path("MSS", "CryosatMSS-arco-2yr-140821_with_geoid_h.csv")

    # where results are stored
    store_path = os.path.join(result_dir, result_file)

    # --
    # read in ground truth
    # --

    gt = pd.read_csv(gt_file)

    # ------------
    # read in previous config
    # ------------

    with pd.HDFStore(store_path, mode='r') as store:
        oi_config = store.get_storer("oi_config").attrs['oi_config']

    # coordinate columns
    coords_col = oi_config['data']['coords_col']

    # get the input data
    # data_source = oi_config['data']['data_source']

    # -----
    # read in observation data
    # -----

    input_data = LocalExpertData(**oi_config['data'])
    input_data.set_data_source()

    obs = input_data.data_source

    # ---
    # read in oi results data
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

    # ----
    # select input observations -  in window
    # -----

    # get the expert location
    exp_locs = dfs['preds'][input_data.coords_col].drop_duplicates()
    # - just take the first location - assumed to be all on the same date
    rl = exp_locs.iloc[0, :]
    local_select = oi_config['data']['local_select']

    # HACK: remove any distance measurements - or only take 't' - this is rigid
    # local_select = [ls for ls in local_select if ls['col'] == 't']

    # TODO: use global select here

    # --
    # get the observations used (for a given date)
    # --

    global_select = oi_config['data'].get("global_select", None)
    if global_select is None:
        global_select = []

    # get current where list
    global_where = DataLoader.get_where_list(global_select,
                                             local_select=local_select,
                                             ref_loc=rl)

    # TODO: make sure this works with raw data (from h5 file)!  need to specify table
    # - get input from oi_config['data'][...]
    df_global = DataLoader.data_select(obj=input_data.data_source,
                                       table=input_data.table,
                                       where=global_where,
                                       return_df=True,
                                       reset_index=True)

    DataLoader.add_cols(df_global, oi_config['data'].get('col_funcs', None))

    # --
    # get the observations mean (removed before hand?)
    # --

    # HACK: this is clunky - in some occasions the observation mean was subtract when input data was stored
    # NOTE: this is not robust, specific to sampling from ground_truth

    # get the obs mean - if it exists and has been taken off
    # - for binned data might have to go to input data
    if isinstance(obs, (xr.core.dataarray.DataArray, xr.core.dataarray.Dataset)):
        if "input_info" in obs.attrs['bin_config']:
            input_file = obs.attrs['bin_config']['input_info']['file']
            input_table = obs.attrs['bin_config']['input_info']['table']
            with pd.HDFStore(input_file, mode="r") as store:
                tmp = store.select(input_table, start=0, stop=10)
                obs_mean = tmp['obs_mean'].mean()
        else:
            warnings.warn("binned data was used as input, however couldn't get 'obs_mean' - will set to 0")
            obs_mean = 0
    elif isinstance(obs,  pd.io.pytables.HDFStore):
        # if observations came from HDFStore -  get info from df_global
        if 'obs_mean' in df_global:
            obs_mean = df_global['obs_mean'].mean()
        else:
            # obs_mean = df_global[input_data.obs_col].mean()
            obs_mean = 0

    # else (assumed DataFrame?)
    else:
        if 'obs_mean' in obs:
            obs_mean = obs['obs_mean'].mean()
        else:
            # obs_mean = obs[input_data.obs_col].mean()
            obs_mean = 0


    # --
    # prep data for plotting
    # --

    # ---
    # multiple predictions
    preds = dfs['preds'].copy(True)

    if use_common_pred_locs:
        preds = common_pred_locs.merge(preds, how='left')

    # remove preidctions with nans
    preds = preds.loc[~np.isnan(preds['f*'])]

    # add back the observation mean
    preds['f*'] += obs_mean

    # average predictions
    pcoord_cols = [f"pred_loc_{c}" for c in coords_col]
    pave = pd.pivot_table(preds,
                          index=pcoord_cols,
                          values=['f*', "f*_var", "y_var"],
                          aggfunc='mean').reset_index()

    pave['lon'], pave['lat'] = EASE2toWGS84(pave['pred_loc_x'], pave['pred_loc_y'])

    # pave = pave.loc[pave['lat'] <= 88.0]

    # remove predictions over land
    pave["is_in_ocean"] = globe.is_ocean(pave['lat'], pave['lon'])

    pave = pave.loc[pave['is_in_ocean']]

    # put data on 2d array
    # - assumes predictions are already on some sort of regular grid!
    delta_x = np.diff(np.sort(pave['pred_loc_x'].unique())).min()
    delta_y = np.diff(np.sort(pave['pred_loc_y'].unique())).min()

    x_start = pave['pred_loc_x'].min()
    x_end = pave['pred_loc_x'].max()
    x_coords = np.arange(x_start, x_end + delta_x, delta_x)

    y_start = pave['pred_loc_y'].min()
    y_end = pave['pred_loc_y'].max()
    y_coords = np.arange(y_start, y_end + delta_y, delta_y)

    x_grid, y_grid = np.meshgrid(x_coords, y_coords)

    pave['grid_loc_x'] = match(pave['pred_loc_x'].values, x_coords)
    pave['grid_loc_y'] = match(pave['pred_loc_y'].values, y_coords)

    # check there is only one grid_loc for each point
    pave[['grid_loc_x', 'grid_loc_y']].drop_duplicates().shape[0] == pave.shape[0]

    lon_grid, lat_grid = EASE2toWGS84(x_grid, y_grid)

    # populate a 2d array of values to plot
    pave2d = np.full(x_grid.shape, np.nan)
    pvar2d = np.full(x_grid.shape, np.nan)

    pave2d[pave['grid_loc_y'].values, pave['grid_loc_x'].values] = pave['f*'].values
    pvar2d[pave['grid_loc_y'].values, pave['grid_loc_x'].values] = pave[use_var_col].values

    # ---
    # ground truth
    # ---

    # TODO: wrap the following up into a method
    # - get the closest points

    gt['z'] = gt['mss'] - gt['h']
    gt['x'], gt['y'] = WGS84toEASE2(gt['lon'], gt['lat'])

    gt = gt.loc[gt['lat'] > pave['lat'].min()]

    kdt = KDTree(gt[['x','y']].values)
    r, idx = kdt.query(pave[['pred_loc_x', 'pred_loc_y']].values, k=1)

    pave['r'], pave['idx'] = r, idx
    pave['gt'] = gt['z'].values[idx]

    # set ground truth values that were 'some' distance away from prediction location to nan
    pave.loc[pave['r'] > delta_x, 'gt'] = np.nan

    gt2d = np.full(x_grid.shape, np.nan)
    gt2d[pave['grid_loc_y'].values, pave['grid_loc_x'].values] = pave['gt'].values

    # error
    pave['error'] = pave['f*'].values - pave['gt'].values
    error_2d = np.full(x_grid.shape, np.nan)
    error_2d[pave['grid_loc_y'].values, pave['grid_loc_x'].values] = pave['error']

    # normalised error
    pave['norm_error'] = pave['error'] / np.sqrt(pave[use_var_col])
    norm_error_2d = np.full(x_grid.shape, np.nan)
    norm_error_2d[pave['grid_loc_y'].values, pave['grid_loc_x'].values] = pave['norm_error']

    # negative log likelihood
    pave['nll'] = nll(y=pave['f*'], mu=pave['gt'], sig=np.sqrt(pave[use_var_col]), return_tot=False)
    nll2d = np.full(x_grid.shape, np.nan)
    nll2d[pave['grid_loc_y'].values, pave['grid_loc_x'].values] = pave['nll']

    # -----
    # plot dictionary
    # -----

    # --
    # error plot data
    # ---
    std_error = np.nanstd(error_2d)

    error_plot = {
            "plot_data": error_2d,
            "lon": lon_grid,
            "lat": lat_grid,
            "cbar_label": "prediction - ground truth",
            "title": f"Error: Prediction - Ground Truth (Nearest to Prediction Location) \n:Std Dev of Error: {std_error:.4f}",
            "cmap": "bwr",
            "vminmax": [-0.275, 0.275],
            "q_vminmax": [0.005, 0.99]
    }
    # create one plot with histogram, other with heat map
    error_plots = [{**error_plot, **{"plot_hist": tf}} for tf in [False, True]]

    # --
    # normalised error plot data
    # ---

    norm_err_std = np.nanstd(norm_error_2d)

    nerror_plot = {
            "plot_data": norm_error_2d,
            "lon": lon_grid,
            "lat": lat_grid,
            "cbar_label": f"(prediction - ground truth) / prediction std dev (sqrt({use_var_col}))",
            "title": f"Normalised Error: Prediction - Ground Truth / Prediction Std Dev (sqrt({use_var_col})) "
                     f"\n:Std Dev of Normalised Error: {norm_err_std:.3f}",
            "cmap": "bwr",
            # for heatmap if vminmax is provide will use those values , otherwise will use quantiles using q_vminmax
            "vminmax": [-3.0, 3.0],
            # vminmax not provided to histogram, only q_vminmax is used
            "q_vminmax": [0.005, 0.995]
    }
    # create one plot with histogram, other with heat map
    nerror_plots = [{**nerror_plot, **{"plot_hist": tf}} for tf in [False, True]]

    # --
    # NLL plot data
    # ---

    mean_nll = np.mean(nll2d[~np.isnan(nll2d)])

    nll_plot = {
            "plot_data": nll2d,
            "lon": lon_grid,
            "lat": lat_grid,
            "cbar_label": f"NLL: prediction variance column ({use_var_col})",
            "title": f"Negative Log Likelihood \n Mean NLL: {mean_nll:.4f}",
            "cmap": "nipy_spectral",
            "q_vminmax": [0.05, 0.95],
            "vminmax": [-3.2, 3.2],
            "stats_loc": (0.35, 0.85)
    }
    # create one plot with histogram, other with heat map
    nll_plot = [{**nll_plot, **{"plot_hist": tf}} for tf in [False, True]]

    # --
    # lengthscales
    # --

    coords_col = oi_config['data']['coords_col']
    dim_map = {i: cc for i, cc in enumerate(coords_col)}

    ls = dfs['lengthscales'].copy(True)
    ls['dim_name'] = ls['_dim_0'].map(dim_map)

    plot_length_scales = []
    for cc in coords_col:

        _ = ls.loc[ls['dim_name'] == cc]

        tmp = {
            "plot_data": _['lengthscales'],
            "lon": _['lon'],
            "lat": _['lat'],
            "cbar_label": f"(scaled) lengthscale for dim: {cc}",
            # "title": "Along Track Observations of Ground Truth (MSS - Geoid) - With Noise",
            "title": f"Local Expert Length Scale for dim: {cc}",
            "s": 250,
            "q_vminmax": [0.001, 0.999]
        }
        plot_length_scales.append(tmp)

    # variances

    plot_variances = {}
    for var_name in [k for k in dfs.keys() if re.search("variance", k)]:
        _ = dfs[var_name]
        tmp = {
            "plot_data": _[var_name],
            "lon": _['lon'],
            "lat": _['lat'],
            "cbar_label": f"{var_name}",
            # "title": "Along Track Observations of Ground Truth (MSS - Geoid) - With Noise",
            "title": f"{var_name}",
            "s": 650,
            "q_vminmax": [0.01, 0.99]
        }
        plot_variances[var_name] = tmp


    model_name = oi_config['model']
    bin_or_raw = "Binned" if re.search("\.zarr$|\.nc$", oi_config['data']['data_source']) else "Raw (Along Track)"
    obs_col = oi_config['data']['obs_col']

    plt_dict = {
        "ground truth": {
            "plot_data": gt2d,
            "lon": lon_grid,
            "lat": lat_grid,
            "cbar_label": "ground truth: MSS - Geoid",
            "title": "(Nearest to Prediction Location) Ground Truth"
        },
        "observations": {
            "plot_data": df_global[input_data.obs_col] + obs_mean,
            "lon": df_global['lon'],
            "lat": df_global['lat'],
            "cbar_label": f"{obs_col} data values",
            "title": f"{bin_or_raw} Observations of Ground Truth (MSS - Geoid) - With Noise",
            "s": s_obs
        },
        "predictions": {
            "plot_data": pave2d,
            "lon": lon_grid,
            "lat": lat_grid,
            "cbar_label": "OI: MSS - Geoid",
            "title": f"{oi_config['model']['oi_model']} Predictions (averaged) from Multiple Local Experts using "
                     f"binned input"
        },
        "errors": error_plots,
        "normalised errors": nerror_plots,
        "Negative Log Likelihood": nll_plot,
        "lengthscales": plot_length_scales,
        **plot_variances
    }

    # ----
    # plot data
    # -----

    print(f"writing plots to:\n{image_file}")
    with PdfPages(image_file) as pdf:

        for k, v in plt_dict.items():

            print(k)
            # single plot
            if isinstance(v, dict):
                rcs = num_plots_row_col_size[1]
                nrows, ncols, fig_size = rcs['nrows'], rcs['ncols'], rcs['fig_size']
                fig = plt.figure(figsize=fig_size)
                plot_helper(fig, nrows, ncols, 1, **v)
            # list of plots
            elif isinstance(v, list):
                rcs = num_plots_row_col_size[len(v)]
                nrows, ncols, fig_size = rcs['nrows'], rcs['ncols'], rcs['fig_size']
                fig = plt.figure(figsize=fig_size)
                plt_idx = 0
                for vv in v:
                    plt_idx += 1
                    fig = plot_helper(fig, nrows, ncols, plt_idx, **vv)

            plt.tight_layout()
            pdf.savefig(fig)

            plt.show()


    print(f"FINISHED writing plots to:\n{image_file}")

    # create a table
    stat_row = pd.DataFrame({
        "file": os.path.basename(image_file),
        "mean NLL": np.round(mean_nll, 5),
        "std norm error": np.round(norm_err_std, 5),
        "std error": np.round(std_error, 5)
    }, index=[0])
    stat_list.append(stat_row)


out_stats = pd.concat(stat_list)
out_stats.to_csv(os.path.join(image_dir, f"{base_image_file}_stats.csv"), index=False)





# image_file = os.path.join(image_dir, "GroundTruth_sampled_prediction_and_diff_cs2.pdf")
# image_file = os.path.join(image_dir, "GroundTruth_sampled_prediction_and_diff_noise0p2.pdf")
# image_file = os.path.join(image_dir, "GroundTruth_sampled_prediction_and_diff_binned_obs_seaice.pdf")

# image_file = os.path.join(image_dir, "GroundTruth_sampled_prediction_and_diff_RAW_SGPR.pdf")
# image_file = os.path.join(image_dir, "GroundTruth_sampled_prediction_and_diff_binned_obs_seaice_50x50.pdf")
# image_file = os.path.join(image_dir, "GroundTruth_sampled_prediction_and_diff_binned_obs_seaice_25x25.pdf")
# image_file = os.path.join(image_dir, "GroundTruth_sampled_prediction_and_diff_binned_obs_seaice_10x10.pdf")
