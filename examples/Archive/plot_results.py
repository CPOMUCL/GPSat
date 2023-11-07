# generate plots for results
# - for each selected dim (e.g. t):
# - plot hyper parameters at expert locations
# - plot predictions and uncertainty

import re
import copy

import pandas as pd
import numpy as np

import cartopy.crs as ccrs

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from GPSat import get_parent_path
from GPSat.utils import cprint
from GPSat.local_experts import get_results_from_h5file
from GPSat.plot_utils import get_projection, plot_xy_from_results_data,\
    plot_pcolormesh_from_results_data, plot_hist_from_results_data
from GPSat.dataloader import DataLoader

# -----
# helper functions
# -----


pd.set_option("display.max_columns", 200)

# layout for number of plots
num_plots_row_col_size = {
    1: {
        "nrows": 1, "ncols": 1, "fig_size": (15, 15)
    },
    2: {
        "nrows": 1, "ncols": 2, "fig_size": (20, 10)
    },
    3: {
        "nrows": 2, "ncols": 2, "fig_size": (15, 15)
    },
    4: {
        "nrows": 2, "ncols": 2, "fig_size": (15, 15)
    },
    5: {
        "nrows": 2, "ncols": 3, "fig_size": (15, 10)
    },
    6: {
        "nrows": 2, "ncols": 3, "fig_size": (15, 10)
    },
    7: {
        "nrows": 3, "ncols": 3, "fig_size": (15, 15)
    },
    8: {
        "nrows": 3, "ncols": 3, "fig_size": (15, 15)
    },
    9: {
        "nrows": 3, "ncols": 3, "fig_size": (15, 15)
    },
}


# ----
# parameters
# ----

# results_file = get_parent_path("results", "ground_truth", "ground_truth_ABC_binned_seaice_25x25.h5")
results_file = get_parent_path("results", "example", "ABC_binned_example.h5")
# results_file = get_parent_path("results", "elev", "GPOD_elev_lead_binned_25x25km.h5")
# results_file = get_parent_path("results", "elev", "GPOD_elev_lead_binned_25x25km.h5")

# results_file = get_parent_path("results", "xval", "cs2cpom_lead_binned_date_2019_2020_25x25km.h5")


# increment over
increment_over = "date"

#
table_names = ["lengthscales", "kernel_variance", "likelihood_variance"]

# image_file = re.sub("\.h5$", "_plot_summary.pdf", results_file)
global_col_funcs = None
# global_col_funcs = {"date":
#     {
#         "func": "lambda t: t.astype('datetime64[D]')",
#         "col_args": "t"
#     }
# }
merge_on_expert_locations = True

# -----
# read in results data - get dict of DataFrames
# -----

dfs, oi_config = get_results_from_h5file(results_file,
                                         global_col_funcs=global_col_funcs,
                                         merge_on_expert_locations=merge_on_expert_locations)

# ----
# plot hyper parameters
# ----

# ----

sup_title = "hyper params"

dim_map = {idx: _ for idx, _ in enumerate(oi_config[0]['data']['coords_col'])}

# which colum  should be used for each column
table_to_col = {k: k for k in table_names}

# determine the hyper parameters per
dim_vals = {k: dfs[k]["_dim_0"].unique() for k in table_names}

# num_plots = np.sum([len(v) for k, v in dim_vals.items()])

# get the row_select
row_selects = {k: [{"col": "_dim_0", "comp": "==", "val": _} for _ in v]
               for k, v in dim_vals.items()}

# get the vmin/vmax values, based off of quantiles
vmin_max = {}

for k, v in row_selects.items():

    res = []
    for rs in v:
        _ = DataLoader.load(dfs[k],
                            row_select=rs)
        vmin, vmax = np.nanquantile(_[k], q=[0.01, 0.99])
        res.append({"vmin": vmin, "vmax": vmax})

    vmin_max[k] = res

# plot_template = {
#     "plot_type": "heatmap",
#     "lon_col": "lon",
#     "lat_col": "lat",
#     "subplot_kwargs": {"projection": "north"},
#     # any additional arguments for plot_hist
#     "plot_kwargs": {
#         "scatter": True,
#     }
# }

plot_template = {
    "plot_type": "heatmap",
    "x_col": "x",
    "y_col": "y",
    "lat_0": 90,
    "lon_0": 0,
    "subplot_kwargs": {"projection": "north"},
    # any additional arguments for plot_hist
    "plot_kwargs": {
        "scatter": False,
    }
}

increment_over_vals = dfs['run_details'][increment_over].unique()


image_file = re.sub("\.h5", "_HYPER_PARAMS.pdf", results_file)
cprint(f"writing plots of hyper parameters to:\n{image_file}")
with PdfPages(image_file) as pdf:

    for icv in increment_over_vals:

        # ---
        # create a list of plot configs
        # ---

        plot_configs = []

        # increment over the hyper parmaeters
        for hp in table_names:

            # increment over the row select
            for idx, rs in enumerate(row_selects[hp]):
                load_kwargs = {
                    "row_select": [{"col": increment_over, "comp": "==", "val": icv},
                                   rs]
                }
                # vmin/max
                vm = vmin_max[hp][idx]

                tmp = copy.deepcopy(plot_template)

                tmp['val_col'] = table_to_col[hp]
                tmp['table'] = hp
                tmp['load_kwargs'] = {**tmp.get("load_kwargs", {}), **load_kwargs}
                tmp['plot_kwargs'] = {**tmp.get("plot_kwargs", {}), **vm}

                dim_name = dim_map[rs['val']] if len(row_selects[hp]) > 1 else ""
                tmp['plot_kwargs']["title"] = f"{hp} {dim_name}"

                plot_configs.append(tmp)

        # ---
        # generate the plot
        # ---
        plt_idx = 1

        rcs = num_plots_row_col_size[len(plot_configs)]
        nrows, ncols, fig_size = rcs['nrows'], rcs['ncols'], rcs['fig_size']
        fig = plt.figure(figsize=fig_size)

        fig.suptitle(f"{sup_title}\n{increment_over} == {icv}")

        for p in plot_configs:

            # create a new subplot
            subplot_kwargs = p.get("subplot_kwargs", {})
            if "projection" in subplot_kwargs:
                subplot_kwargs["projection"] = get_projection(subplot_kwargs["projection"])

            ax = fig.add_subplot(nrows, ncols, plt_idx, **subplot_kwargs)

            assert 'plot_type' in p, "'plot_type' is not specified in plot_config"
            if p['plot_type'] == "plot_xy":
                plot_xy_from_results_data(ax=ax, dfs=dfs, **p)
            elif p['plot_type'] == "hist":
                plot_hist_from_results_data(ax=ax, dfs=dfs, **p)
            elif p['plot_type'] == "heatmap":
                plot_pcolormesh_from_results_data(ax=ax, dfs=dfs, fig=fig, **p)
            else:
                raise NotImplementedError(f"plot_type: '{p['plot_type']}' is not implemented")
            plt_idx += 1

        plt.tight_layout()
        # plt.show()
        pdf.savefig(fig)


# ------
# predictions
# ------

# which colum  should be used for each column
# table_to_col = {k: k for k in table_names}
col_to_table = {"f*": "preds",
                "f*_var": "preds"}

# get the vmin/vmax values, based off of quantiles
vmin_max = {}

for k, v in col_to_table.items():

    _ = DataLoader.load(dfs[v])
    # vmin, vmax = np.nanquantile(_[k], q=[0.01, 0.99])
    vmax = np.nanquantile(np.abs(_[k]), q=0.99)
    vmin = -vmax

    vmin_max[k] = {"vmin": vmin, "vmax": vmax}


plot_template = {
    "plot_type": "heatmap",
    # "table": "preds",
    # "load_kwargs": {
    #                 "col_funcs": {
    #                     ("pred_lon", "pred_lat"): {
    #                         "source": "GPSat.utils",
    #                         "func": "EASE2toWGS84_New",
    #                         "col_kwargs": {
    #                             "x": "pred_loc_x",
    #                             "y": "pred_loc_y"
    #                         },
    #                         "kwargs": {
    #                             "lat_0": 90
    #                         }
    #                     }
    #                 }
    #             },
    # "val_col": "f*",
    # "lon_col": "pred_lon",
    # "lat_col": "pred_lat",
    "x_col": "pred_loc_x",
    "y_col": "pred_loc_y",
    "subplot_kwargs": {"projection": "north"},
    "weighted_values_kwargs": {
        "ref_col": ["pred_loc_x", "pred_loc_y", "pred_loc_t"],
        "dist_to_col": ["x", "y", "t"],
        "val_cols": "f*",
        "weight_function": "gaussian",
        "lengthscale": 200_000
    },
    # any additional arguments for plot_hist
    "plot_kwargs": {
        "scatter": False,
        "cmap": "bwr",
        "ocean_only": True
    }
}


increment_over_vals = dfs['run_details'][increment_over].unique()


image_file = re.sub("\.h5", "_PREDICTIONS.pdf", results_file)
cprint(f"writing plots of predictions to:\n{image_file}")
with PdfPages(image_file) as pdf:

    for icv in increment_over_vals:

        # ---
        # create a list of plot configs
        # ---

        plot_configs = []

        # increment over the hyper parmaeters
        for col, table in col_to_table.items():

            # increment over the row select
            for idx, rs in enumerate(row_selects[hp]):
                load_kwargs = {
                    "row_select": [{"col": increment_over, "comp": "==", "val": icv}]
                }
                # vmin/max
                vm = vmin_max[col]

                tmp = copy.deepcopy(plot_template)

                tmp['val_col'] = col
                tmp['table'] = table
                tmp["weighted_values_kwargs"]["val_cols"] = col

                tmp['load_kwargs'] = {**tmp.get("load_kwargs", {}), **load_kwargs}
                tmp['plot_kwargs'] = {**tmp.get("plot_kwargs", {}), **vm}
                tmp['plot_kwargs']["title"] = f"{col} - from {table_to_col}"

                plot_configs.append(tmp)

        # ---
        # generate the plot
        # ---
        plt_idx = 1

        rcs = num_plots_row_col_size[len(plot_configs)]
        nrows, ncols, fig_size = rcs['nrows'], rcs['ncols'], rcs['fig_size']
        fig = plt.figure(figsize=fig_size)

        fig.suptitle(f"{sup_title}\n{increment_over} == {icv}")

        for p in plot_configs:

            # create a new subplot
            subplot_kwargs = p.get("subplot_kwargs", {})
            if "projection" in subplot_kwargs:
                subplot_kwargs["projection"] = get_projection(subplot_kwargs["projection"])

            ax = fig.add_subplot(nrows, ncols, plt_idx, **subplot_kwargs)

            assert 'plot_type' in p, "'plot_type' is not specified in plot_config"
            if p['plot_type'] == "plot_xy":
                plot_xy_from_results_data(ax=ax, dfs=dfs, **p)
            elif p['plot_type'] == "hist":
                plot_hist_from_results_data(ax=ax, dfs=dfs, **p)
            elif p['plot_type'] == "heatmap":
                plot_pcolormesh_from_results_data(ax=ax, dfs=dfs, fig=fig, **p)
            else:
                raise NotImplementedError(f"plot_type: '{p['plot_type']}' is not implemented")
            plt_idx += 1

        plt.tight_layout()
        # plt.show()
        pdf.savefig(fig)



# scatter plot of predictions
plot_heatmap_config = {
    "plot_type": "heatmap",
    "table": "preds",
    "load_kwargs": {
                    "col_funcs": {
                        ("pred_lon", "pred_lat"): {
                            "source": "GPSat.utils",
                            "func": "EASE2toWGS84_New",
                            "col_kwargs": {
                                "x": "pred_loc_x",
                                "y": "pred_loc_y"
                            },
                            "kwargs": {
                                "lat_0": 90
                            }
                        }
                    }
                },
    "val_col": "f*",
    "lon_col": "pred_lon",
    "lat_col": "pred_lat",
    "subplot_kwargs": {"projection": "north"},
    "weighted_values_kwargs": {
        "ref_col": ["pred_loc_x", "pred_loc_y", "pred_loc_t"],
        "dist_to_col": ["x", "y", "t"],
        "val_cols": "f*",
        "weight_function": "gaussian",
        "lengthscale": 200_000
    },
    # any additional arguments for plot_hist
    "plot_kwargs": {
        "scatter": True,
        "cmap": "bwr",
        "vmin": -0.2,
        "vmax": 0.2
    }
}


# ---
# determine how to iterate over results
# --


# make load_kwargs for each config to take in
# put load_kwargs in

# date = increment_over_vals[0]


for icv in increment_over_vals:

    load_kwargs = {
        "row_select": {"col": increment_over, "comp": "==", "val": icv}
    }




    # generate plot a single plot
    rcs = num_plots_row_col_size[1]
    nrows, ncols, fig_size = rcs['nrows'], rcs['ncols'], rcs['fig_size']
    plt_idx = 1

    fig = plt.figure(figsize=fig_size)

    row_select = " ".join([str(v) for k, v in load_kwargs['row_select'].items()])
    fig.suptitle(row_select)

    # subplot - kwargs
    subplot_kwargs = plot_heatmap_config.get("subplot_kwargs", {})
    if "projection" in subplot_kwargs:
        subplot_kwargs["projection"] = get_projection(subplot_kwargs["projection"])

    ax = fig.add_subplot(nrows, ncols, plt_idx, **subplot_kwargs)

    plot_pcolormesh_from_results_data(ax=ax, dfs=dfs, fig=fig, **plot_heatmap_config)

    plt.tight_layout()
    plt.show()



# ----
# simple x,y plot
# ----

# pick a coordinate

run_dets = dfs['run_details']


plot_run_time = {
    # plot type
    "plot_type": "plot_xy",
    "table": "run_details",
    "load_kwargs": {
        **load_kwargs
    },
    # plot parameters
    "x_col": "num_obs",
    "y_col": "run_time",

    # keyword arguments for plot_xy
    "plot_kwargs": {
        "title": "Number of Input Observations VS Run Time",
        "x_label": "Number of Obs",
        "y_label": "Run Time (in seconds)",
        "scatter": True,
        "color": "red",
        "alpha": 0.5
    }
}

# generate plot a single plot
rcs = num_plots_row_col_size[1]
nrows, ncols, fig_size = rcs['nrows'], rcs['ncols'], rcs['fig_size']
plt_idx = 1

fig = plt.figure(figsize=fig_size)
ax = fig.add_subplot(nrows, ncols, plt_idx)

plot_xy_from_results_data(ax, dfs, **plot_run_time)

plt.tight_layout()
plt.show()

# ----
# histogram
# ----


plot_hist_config = {
    "plot_type": "hist",
    "table": "preds_SMOOTHED",
    "load_kwargs": {
        **load_kwargs
    },
    "val_col": "f*",
    # any additional arguments for plot_hist
    "plot_kwargs": {
        "ylabel": "count",
        "stats_values": ['mean', 'std', 'skew', 'kurtosis', 'min', 'max', 'num obs'],
        "title": "Prediction Values",
        "xlabel": "f*",
        "stats_loc": (0.2, 0.8)
    }
}


# generate plot a single plot
rcs = num_plots_row_col_size[1]
nrows, ncols, fig_size = rcs['nrows'], rcs['ncols'], rcs['fig_size']
plt_idx = 1

fig = plt.figure(figsize=fig_size)
ax = fig.add_subplot(nrows, ncols, plt_idx)


plot_hist_from_results_data(ax=ax, dfs=dfs, **plot_hist_config)

plt.show()


# ----
# plot heatmap/pcolormesh: scatter
# ----

#
# plot_heatmap_config = {
#     "plot_type": "heatmap",
#     "table": "preds",
#     "load_kwargs": None,
#     "val_col": "f*",
#     "lon_col": "lon",
#     "lat_col": "lat",
#     "subplot_kwargs": {"projection": "north"},
#     # "weighted_values_kwargs": {
#     #     "ref_col": ["pred_loc_x", "pred_loc_y", "pred_loc_t"],
#     #     "dist_to_col": ["x", "y", "t"],
#     #     "val_cols": "f*",
#     #     "weight_function": "gaussian",
#     #     "lengthscale": 200_000
#     # },
#     # any additional arguments for plot_hist
#     "plot_kwargs": {
#         "scatter": True,
#         "cmap": "YlGnBu_r"
#     }
# }
#
#
# # generate plot a single plot
# rcs = num_plots_row_col_size[1]
# nrows, ncols, fig_size = rcs['nrows'], rcs['ncols'], rcs['fig_size']
# plt_idx = 1
#
# fig = plt.figure(figsize=fig_size)
#
# # subplot - kwargs
# subplot_kwargs = plot_heatmap_config.get("subplot_kwargs", {})
# if "projection" in subplot_kwargs:
#     subplot_kwargs["projection"] = get_projection(subplot_kwargs["projection"])
#
# ax = fig.add_subplot(nrows, ncols, plt_idx, **subplot_kwargs)
#
#
# plot_pcolormesh_from_results_data(ax=ax, dfs=dfs, fig=fig, **plot_heatmap_config)
#
# plt.show()
#

# ----
# plot heatmap: 2D
# ----

plot_heatmap_config = {
    "plot_type": "heatmap",
    "table": "preds_SMOOTHED",
    "load_kwargs": {
        "row_select": [{"col": "t", "comp": "==", "val": 18293}]
    },
    "val_col": "f*",
    "x_col": "pred_loc_x",
    "y_col": "pred_loc_y",
    "lat_0": 90,
    "lon_0": 0,
    "subplot_kwargs": {"projection": "north"},
    # any additional arguments for plot_hist
    "plot_kwargs": {
        "scatter": False,
        "cmap": "bwr",
        "ocean_only": True,
        "qvmin": 0.01,
        "qvmax": 0.99
    },
    "weighted_values_kwargs": {
        "ref_col": ["pred_loc_x", "pred_loc_y", "pred_loc_t"],
        "dist_to_col": ["x", "y", "t"],
        "val_cols": "f*",
        "weight_function": "gaussian",
        "lengthscale": 200_000
    }
}

# generate plot a single plot
rcs = num_plots_row_col_size[1]
nrows, ncols, fig_size = rcs['nrows'], rcs['ncols'], rcs['fig_size']
plt_idx = 1

fig = plt.figure(figsize=fig_size)

# subplot - kwargs
subplot_kwargs = plot_heatmap_config.get("subplot_kwargs", {})
if "projection" in subplot_kwargs:
    subplot_kwargs["projection"] = get_projection(subplot_kwargs["projection"])

ax = fig.add_subplot(nrows, ncols, plt_idx, **subplot_kwargs)

plot_pcolormesh_from_results_data(ax=ax, dfs=dfs, fig=fig, **plot_heatmap_config)

plt.show()


# ----
# multiple plots
# ----

# create an x,y plot config for marginal loglikelihood (objective function)
plot_mll = copy.deepcopy(plot_run_time)
plot_mll['y_col'] = "mll"
plot_mll['plot_kwargs']["title"] = "Number of Input Observations VS MLL"

plot_configs = [plot_run_time, plot_mll, plot_hist_config, plot_heatmap_config]

# determine plot rows, cols and size from number of subplots
rcs = num_plots_row_col_size[len(plot_configs)]
nrows, ncols, fig_size = rcs['nrows'], rcs['ncols'], rcs['fig_size']
fig = plt.figure(figsize=fig_size)

plt_idx = 1

for p in plot_configs:

    # create a new subplot
    subplot_kwargs = p.get("subplot_kwargs", {})
    if "projection" in subplot_kwargs:
        subplot_kwargs["projection"] = get_projection(subplot_kwargs["projection"])

    ax = fig.add_subplot(nrows, ncols, plt_idx, **subplot_kwargs)

    assert 'plot_type' in p, "'plot_type' is not specified in plot_config"
    if p['plot_type'] == "plot_xy":
        plot_xy_from_results_data(ax=ax, dfs=dfs, **p)
    elif p['plot_type'] == "hist":
        plot_hist_from_results_data(ax=ax, dfs=dfs, **p)
    elif p['plot_type'] == "heatmap":
        plot_pcolormesh_from_results_data(ax=ax, dfs=dfs, fig=fig, **p)
    else:
        raise NotImplementedError(f"plot_type: '{p['plot_type']}' is not implemented")
    plt_idx += 1

plt.tight_layout()
plt.show()
