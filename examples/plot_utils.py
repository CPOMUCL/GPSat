# example of use of plotting data from results (local expert oi output)
# - based on configs
import copy

import pandas as pd
import numpy as np

import cartopy.crs as ccrs

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from GPSat import get_parent_path


from GPSat.local_experts import get_results_from_h5file
from GPSat.plot_utils import get_projection, plot_xy_from_results_data,\
    plot_pcolormesh_from_results_data, plot_hist_from_results_data

# -----
# helper functions
# -----


pd.set_option("display.max_columns", 200)

# layout for number of plots
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
    5: {
        "nrows": 2, "ncols": 3, "fig_size": (15, 10)
    },
    6: {
        "nrows": 2, "ncols": 3, "fig_size": (15, 10)
    },
    7: {
        "nrows": 3, "ncols": 3, "fig_size": (20, 20)
    },
    8: {
        "nrows": 3, "ncols": 3, "fig_size": (20, 20)
    },
    9: {
        "nrows": 3, "ncols": 3, "fig_size": (20, 20)
    },
}


# ----
# parameters
# ----

# results_file = get_parent_path("results", "ground_truth", "ground_truth_ABC_binned_seaice_25x25.h5")
results_file = get_parent_path("results", "example", "ABC_binned_example.h5")


# image_file = re.sub("\.h5$", "_plot_summary.pdf", results_file)
global_col_funcs = None
merge_on_expert_locations = True

# -----
# read in results data - get dict of DataFrames
# -----


dfs, oi_config = get_results_from_h5file(results_file, global_col_funcs=global_col_funcs,
                                         merge_on_expert_locations=merge_on_expert_locations)


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
    "table": "preds",
    "load_kwargs": None,
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


plot_heatmap_config = {
    "plot_type": "heatmap",
    "table": "preds",
    "load_kwargs": None,
    "val_col": "f*",
    "lon_col": "lon",
    "lat_col": "lat",
    "subplot_kwargs": {"projection": "north"},
    # any additional arguments for plot_hist
    "plot_kwargs": {
        "scatter": True,
        "cmap": "YlGnBu_r"
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
# plot heatmap: 2D
# ----

# TODO: add example for 2D heatmap, is plot_utils or plot_from_results as reference


# ----
# multiple plots
# ----

# create an x,y plot config for marginal loglikelihood (objective function)
plot_mll = copy.deepcopy(plot_run_time)
plot_mll['y_col'] = "objective_value"
plot_mll['plot_kwargs']["title"] = "Number of Input Observations VS objective_value"

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
