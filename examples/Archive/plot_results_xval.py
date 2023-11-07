# generate plots for xval results
# - for each selected dim (e.g. t):
# - plot hyper parameters at expert locations
# - plot predictions and uncertainty

import re
import copy

import pandas as pd
import numpy as np

from functools import reduce

import cartopy.crs as ccrs

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from GPSat import get_parent_path
from GPSat.utils import cprint
from GPSat.local_experts import get_results_from_h5file
from GPSat.plot_utils import get_projection, plot_xy_from_results_data,\
    plot_pcolormesh_from_results_data, plot_hist_from_results_data
from GPSat.dataloader import DataLoader
from GPSat.local_experts import LocalExpertOI
from GPSat.utils import get_weighted_values

# -----
# helper functions
# -----

def plots_from_config(plot_configs, dfs, num_plots_row_col_size, suptitle=""):
    plt_idx = 1

    rcs = num_plots_row_col_size[len(plot_configs)]
    nrows, ncols, fig_size = rcs['nrows'], rcs['ncols'], rcs['fig_size']
    fig = plt.figure(figsize=fig_size)

    fig.suptitle(suptitle, y=1.0)

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

    return fig


def nll(y, mu, sig, return_tot=True):
    # negative log likelihood assuming independent normal observations (y)
    out = np.log(sig * np.sqrt(2 * np.pi)) + (y - mu)**2 / (2 * sig**2)
    if return_tot:
        return np.sum(out[~np.isnan(out)])
    else:
        return out

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

results_file = get_parent_path("results", "xval", "cs2cpom_lead_binned_date_2019_2020_25x25km.h5")

# prediction tables
select_tables = ["preds", "preds_SMOOTHED", 'run_details']
pred_tables = [k for k in select_tables if re.search("^pred", k)]

# increment over
increment_over = "date"

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
                                         merge_on_expert_locations=merge_on_expert_locations,
                                         select_tables=select_tables)

# -----
# get the weighted combination of predictions
# -----

print("getting weighted combination")

weighted_values_kwargs = {
    "ref_col": ["pred_loc_x", "pred_loc_y", "pred_loc_t"],
    "dist_to_col": ["x", "y", "t"],
    "val_cols": ["f*", "f*_var", "y_var"],
    "weight_function": "gaussian",
    "lengthscale": 200_000
}

plt_data = {}
for k in pred_tables:
    print(k)
    plt_data[k] = get_weighted_values(df=dfs[k], **weighted_values_kwargs)

# ----
# get the prediction location - full data (assumed observations are in hold out locations)
# ---

pred_locs = []
for oic in oi_config:

    # NOTE: this will only work if reading from source
    assert oic['pred_loc']['method'] == "from_source"
    p = DataLoader.load(reset_index=False, **oic['pred_loc']['load_kwargs'])
    pred_locs.append(p)
pred_locs = pd.concat(pred_locs)

# -----
# merge on predictions with the prediction locations
# -----

obs_col = oi_config[0]['data']['obs_col']

coords_col = oi_config[0]['data']['coords_col']

# NOTE: this might not work every time, pred_locs may not have 'date'
extra_col = ["date", "lon", "lat"]

pred_coords_col = [f"pred_loc_{_}" for _ in coords_col]

xval_data = {}

for k in pred_tables:
    xval_data[k] = plt_data[k][["f*", "f*_var", "y_var"] + pred_coords_col].merge(pred_locs[[obs_col] + coords_col + extra_col],
                                                                                  left_on=pred_coords_col,
                                                                                  right_on=coords_col,
                                                                                  how='inner')

    xval_data[k]['diff'] = xval_data[k][obs_col] - xval_data[k]["f*"]
    xval_data[k]['norm_diff'] = xval_data[k]['diff'] / np.sqrt(xval_data[k]["y_var"])
    xval_data[k]['nll'] = nll(y=xval_data[k][obs_col],
                              mu=xval_data[k]["f*"],
                              sig=np.sqrt(xval_data[k]["y_var"]),
                              return_tot=False)


# ----
# get only common predictions locations
# ----

# WHY IS THIS NEEDED? seems like some of the original values are very bad (nan / inf, due to bad y_var?)
# - is this needed?
# TODO: consider commenting out
common_locs = reduce(lambda x, y: x.merge(y, on=pred_coords_col, how='inner'),
                     [xval_data[k][pred_coords_col] for k in pred_tables])

for k in pred_tables:
    xval_data[k] = common_locs.merge(xval_data[k],
                                     on=pred_coords_col,
                                     how='left')


# ------
# predictions
# ------

# --
# hist
# --

plot_hist_config = {
    "plot_type": "hist",
    # "table": "preds_SMOOTHED",
    "load_kwargs": {
        # **load_kwargs
    },
    "val_col": "norm_diff",
    # any additional arguments for plot_hist
    "plot_kwargs": {
        "ylabel": "count",
        "stats_values": ['mean', 'std', 'skew', 'kurtosis', 'min', 'max', 'num obs'],
        "title": "X-val:",
        "xlabel": "norm_diff",
        "stats_loc": (0.2, 0.8)
    }
}

plot_configs = []


for k in pred_tables:
    c = copy.deepcopy(plot_hist_config)
    c['table'] = k
    plot_configs.append(c)

# generate the plots from config
plots_from_config(plot_configs=plot_configs,
                  dfs=xval_data,
                  num_plots_row_col_size=num_plots_row_col_size)
plt.show()

# ----
# daily break down
# ----

# heatmap scatter plot, histogram

plot_col = "norm_diff"
cmap = "bwr"

plot_col = "nll"
cmap = "YlGnBu_r"

# get the vmin/vmax values, based off of quantiles
vmin_max = {}

_ = pd.concat([DataLoader.load(v) for k, v in xval_data.items()])

if re.search("diff", plot_col):
    vmax = np.nanquantile(np.abs(_[plot_col]), q=0.99)
    vmin = -vmax
else:
    vmin, vmax = np.nanquantile(_[plot_col], q=[0.01, 0.99])

vmin_max['vmin'] = vmin
vmin_max['vmax'] = vmax

# heatmap plot template

plot_template = {
    "plot_type": "heatmap",
    # "table": "preds",
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
    # "val_col": plot_col,
    "lon_col": "pred_lon",
    "lat_col": "pred_lat",
    # "x_col": "pred_loc_x",
    # "y_col": "pred_loc_y",
    "subplot_kwargs": {"projection": "north"},
    # any additional arguments for plot_hist
    "plot_kwargs": {
        "scatter": True,
        "cmap": cmap,
        "ocean_only": True
    }
}

plot_configs = []
for k in pred_tables:

    # heat map
    c = copy.deepcopy(plot_template)
    c['table'] = k
    c['val_col'] = plot_col
    # TODO: add title
    c['plot_kwargs']['vmin'] = vmin_max['vmin']
    c['plot_kwargs']['vmax'] = vmin_max['vmax']
    c['plot_kwargs']['title'] = k
    c['plot_kwargs']['s'] = 4

    plot_configs.append(c)

    c = copy.deepcopy(plot_hist_config)
    c['table'] = k
    c['val_col'] = plot_col
    plot_configs.append(c)

# generate the plots from config
# this is really slow with plot_col='nll'?
plots_from_config(plot_configs=plot_configs,
                  dfs=xval_data,
                  num_plots_row_col_size=num_plots_row_col_size,
                  suptitle=f"All XVal Data:\n{plot_col}")
plt.show()


# ----
# increment over dates
# ----

increment_over_vals = dfs['run_details'][increment_over].unique()

image_file = re.sub("\.h5", f"_XVAL_COMPARE_{plot_col}_{pred_tables[0]}vs{pred_tables[1]}.pdf", results_file)
cprint(f"writing plots of predictions to:\n{image_file}")
with PdfPages(image_file) as pdf:

    # ---
    # plot everything
    # ---

    plot_configs = []
    for k in pred_tables:
        # heat map
        c = copy.deepcopy(plot_template)
        c['table'] = k
        c['val_col'] = plot_col
        # TODO: add title
        c['plot_kwargs']['vmin'] = vmin_max['vmin']
        c['plot_kwargs']['vmax'] = vmin_max['vmax']
        c['plot_kwargs']['title'] = f"{k}\nAll {increment_over}s"
        c['plot_kwargs']['s'] = 4

        plot_configs.append(c)

        c = copy.deepcopy(plot_hist_config)
        c['table'] = k
        c['val_col'] = plot_col
        c['plot_kwargs']['title'] = f"{k}\nval_col: {plot_col}"

        plot_configs.append(c)

    # generate the plots from config
    fig = plots_from_config(plot_configs=plot_configs,
                            dfs=xval_data,
                            num_plots_row_col_size=num_plots_row_col_size,
                            suptitle="")
    pdf.savefig(fig)

    for icv in increment_over_vals:

        cprint(icv, c="OKBLUE")

        # ---
        # create a list of plot configs
        # ---

        plot_configs = []

        load_kwargs = {
            "row_select": [{"col": increment_over, "comp": "==", "val": icv}]
        }

        plot_configs = []
        for k in pred_tables:
            # heat map
            c = copy.deepcopy(plot_template)
            c['table'] = k
            c['val_col'] = plot_col

            c['plot_kwargs']['vmin'] = vmin_max['vmin']
            c['plot_kwargs']['vmax'] = vmin_max['vmax']
            c['plot_kwargs']['title'] = f"{k}\n{icv}"
            c['plot_kwargs']['s'] = 4

            c['load_kwargs'] = {**c.get("load_kwargs", {}), **load_kwargs}

            plot_configs.append(c)

            # histogram
            c = copy.deepcopy(plot_hist_config)
            c['load_kwargs'] = {**c.get("load_kwargs", {}), **load_kwargs}

            c['table'] = k
            c['val_col'] = plot_col
            c['plot_kwargs']['title'] = f"{k}\nval_col: {plot_col}"

            plot_configs.append(c)

        # generate the plots from config
        row_select = " ".join([str(v)
                               for rs in load_kwargs['row_select']
                               for k, v in rs.items()])

        fig = plots_from_config(plot_configs=plot_configs,
                                dfs=xval_data,
                                num_plots_row_col_size=num_plots_row_col_size,
                                suptitle=row_select)
        # plt.show()
        pdf.savefig(fig)

