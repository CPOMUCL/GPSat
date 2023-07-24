# generate side by comparison of hyper parameters
# - useful for comparing impact from smoothing


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

# -----
# helper functions
# -----


pd.set_option("display.max_columns", 200)

# ----
# parameters
# ----

# TODO: implement adding the difference correctly
include_diff = False
assert include_diff == False

# NOTE: this script

# results_file = get_parent_path("results", "xval", "cs2cpom_lead_binned_date_2019_2020_25x25km.h5")
# results_file = get_parent_path("results", "WG", "G21_CS2S3_20181114_50km.h5")
# results_file = get_parent_path("results", "SGPR_vs_GPR_cs2s3cpom_2019-2020_25km.h5")
# results_file = get_parent_path("results", "cs2s3cpom_spaced_local_experts.h5")
# results_file = get_parent_path("results", "SGPR_gpod_lead_elev_10x10km.h5")
# results_file = get_parent_path("results", "SGPR_gpod_freeboard_10x10km.h5")
results_file = get_parent_path("results", "XVAL_gpod_freeboard_10x10km.h5")


# specify which 'flavours' of tables to be compared
# table_suffixes = ["_GPR", "_SGPR"]
# table_suffixes = ["_GPR", "_GPR_SMOOTHED"]
# table_suffixes = ["_not_spaced", "_spaced"]
# table_suffixes = ["_GPR_SMOOTHED_GPR", "_SMOOTHED_SGPR"]
table_suffixes = ["_SGPR", "_SGPR_SMOOTHED"]

# which tables to compare
# - table suffixes will be added
table_names = ["lengthscales", "kernel_variance", "likelihood_variance"]

# what column to increment over?
# increment_over = "date"
increment_over = "t"


# layout for number of plots

# num_plots_row_col_size = {i: {"nrows": i, "ncols": 2*i, "fig_size": (15, 15)} for i in range(20)}


# plot template - specific table, col, load_kwargs to be added for each subplot
plot_template = {
    "plot_type": "heatmap",
    "x_col": "x",
    "y_col": "y",
    # "lon_col": "lon",
    # "lat_col": "lat",
    "lat_0": 90,
    "lon_0": 0,
    "subplot_kwargs": {"projection": "north"},
    "plot_kwargs": {
        "scatter": False,
        # "s": 4
    }
}

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


# image_file = re.sub("\.h5$", "_plot_summary.pdf", results_file)
global_col_funcs = None
# global_col_funcs = {"date":
#     {
#         "func": "lambda t: t.astype('datetime64[D]')",
#         "col_args": "t"
#     }
# }
merge_on_expert_locations = False


# ----
# plot count -> layout
# ----
# - change fig size depending on the number of plots?

plot_per_row = 3 if include_diff else 2
# each key is the total number of plots
num_plots_row_col_size = {i+1: {"nrows": i//plot_per_row+1,
                                "ncols": plot_per_row,
                                "fig_size": ((plot_per_row * 6), (i//plot_per_row+1) * 6)}
                          for i in range(20)}


# -----
# read in results data - get dict of DataFrames
# -----

# select_tables = [f"{tn}{ts}"
#                  for tn in table_names + ["expert_locs", "oi_config"]
#                  for ts in table_suffixes]


tmp = []
for ts in table_suffixes:
    cprint(ts, c="OKGREEN")
    dfs, oi_config = get_results_from_h5file(results_file,
                                             global_col_funcs=global_col_funcs,
                                             merge_on_expert_locations=merge_on_expert_locations,
                                             select_tables=table_names + ["expert_locs", "oi_config"],
                                             table_suffix=ts,
                                             add_suffix_to_table=True)
    tmp.append(dfs)

# TODO: handle oi_config, put in a dict with key = table_suffixes
dfs = reduce(lambda x, y: {**x, **y}, tmp)

# ----
# get diff tables
# ----

# TODO: add differences

# if include_diff:
#     for tn in table_names:
#
#         tmp = []
#         for ts in table_suffixes:
#             tmp.append(dfs[f'{tn}{ts}'])
#
#         # this works if table name same as column name
#         merge_col = [i for i in tmp[0].columns if i != tn]
#
#         dif = tmp[0].merge(tmp[1],
#                            how='inner',
#                            on=merge_col,
#                            suffixes=table_suffixes)
#
#         dif[f"{tn}_ABS_DIFF"] = dif[f'{tn}{table_suffixes[0]}'] - dif[f'{tn}{table_suffixes[1]}']
#

# TODO: include normalised differences
# diff_table['_dim_0'] = 0
# dfs['diff_table'] = diff_table

# ----
# plot hyper parameters: get info needed for each plot
# ----

# used for lengthscales
try:
    dim_map = {idx: _ for idx, _ in enumerate(oi_config[-1]['data']['coords_col'])}
except Exception as e:
    print(f"in trying to create a dim_map (for lengthscales) got: {e}")
    # create a dummy dim_map
    dim_map = {i:i for i in range(100)}

sup_title = "hyper params"

# create a dict, which will have key: table _ dim
# value will be a list of length two, for each table_suffix
plt_info = {}
# increment over the table
for tn in table_names:
    # get the dimensions for each
    table_dims = {}
    for ts in table_suffixes:
        table_dims[f"{tn}{ts}"] = dfs[f"{tn}{ts}"]["_dim_0"].unique()

    t0 = f"{tn}{table_suffixes[0]}"
    t1 = f"{tn}{table_suffixes[1]}"
    assert np.all(table_dims[t0] == table_dims[t1])

    # for hyper parameters the column is the table name
    col = tn

    # increment over the dims - create the row select
    for d in table_dims[t0]:
        row_select = {"col": "_dim_0", "comp": "==", "val": d}
        # get the vmin/vmax for the
        tmp = []
        for k in [t0, t1]:
            tmp.append(DataLoader.load(dfs[k], row_select=row_select))
        _ = pd.concat(tmp)

        vmin, vmax = np.nanquantile(_[col], q=[0.01, 0.99])

        load_kwargs = {
            "row_select": [row_select]
        }

        # plot title
        pt0, pt1 = t0, t1
        if re.search("^lengthscales", tn):
            pt0 = f"{t0} - {dim_map[d]}"
            pt1 = f"{t1} - {dim_map[d]}"

        tmp = {
            "table": t0,
            "val_col": col,
            "load_kwargs": load_kwargs,
            "plot_kwargs": {"vmin": vmin, "vmax": vmax, "title": pt0}
        }
        tmp1 = copy.deepcopy(tmp)
        tmp1['table'] = t1
        tmp1["plot_kwargs"]["title"] = pt1

        # store
        plt_info[f"{tn} - {d}"] = [tmp, tmp1]


# ----
# increment over values (dates?) - plotting each
# ----

image_file = re.sub("\.h5",
                    f"_HYPER_PARAM_COMPARE_{table_suffixes[0]}vs{table_suffixes[1]}.pdf",
                    results_file)
cprint(f"writing plots of hyper parameters to:\n{image_file}", "OKGREEN")

increment_over_vals = dfs[f"{table_names[0]}{table_suffixes[0]}"][increment_over].unique()

with PdfPages(image_file) as pdf:

    for icv in increment_over_vals:
        print("-" * 10)
        print(f"{increment_over}: {icv}")
        # ---
        # create a list of plot configs
        # ---

        plot_configs = []

        for k, v in plt_info.items():

            for _ in v:
                # copy content from plt_info into plot template
                _ = copy.deepcopy(_)
                tmp = copy.deepcopy(plot_template)
                for c in ['table', 'val_col']:
                    tmp[c] = _[c]

                # add the current (increment over value) to the row_select
                increment_select = {"col": increment_over, "comp": "==", "val": icv}
                _["load_kwargs"]["row_select"].append(increment_select)

                for c in ["load_kwargs", "plot_kwargs"]:
                    tmp[c] = {**tmp.get(c, {}), **_[c]}

                plot_configs.append(tmp)

        # ---
        # generate plots
        # ---

        plt_idx = 1

        rcs = num_plots_row_col_size[len(plot_configs)]
        nrows, ncols, fig_size = rcs['nrows'], rcs['ncols'], rcs['fig_size']
        fig = plt.figure(figsize=fig_size)

        fig.suptitle(f"{sup_title}\n{increment_over} == {icv}", y=1.0)

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

cprint(f"wrote plots of hyper parameters to:\n{image_file}", "OKBLUE")
