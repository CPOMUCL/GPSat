# generate side by comparison of hyper parameters
# - useful for comparing impact from smoothing


import re
import copy

import pandas as pd
import numpy as np

import cartopy.crs as ccrs

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from PyOptimalInterpolation import get_parent_path
from PyOptimalInterpolation.utils import cprint
from PyOptimalInterpolation.local_experts import get_results_from_h5file
from PyOptimalInterpolation.plot_utils import get_projection, plot_xy_from_results_data,\
    plot_pcolormesh_from_results_data, plot_hist_from_results_data
from PyOptimalInterpolation.dataloader import DataLoader

# -----
# helper functions
# -----


pd.set_option("display.max_columns", 200)

# layout for number of plots
# - change fig size depending on the number of plots?
# num_plots_row_col_size = {i: {"nrows": i, "ncols": 2*i, "fig_size": (15, 15)} for i in range(20)}
num_plots_row_col_size = {i+1: {"nrows": i//2+1, "ncols": 2, "fig_size": (10, (i//2+1) * 5)}
                          for i in range(20)}

# ----
# parameters
# ----

# results_file = get_parent_path("results", "xval", "cs2cpom_lead_binned_date_2019_2020_25x25km.h5")
results_file = get_parent_path("results", "WG", "G21_CS2S3_20181114_50km.h5")

# plot template - specific table, col, load_kwargs to be added for each subplot
plot_template = {
    "plot_type": "heatmap",
    # "x_col": "x",
    # "y_col": "y",
    "lon_col": "lon",
    "lat_col": "lat",
    "lat_0": 90,
    "lon_0": 0,
    "subplot_kwargs": {"projection": "north"},
    "plot_kwargs": {
        "scatter": True,
        "s": 4
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


# increment over
increment_over = "date"

# which tables to compare
# - table suffixes will be added
table_names = ["lengthscales", "kernel_variance", "likelihood_variance"]

# specify which 'flavours' of tables to be compared
table_suffixes = ["", "_SMOOTHED"]


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

select_tables = [f"{tn}{ts}"
                 for tn in table_names + ["expert_locs"]
                 for ts in table_suffixes]
dfs, oi_config = get_results_from_h5file(results_file,
                                         global_col_funcs=global_col_funcs,
                                         merge_on_expert_locations=merge_on_expert_locations,
                                         select_tables=select_tables)

# ----
# plot hyper parameters: get info needed for each plot
# ----

# used for lengthscales
dim_map = {idx: _ for idx, _ in enumerate(oi_config[0]['data']['coords_col'])}


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
        if tn == "lengthscales":
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
