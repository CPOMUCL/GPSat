# generate side by comparison of predictions
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
from GPSat.utils import cprint, get_weighted_values
from GPSat.local_experts import get_results_from_h5file
from GPSat.plot_utils import plots_from_config

from GPSat.dataloader import DataLoader


# -----
# helper functions
# -----

pd.set_option("display.max_columns", 200)

# layout for number of plots
# - change fig size depending on the number of plots?
# num_plots_row_col_size = {i+1: {"nrows": i//2+1, "ncols": 2, "fig_size": (10, (i//2+1) * 5)}
#                           for i in range(20)}

# go wider if including differences
plot_per_row = 3
num_plots_row_col_size = {i+1: {"nrows": i//plot_per_row+1, "ncols": plot_per_row, "fig_size": ((plot_per_row * 6), (i//plot_per_row+1) * 6)}
                          for i in range(20)}

# ----
# parameters
# ----

# results_file = get_parent_path("results", "xval", "cs2cpom_lead_binned_date_2019_2020_25x25km.h5")
# results_file = get_parent_path("results", "WG", "G21_CS2S3_20181114_50km.h5")
results_file = get_parent_path("results", "SGPR_vs_GPR_cs2s3cpom_2019-2020_25km.h5")

# plot template - specific table, col, load_kwargs to be added for each subplot
plot_template = {
    "plot_type": "heatmap",
    "x_col": "pred_loc_x",
    "y_col": "pred_loc_y",
    # "lon_col": "lon",
    # "lat_col": "lat",
    "lat_0": 90,
    "lon_0": 0,
    "subplot_kwargs": {"projection": "north"},
    "plot_kwargs": {
        "scatter": False,
        "ocean_only": True

    }
}


plot_col_map = {
    "preds": "f*"
}

# weight function
weighted_values_kwargs = {
        "ref_col": ["pred_loc_x", "pred_loc_y", "pred_loc_t"],
        "dist_to_col": ["x", "y", "t"],
        "val_cols": ["f*", "f*_var"],
        "weight_function": "gaussian",
        "lengthscale": 200_000
    }

# col_funcs to apply after weighted combination is taken
col_funcs = {
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
        },
        "date": {
            # NOTE: may need a differ
            # "func": "lambda x: (24 * 60 * 60).astype('datetime64[s]')",
            "func": "lambda x: x.astype('datetime64[D]')",
            "col_args": "pred_loc_t"
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
table_names = ["preds"]
sup_title = " ".join(table_names)

# specify which 'flavours' of tables to be compared
table_suffixes = ["_GPR", "_SGPR"]


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

# select_tables = [f"{tn}{ts}"
#                  for tn in table_names + ["expert_locs", "oi_config"]
#                  for ts in table_suffixes]


tmp = []
oi_configs = {}
for ts in table_suffixes:
    cprint(ts, c="OKGREEN")
    dfs, oi_config = get_results_from_h5file(results_file,
                                             global_col_funcs=global_col_funcs,
                                             merge_on_expert_locations=merge_on_expert_locations,
                                             select_tables=table_names + ["expert_locs", "oi_config", "run_details"],
                                             table_suffix=ts,
                                             add_suffix_to_table=True)
    oi_configs[ts] = oi_config
    tmp.append(dfs)

# TODO: handle oi_config, put in a dict with key = table_suffixes
dfs = reduce(lambda x, y: {**x, **y}, tmp)

# -----
# get weighted combinations of predictions
# -----


assert isinstance(weighted_values_kwargs, dict)

for table in table_names:
    for ts in table_suffixes:
        plt_data = dfs[f"{table}{ts}"]
        plt_data = get_weighted_values(df=plt_data, **weighted_values_kwargs)

        plt_data = DataLoader.load(source=plt_data, col_funcs=col_funcs)
        # HACK: to plotting gets split out by unique _dim_0 values, review
        plt_data['_dim_0'] = 0
        dfs[f"{table}{ts}"] = plt_data

# ---
# get the difference
# ---

# this will break if there's not exactly two tables
t1, t2 = [f"{t}{ts}" for t in table_names for ts in table_suffixes]

merge_col = weighted_values_kwargs['ref_col']
val_cols = weighted_values_kwargs['val_cols']

diff_table = dfs[t1][merge_col + val_cols].merge(dfs[t2][merge_col + val_cols],
                                                 on=merge_col,
                                                 how='inner',
                                                 suffixes=table_suffixes)
# apply column functions
diff_table = DataLoader.load(source=diff_table, col_funcs=col_funcs)

# get the differences
for vc in val_cols:
    diff_table[f"{vc}_diff"] = diff_table[f"{vc}{table_suffixes[0]}"] - diff_table[f"{vc}{table_suffixes[1]}"]

# TODO: include normalised differences
diff_table['_dim_0'] = 0
dfs['diff_table'] = diff_table

# ----
# plot hyper parameters: get info needed for each plot
# ----

# dimension map used for lengthscales
# -
dim_map = {idx: _ for idx, _ in enumerate(oi_configs[table_suffixes[0]][0]['data']['coords_col'])}


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
    col = plot_col_map[tn]

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

        # first plot
        tmp = {
            "table": t0,
            "val_col": col,
            "load_kwargs": load_kwargs,
            "plot_kwargs": {"vmin": vmin, "vmax": vmax, "title":  f"{pt0} {col}"}
        }
        # second plot
        tmp1 = copy.deepcopy(tmp)
        tmp1['table'] = t1
        tmp1["plot_kwargs"]["title"] = f"{pt1} {col}"

        # difference plot
        tmp2 = copy.deepcopy(tmp)
        tmp2['table'] = "diff_table"
        tmp2["val_col"] = f'{col}_diff'
        tmp2["plot_kwargs"]["title"] = "difference"
        vmax = np.nanquantile(np.abs(dfs[tmp2['table']][tmp2['val_col']]), q=0.975)
        vmin = -vmax
        tmp2["plot_kwargs"]['vmin'] = vmin
        tmp2["plot_kwargs"]['vmax'] = vmax
        tmp2['plot_kwargs']["cmap"] = "bwr"
        # store
        # plt_info[f"{tn} - {d}"] = [tmp, tmp1]
        plt_info[f"{tn} - {d}"] = [tmp, tmp1, tmp2]


# ----
# increment over values (dates?) - plotting each
# ----

image_file = re.sub("\.h5",
                    f"_PREDS_{table_suffixes[0]}vs{table_suffixes[1]}.pdf",
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

        st = sup_title + f"\n{increment_over} == {icv}"

        fig = plots_from_config(plot_configs, dfs, num_plots_row_col_size, st)
        pdf.savefig(fig)

cprint(f"wrote plots of hyper parameters to:\n{image_file}", "OKBLUE")
