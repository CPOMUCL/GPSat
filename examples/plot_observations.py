# generate plots of observation / input data
# - allow for plot_by along a single dimension, e.g. date
# - plot spatial data along side distribution

# TODO: this script needs significant tidying up!

import json
import os
import re
import time
import warnings

import pandas as pd
import numpy as np
import xarray as xr
import cartopy.crs as ccrs


# for land / ocean mask - requires
#  pip install global-land-mask
# from global_land_mask import globe

from GPSat.utils import match

from GPSat.utils import json_serializable, stats_on_vals, \
    nested_dict_literal_eval, get_config_from_sysargv, cprint
from GPSat.plot_utils import plot_pcolormesh, plot_hist
from GPSat.dataloader import DataLoader
from GPSat import get_parent_path, get_data_path
from GPSat.decorators import timer

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


pd.set_option("display.max_columns", 200)

# ----
# helper functions
# ----

@timer
def plot_wrapper(plt_df, val_col,
                 lon_col='lon',
                 lat_col='lat',
                 scatter_plot_size=2,
                 plt_where=None,
                 projection=None,
                 extent=None,
                 max_obs=1e6,
                 vmin_max=None,
                 q_vminmax=None,
                 abs_vminmax=False,
                 stats_loc=None,
                 figsize=None,
                 where_sep="\n "):


    # projection
    if projection is None:
        projection = ccrs.NorthPolarStereo()
        extent = [-180, 180, 60, 90]
    elif isinstance(projection, str):
        if re.search("north", projection, re.IGNORECASE):
            projection = ccrs.NorthPolarStereo()
            if extent is None:
                extent = [-180, 180, 60, 90]
        elif re.search("south", projection, re.IGNORECASE):
            projection = ccrs.SouthPolarStereo()
            if extent is None:
                extent = [-180, 180, -60, -90]
        else:
            raise NotImplementedError(f"projection provide as str: {projection}, not implemented")

    if figsize is None:
        figsize = (10, 5)

    fig = plt.figure(figsize=figsize)

    # get the statistics on all values
    stats_df = stats_on_vals(plt_df[val_col].values,
                             measure=val_col,
                             qs=[0.001, 0.005, 0.01, 0.05] + np.arange(0.1, 1.0, 0.1).tolist() + [0.95, 0.99, 0.995, 0.999])


    # randomly select a subset
    # WILL THIS SAVE TIME ON PLOTTING?
    if len(plt_df) > max_obs:
        len_df = len(plt_df)
        p = max_obs / len_df
        # print(p)
        # b = np.random.binomial(len_df, p=p)
        b = np.random.uniform(0, 1, len_df)
        b = b <= p
        print(f"there were too many points {len(df)}>max_obs: {max_obs}\n"
              f"selecting {100 * b.mean():.2f}% ({b.sum()}) points at random for raw data plot")
        _ = plt_df.loc[b, :]

        frac_of_obs = b.mean()
    else:
        frac_of_obs = 1.00
        _ = plt_df

    if plt_where is None:
        plt_where = []

    # figure title
    where_print = where_sep.join([" ".join([str(v.astype('datetime64[s]'))
                                        if isinstance(v, np.datetime64) else
                                        str(v)
                                        for k, v in pw.items()])
                             for pw in plt_where])
    # put data source in here?
    # f"min datetime {str(plt_df[ date_col].min())}, " \
    # f"max datetime: {str(plt_df[ date_col].max())} \n" \

    sup_title = f"val_col: {val_col}    " \
                f"where conditions:\n" + where_print
    fig.suptitle(sup_title, fontsize=10)

    nrows, ncols = 1, 2

    print("plotting pcolormesh...")
    # first plot: heat map of observations
    ax = fig.add_subplot(1, 2, 1,
                         projection=projection)

    plot_data = _[val_col].values

    # get vmin/vmax values
    if vmin_max is None:
        if not abs_vminmax:
            vmin, vmax = np.nanquantile(plot_data, q=q_vminmax)
        else:
            max_q = max(q_vminmax) if isinstance(q_vminmax, (tuple, list)) else q_vminmax

            vmax = np.nanquantile(np.abs(plot_data), q=max_q)
            vmin = -vmax
    else:
        assert len(vmin_max) == 2
        vmin, vmax = vmin_max[0], vmin_max[1]

    plt_title = f"showing: {frac_of_obs * 100:.2f}% of observations\nshowing vals in range: [{vmin:.2f} :: {vmax:.2f}]"

    plot_pcolormesh(ax=ax,
                    lon=_[lon_col].values,
                    lat=_[lat_col].values,
                    plot_data=_[val_col].values,
                    fig=fig,
                    title=plt_title,
                    vmin=vmin,
                    vmax=vmax,
                    cmap='YlGnBu_r',
                    # cbar_label=cbar_labels[midx],
                    scatter=True,
                    s=scatter_plot_size,
                    extent=extent)

    ax = fig.add_subplot(1, 2, 2)

    print("plotting hist (using all data)...")
    if stats_loc is None:
        stats_loc = (0.2, 0.8)

    plt_title = f"{val_col}"
    if q_vminmax is not None:
        plt_title += f"\nshowing values in quantile range: {q_vminmax}"

    plot_hist(ax=ax,
              data=plot_data,  # plt_df[val_col].values,
              ylabel="",
              stats_values=['mean', 'std', 'skew', 'kurtosis', 'min', 'max', 'num obs'],
              title=plt_title,
              xlabel=val_col,
              stats_loc=stats_loc,
              q_vminmax=q_vminmax)

    plt.tight_layout()

    return fig, stats_df.T

def get_config_for_plot_observations():
    config = get_config_from_sysargv()

    if config is None:
        config_file = get_parent_path("configs", "example_plot_observations.json")

        warnings.warn(f"\nconfig is empty / not provided, will just use an example config:\n{config_file}")

        with open(config_file, "r") as f:
            config = nested_dict_literal_eval(json.load(f))

        # override the defaults
        config['input_data']['source'] = get_data_path("example", "ABC.h5")
        config['output']['dir'] = get_parent_path("images", "raw_observations")
        config['output']['file'] = "raw_obs_ABC.pdf"

        # check inputs source exists
        assert os.path.exists(config['input_data']['source']), \
            f"config['input']['source']:\n{config['input']['source']}\ndoes not exists. " \
            f"to create run: python -m GPSat.read_and_store"

    return config


if __name__ == "__main__":

    # ----
    # config
    # ----

    cprint("*" * 50, "OKBLUE")
    cprint("plotting observations specified in a config file")

    config = get_config_for_plot_observations()

    comment = config.pop("comment", None)
    print(f"config 'comment':\n{comment}\n")

    print("full config:")

    print(json.dumps(json_serializable(config), indent=4))

    # ---
    # extract parameters
    # ---

    image_dir = config["output"]["dir"]
    image_file = os.path.join(image_dir, config["output"]["file"])
    if not os.path.exists(image_dir):
        print(f"creating ['output']['dir']: {image_dir}")
        os.makedirs(image_dir)

    # pop off some values, pass everything else to plot_wrapper
    plt_details = config["plot_details"]
    plt_by = plt_details.pop("by", None)

    input_config = config['input_data']

    # plot everything at once in "by" not specified
    if plt_by is None:

        df = DataLoader.load(**input_config)

        with PdfPages(image_file) as pdf:
            fig = plot_wrapper(df,
                               plt_where=input_config.get("where", []),
                               **plt_details)

            pdf.savefig(fig)

    # otherwise iterate through the data
    else:

        # determine all the unique values of 'by' in the data
        by = plt_by["col"]
        print(f"will plot by: {by}")
        # by = by if isinstance(by, list) else [by]
        assert isinstance(by, str)

        store = pd.HDFStore(input_config['source'], mode="r")
        where = input_config.get("where", [])
        if where is None:
            where = []

        # get a table iterator - to determine the blocks for by values to plot
        chunksize = 5000000
        print("getting table iterator")
        df_iter = DataLoader.data_select(store,
                                         table=input_config['table'],
                                         where=where,
                                         iterator=True,
                                         chunksize=chunksize)

        nrows = store.get_storer(input_config['table']).nrows
        print(f'number of rows in data: {nrows}\n'
              f'(max) number of iterations given chunksize:{np.ceil(nrows/chunksize)}\n'
              f'(there could be fewer if "where" supplied)')
        # ---
        # get the unique 'by' values in data
        # ---
        t0 = time.time()
        unique_bin_bys = []
        for idx, df in enumerate(df_iter):

            print(idx)
            df = DataLoader._modify_df(df=df,
                                       col_funcs=input_config.get('col_funcs', None),
                                       row_select=input_config.get('row_select', None),
                                       col_select=input_config.get('col_select', None))
            unique_bin_bys.append(df[[by]].drop_duplicates())

        unique_bin_bys = pd.concat(unique_bin_bys)
        unique_bin_bys.drop_duplicates(inplace=True)
        unique_bin_bys.sort_values(by, inplace=True)
        t1 = time.time()
        print(f"time to get unique bin_by values: {t1-t0:.2f} seconds")

        # determine the number of 'by' blocks
        block_size = plt_by.get("block_size", 1)

        tmp = unique_bin_bys[by].values.flatten()
        tmp_idx = np.arange(len(tmp))

        num_blocks = np.ceil(len(tmp) / block_size)

        min_max = []
        for i in range(int(num_blocks)):
            select = (tmp_idx // block_size) == i
            min_max.append([tmp[select].min(), tmp[select].max()])

        print(f"have: {len(min_max)} plots to generate ")
        # increment over the blocks, plotting along the way

        with PdfPages(image_file) as pdf:
            plot_count = 0

            all_stats = []
            for mm in min_max:

                plot_count += 1
                cprint("-" * 20, c="OKCYAN")
                print(f"plot_count: {plot_count}")

                min_where = {"col": by, "comp": ">=", "val": mm[0]}
                max_where = {"col": by, "comp": "<=", "val": mm[1]}

                print(f"min_where: {min_where}")
                print(f"max_where: {max_where}")

                plt_where = [min_where, max_where] + where
                # df = DataLoader.data_select(obj=store,
                #                             where=plt_where,
                #                             table=input_config['table'])

                df = DataLoader.load(source=store,
                                     where=plt_where,
                                     table=input_config['table'],
                                     row_select=input_config.get("row_select", None),
                                     col_select=input_config.get("col_select", None),
                                     col_funcs=input_config.get("col_funcs", None))

                fig, stats_df = plot_wrapper(df,
                                             plt_where=plt_where,
                                             **plt_details)

                stats_df[f'min_{by}'] = mm[0]
                stats_df[f'max_{by}'] = mm[1]

                all_stats.append(stats_df)

                pdf.savefig(fig)

            all_stats = pd.concat(all_stats)

            # HACK: convert columns from object
            for c in all_stats:
                if str(all_stats[c].dtype) =='object':
                    try:
                        all_stats[c] = all_stats[c].astype(float)
                    except Exception as e:
                        print(c,e)

            # TODO: plot stats overtime
            stats_out = re.sub("\.pdf$", "_stats.csv", image_file)
            stats_out_sum = re.sub("\.pdf$", "_stats_summary.csv", image_file)
            print(f"writing stats from each plot to:\n{stats_out}")
            all_stats.to_csv(stats_out, index=False)

            cprint("summary of stats:", c="OKCYAN")
            desc = all_stats.describe()
            print(desc)
            cprint(f"writing summary of stats to:\n{stats_out_sum}", c="OKGREEN")
            desc.to_csv(stats_out_sum, index=False)

            #
            # fig, ax = plt.subplots()
            # q_cols = [c for c in all_stats.columns if re.search("^q", c)]
            # for q in q_cols:
            #     ax.plot(all_stats[f'min_{by}'].values, all_stats[q].values, label=None)
            # ax.set_title("Quantiles")
            # # ax.get_xticklabels()
            # # ax.set_xticklabels(rotation=45, labels=ax.get_xticklabels())
            # plt.show()

        store.close()


    cprint("-" * 50, c="OKGREEN")
    cprint(f"finished, results written to:\n{image_file}", c="OKGREEN")

