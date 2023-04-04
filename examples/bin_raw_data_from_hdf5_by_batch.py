# Bin Raw Data - in batches: to avoid memory overflow
# 
# select subset / remove outliers before binning
#

# need to add parent directory to sys.path...
import sys
import os
import json
import warnings

import pandas as pd
import numpy as np
import time

import matplotlib.pyplot as plt
import cartopy.crs as ccrs

from IPython.display import display
from PyOptimalInterpolation import get_data_path
from PyOptimalInterpolation.dataprepper import DataPrep
from PyOptimalInterpolation.dataloader import DataLoader
from PyOptimalInterpolation.utils import stats_on_vals, EASE2toWGS84_New
from PyOptimalInterpolation.plot_utils import plot_pcolormesh, plot_hist

from PyOptimalInterpolation import get_parent_path
from PyOptimalInterpolation.utils import json_serializable
import re

from PyOptimalInterpolation.utils import get_config_from_sysargv
pd.set_option('display.max_columns', 200)

# ---
# helper function
# ---

def plot_wrapper(plt_df, val_col,
                 lon_col='lon',
                 lat_col='lat',
                 date_col='date',
                 scatter_plot_size=2,
                 plt_where=None,
                 projection=None,
                 extent=None):


    # projection
    if projection is None:
        projection = ccrs.NorthPolarStereo()
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


    figsize = (10, 5)
    fig = plt.figure(figsize=figsize)

    # randomly select a subset
    # WILL THIS SAVE TIME ON PLOTTING?
    if len(plt_df) > 1e6:
        len_df = len(plt_df)
        p = 1e6 / len_df
        # print(p)
        # b = np.random.binomial(len_df, p=p)
        b = np.random.uniform(0, 1, len_df)
        b = b <= p
        print(f"there were too many points {len(df)}\n"
              f"selecting {100 * b.mean():.2f}% ({b.sum()}) points at random for raw data plot")
        _ = plt_df.loc[b, :]
    else:
        _ = plt_df

    if plt_where is None:
        plt_where = []

    # figure title
    where_print = "\n ".join([" ".join([str(v) for k, v in pw.items()])
                             for pw in plt_where])
    # put data source in here?
    sup_title = f"val_col: {val_col}\n" \
                f"min datetime {str(plt_df[ date_col].min())}, " \
                f"max datetime: {str(plt_df[ date_col].max())} \n" \
                f"where conditions:\n" + where_print
    fig.suptitle(sup_title, fontsize=10)

    nrows, ncols = 1, 2

    print("plotting pcolormesh...")
    # first plot: heat map of observations
    ax = fig.add_subplot(1, 2, 1,
                         projection=projection)

    plot_pcolormesh(ax=ax,
                    lon=_[lon_col].values,
                    lat=_[lat_col].values,
                    plot_data=_[val_col].values,
                    fig=fig,
                    # title=plt_title,
                    # vmin=vmin,
                    # vmax=vmax,
                    cmap='YlGnBu_r',
                    # cbar_label=cbar_labels[midx],
                    scatter=True,
                    s=scatter_plot_size,
                    extent=extent)

    ax = fig.add_subplot(1, 2, 2)

    print("plotting hist (using all data)...")
    plot_hist(ax=ax,
              data=_[val_col].values,  # plt_df[val_col].values,
              ylabel="",
              stats_values=['mean', 'std', 'skew', 'kurtosis', 'min', 'max', 'num obs'],
              title=f"{val_col}",
              xlabel=val_col,
              stats_loc=(0.2, 0.8))

    plt.tight_layout()


def bin_wrapper(df, col_funcs=None, print_stats=True,  **bin_config):
    # simple function to add columns, generate stats, and bin values (after some row selection)

    # with pd.HDFStore(input_file, mode='r') as store:
    #     df = store.select(table, where=where)

    if print_stats:
        print("head of data:")
        print(df.head(3))

    # --
    # (optionally) add columns

    DataLoader.add_cols(df, col_func_dict=col_funcs, verbose=4)

    print("*" * 20)
    print("summary / stats table on metric (use for trimming) - prior to row_select")

    val_col = bin_config['val_col']
    vals = df[val_col].values
    stats_df = stats_on_vals(vals=vals, name=val_col,
                             qs=[0.001, 0.01, 0.05] + np.arange(0.1, 1.0, 0.1).tolist() + [0.95, 0.99, 0.999])

    if print_stats:
        # print(stats_df)
        display(stats_df)

    # get a Dataset of binned data
    print("using row_select")
    print(bin_config.get("row_select", None))

    ds_bin = DataPrep.bin_data_by(df=df, **bin_config)

    return ds_bin, stats_df


if __name__ == "__main__":

    from PyOptimalInterpolation.utils import nested_dict_literal_eval

    # TODO: clean up and add comments
    # TODO: extend comment in default / example config
    # TODO: allow for binned data to be store as zarr/netcdf
    # TODO: clean up this script - move sections into methods in DataPrep(?)
    # TODO: require if using batch the bin_by values exist in the data?
    # TODO: chunksize should be an parameter in 'input'
    # TODO: review / refactor how the output statistics are printed to the screen - particularly for batch

    # ---
    # Config / Parameters
    # ---

    # read in config
    config = get_config_from_sysargv()

    # assert config is not None, f"config not provide"
    if config is None:
        config_file = get_parent_path("configs", "example_bin_raw_data.json")
        warnings.warn(f"\nconfig is empty / not provided, will just use an example config:\n{config_file}")
        with open(config_file, "r") as f:
            config = nested_dict_literal_eval(json.load(f))

        # override the defaults
        config['input']['file'] = get_parent_path("data", "example", "ABC.h5")
        config['output']['file'] = get_parent_path("data", "example", "ABC_binned.h5")

        assert os.path.exists(config['input']['file'] ), \
            f"config['input']['file']:\n{config['input']['file']}\ndoes not exists. " \
            f"to create run: python -m PyOptimalInterpolation.read_and_store"

    # ---
    # extract parameters from config
    # ---

    # make a copy of the config - so can store as attribute
    org_config = config.copy()

    # pop out input and output_file
    input_info = config['input']
    output_file = config['output']['file']
    comment = config.pop("comment", None)

    col_funcs = config.pop("col_funcs", None)

    # parameters for binning
    bin_config = config['bin_config']

    print(f"comment provided in input config:\n{comment}")

    try:
        run_info = DataLoader.get_run_info(script_path=__file__)
    except NameError as e:
        run_info = DataLoader.get_run_info()

    input_file = input_info['file']
    table = input_info.get('table', "data")
    where = input_info.get("where", [])
    batch = input_info.get("batch", False)
    col_funcs = input_info.get("col_funcs", None)

    # connect to HDF store
    print("reading from hdf5 files")
    store = pd.HDFStore(input_file, mode='r')

    # --
    # get the raw data configuration - if it exists
    # - to know where (raw) data came from, will store with output
    # --
    try:
        raw_data_config = store.get_storer(table).attrs['config']
        # print(json.dumps(raw_data_config, indent=4))
    except Exception as e:
        print(e)
        warnings.warn("issue getting raw_data_config? it should exists in attrs")
        raw_data_config = None


    if not batch:

        # read in all the data
        df = DataLoader.data_select(obj=store,
                                    where=where,
                                    table=table)

        # add / modify columns
        DataLoader.add_cols(df, col_func_dict=col_funcs)

        # bin the data
        ds_bin, stats_df = bin_wrapper(df, col_funcs=None,  **bin_config)

        # convert to DataFrame
        df_bin = ds_bin.to_dataframe().dropna().reset_index()

        # add any additional extra columns
        DataLoader.add_cols(df_bin, col_func_dict=config.get('add_output_cols', None))

        # write to file
        print(f"writing results to hdf5 file:\n{output_file}")
        with pd.HDFStore(output_file, mode="w") as store_out:

            out_table = config['output'].get("table", bin_config['val_col'])
            print(f"writing to table: {out_table}")
            store_out.put(key=out_table,
                          value=df_bin,
                          append=True,
                          format='table',
                          data_columns=True)

            store_attrs = store_out.get_storer(out_table).attrs

            # include configs
            store_attrs['raw_data_config'] = raw_data_config
            store_attrs['bin_config'] = config
            store_attrs['run_info'] = run_info

    else:
        print("reading data in by batches")
        bin_by = bin_config['by_cols']

        storer = store.get_storer(table)

        # --
        # determine which columns are required to get the unique bin_by
        # --

        # it's possible the bin_by values will be added later through application of col_funcs
        # - so the columns used in col_funcs must also be fetched
        # get_cols = bin_by
        # tmp_col_func = {}


        # if all the bin_by columns are not in original data - determine how to get them
        missing_bin_by = np.array(bin_by)[~np.in1d(bin_by, storer.attrs['values_cols'])]
        assert len(missing_bin_by) == 0, f"the following bin_by columns: {missing_bin_by} are missing in data, " \
                                         f"they are required to used 'batch=True'"

        # if len(missing_bin_by):
        #     get_cols += get_cols_from_col_funcs(col_funcs)
        #
        #     get_cols = list(set(get_cols))


        df_iter = DataLoader.data_select(store,
                                         table=table,
                                         where=where,
                                         # columns=get_cols,
                                         iterator=True,
                                         chunksize=5000000)
        # df_iter = store.select(table, where=where, columns=bin_by, iterator=True, chunksize=5000000)


        # get the unique values to bin by
        # TODO: here determine ahead of time the number of rows, and then number of iterations to go through
        # TODO: make this into a function, add timer
        t0 = time.time()
        unique_bin_bys = []


        for idx, df in enumerate(df_iter):
            # TODO: could just get everything, apply column funcs and take what is needed
            #  - shouldn't be a memory issue if chuncksize isn't too big
            # TODO: should only apply relevant column funcs
            # NOTE: might be applying too many col funcs
            DataLoader.add_cols(df, col_func_dict=col_funcs)

            # modify columns

            unique_bin_bys.append(df[bin_by].drop_duplicates())

        unique_bin_bys = pd.concat(unique_bin_bys)
        unique_bin_bys.drop_duplicates(inplace=True)
        t1 = time.time()
        print(f"time to get unique bin_by cols: {t1-t0:.2f} seconds")

        unique_bin_bys.sort_values(bin_by, inplace=True)

        # read the data in chunks
        # TODO: allow for
        df_bin_all = []
        stats_all = []
        for idx, row in unique_bin_bys.iterrows():
            print("-"*10)
            print(row)

            # select data - from store, include a where for current bin_by values
            # NOTE: 'date' only where selection can be very fast (?)
            row_where = [{"col": k, "comp": "==", "val": v} for k, v in row.to_dict().items()]
            df = DataLoader.data_select(obj=store,
                                        where=where + row_where,
                                        table=table)

            # add / modify columns
            DataLoader.add_cols(df, col_func_dict=col_funcs)

            # plot values - debugging (?)
            # NOTE: this needs to be commented out as values are HARDCODED
            # plt_df = DataLoader.data_select(df, where=bin_config.get('row_select', None))
            # plot_wrapper(plt_df=plt_df,
            #              val_col=bin_config['val_col'],
            #              lon_col='lon_20_ku',
            #              lat_col='lat_20_ku',
            #              date_col='datetime',
            #              scatter_plot_size=2,
            #              plt_where=where,
            #              projection="south")
            # plt.show()

            ds_bin, stats_df = bin_wrapper(df, col_funcs=None, print_stats=False, **bin_config)

            # convert dataset to DataFrame
            df_bin = ds_bin.to_dataframe().dropna().reset_index()

            # merge on the bin_by info to the stats
            stats_df = stats_df.T
            row_df = row.to_frame().T
            stats_df.set_index(row_df.index, inplace=True)
            stats_all.append(pd.concat([row_df, stats_df], axis=1))

            # TODO: add columns to output
            DataLoader.add_cols(df_bin, col_func_dict=config.get('add_output_cols', None))

            df_bin_all.append(df_bin)

        store.close()
        out = pd.concat(df_bin_all)

        stats_all = pd.concat(stats_all)

        # plot quantiles by stats bin by
        q_cols = [c for c in stats_all.columns if re.search("^q", c)]

        # try:
        #     # TODO: allow for more robust plotting of quantiles by bin-by
        #     #  - merge bin_by values together?
        #     #  - plot only for a main bin_by axis - i.e. dates, and generate separate plots for other
        #     # TODO: also plot min/max - in absolute terms? color code
        #     # by = stats_all[bin_by].copy(True)
        #     # convert to string
        #     # for c in by.columns:
        #     #     by[c] = by[c].values.astype(str)
        #     # stats_all['by'] = stats_all[bin_by]
        #
        #     for q in q_cols:
        #         plt.plot(stats_all[bin_by].values, stats_all[q].values, label=None)
        #     plt.title("Quantiles")
        #     plt.xticks(rotation=45)
        #
        #     plt.tight_layout()
        #     plt.show()
        # except Exception as e:
        #     print(f"in plotting the quantiles by bin_by: {bin_by}, there was an error:")
        #     print(e)

        # write results to file
        with pd.HDFStore(output_file, mode="w") as store_out:

            out_table = config['output'].get("table", bin_config['val_col'])
            print(f"writing to table: {out_table}")
            store_out.put(key=out_table,
                          value=out,
                          append=True,
                          format='table',
                          data_columns=True)

            store_attrs = store_out.get_storer(out_table).attrs

            store_attrs['raw_data_config'] = raw_data_config
            store_attrs['bin_config'] = config
            store_attrs['run_info'] = run_info

        # add current batch to the run_batches attribute
        # store_attrs = store.get_storer(table).attrs

    # ---
    # write config to config dir -
    # ---

    config_file = get_parent_path("configs",
                                  re.sub("\..*$", ".json", os.path.basename(output_file) ))

    if os.path.exists(config_file):
        print(f"config_file:\n{config_file}\nalready exists, not going to over write")
        print("config contents are:\n\n")
        json.dumps(json_serializable(org_config), indent=4)
    else:
        print(f"writing config file to configs dir:\n{config_file}")
        with open(config_file, "w") as f:
            json.dump(json_serializable(org_config), f, indent=4)

    print(f"binning finished, output file is:\n{output_file}\nin table: '{out_table}'")

    # def get_cols_from_col_funcs(cf):
    #     out = []
    #     for k, v in cf.items():
    #         if 'col_args' in v:
    #             out += (v['col_args'] if isinstance(v['col_args'], list) else [v['col_args']])
    #
    #         if 'col_kwargs' in v:
    #             for kk, vv in v['col_kwargs'].items():
    #                 out += (vv if isinstance(vv, list) else [vv])
    #
    #     return list(set(out))
    #
