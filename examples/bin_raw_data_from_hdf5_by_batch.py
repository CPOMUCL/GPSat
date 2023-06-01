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
from dataclasses import dataclass
from typing import List, Dict, Tuple, Union, Type

import matplotlib.pyplot as plt
import cartopy.crs as ccrs

from IPython.display import display
from PyOptimalInterpolation import get_data_path
from PyOptimalInterpolation.dataprepper import DataPrep
from PyOptimalInterpolation.dataloader import DataLoader
from PyOptimalInterpolation.utils import stats_on_vals, cprint, json_serializable, nested_dict_literal_eval
from PyOptimalInterpolation.plot_utils import plot_pcolormesh, plot_hist

from PyOptimalInterpolation import get_parent_path
import re

from PyOptimalInterpolation.utils import get_config_from_sysargv
pd.set_option('display.max_columns', 200)

# ---
# helper function
# ---


class BinData:

    def __init__(self,
                 input: Union[dict, None] = None,
                 bin_config: Union[dict, None] = None,
                 output: Union[dict, None] = None,
                 # comment: Union[str, None] = None,
                 add_output_cols: Union[dict, None] = None):

        # store the inputs as a dict
        self.config = self._method_inputs_to_config(locs=locals(),
                                                    code_obj=self.__init__.__code__)

        # store 'config' attribute in the raw data (to be binned) here
        self.raw_data_config = None

        self.input = input
        self.bin_config = bin_config
        self.output = output
        # this could be put into output...
        self.add_output_cols = add_output_cols

        assert self.input is not None, "input is None, must be supplied (as dict)"
        assert self.bin_config is not None, "bin_config is None, must be supplied (as dict)"
        assert self.output is not None, "output is None, must be supplied (as dict)"

        # TODO: determine what __file__ will be when called from different scripts
        try:
            run_info = DataLoader.get_run_info(script_path=__file__)
        except NameError as e:
            run_info = DataLoader.get_run_info()
        self.run_info = run_info

        # TODO: check min contents of input_info, bin_config and output
        self.output_file = self.output['file']

    def _method_inputs_to_config(self, locs, code_obj, verbose=False):
        # this function aims to take the arguments of a function/method and store them in a dictionary
        # copied from LocalExpertOI
        # TODO: validate this method returns expected values - i.e. the arguments provided to a function
        # TODO: look into making this method into a decorator

        # code_obj: e.g. self.<method>.__code__
        # locs: locals()
        config = {}
        # +1 to include kwargs
        # for k in range(code_obj.co_argcount + 1):
        #   var = code_obj.co_varnames[k]
        for var in code_obj.co_varnames:

            if var == "self":
                continue
            elif var == "kwargs":
                for kw, v in locs[var].items():
                    config[kw] = v
            else:
                # HACK: to deal with 'config' was unexpectedly coming up - in set_model only
                try:
                    config[var] = locs[var]
                except KeyError as e:
                    if verbose:
                        print(f"KeyError on var: {var}\n", e, "skipping")
        return json_serializable(config)


    @staticmethod
    def bin_wrapper(df, col_funcs=None, print_stats=True, **bin_config):
        # simple function to add columns, generate stats, and bin values (after some row selection)

        if print_stats:
            print("head of data:")
            print(df.head(3))

        cprint("bin config provided:", c="OKBLUE")
        cprint(json.dumps(bin_config, indent=4), c="OKGREEN")

        # --
        # (optionally) add columns

        if col_funcs is not None:
            cprint("adding / modifying columns with:", c="OKBLUE")
            cprint(json.dumps(col_funcs, indent=4), c="OKGREEN")
            DataLoader.add_cols(df, col_func_dict=col_funcs, verbose=4)

        # --
        # stats on values
        # --

        val_col = bin_config['val_col']
        vals = df[val_col].values

        cprint(f"getting stats on column: {val_col} from data", c="OKBLUE")
        stats_df = stats_on_vals(vals=vals, name=val_col,
                                 qs=[0.001, 0.01, 0.05] + np.arange(0.1, 1.0, 0.1).tolist() + [0.95, 0.99, 0.999])

        if print_stats:
            print("*" * 20)
            print("summary / stats table on metric (use for trimming) - prior to row_select")
            # print(stats_df)
            display(stats_df)

        # get a Dataset of binned data
        # print("using row_select (on binnned data)")
        # print(bin_config.get("row_select", None))

        cprint("binning data...", c="OKBLUE")
        ds_bin = DataPrep.bin_data_by(df=df, **bin_config)

        return ds_bin, stats_df

    def bin_data_all_at_once(self):

        # read in all the data
        source = self.input_info.get('file', self.input_info.get('source', None))
        table = self.input_info.get("table", "data")

        df = DataLoader.load(source=source,
                             table=table,
                             where=self.input_info.get("where", None),
                             col_funcs=self.input_info.get("col_funcs", None),
                             col_select=self.input_info.get("col_select", None),
                             row_select=self.input_info.get("row_select", None))

        # check if there is config for the raw data store as an attribute, if possible
        self.raw_data_config = DataLoader.get_attribute_from_table(source=source,
                                                                   table=table,
                                                                   attribute_name='config')

        # bin the data
        ds_bin, stats_df = self.bin_wrapper(df, col_funcs=None,  **self.bin_config)

        # convert to DataFrame
        df_bin = ds_bin.to_dataframe().dropna().reset_index()

        # add any additional extra columns
        DataLoader.add_cols(df_bin,
                            col_func_dict=self.add_output_cols)

        # TODO: writing to file should be done separately
        # write to file
        # self.write_dataframe_to_table(df_bin)

        return df_bin, stats_df

    def bin_data_by_batch(self, chunksize=5000000):
        print("reading data in by batches")

        # load_by and bin by can difference
        # e.g. could load by date and bin by track
        load_by = self.input.get("load_by", self.bin_config['by_cols'])
        if isinstance(load_by, str):
            load_by = [load_by]

        # require the load_by values are all in bin by_cols
        for lb in load_by:
            assert lb in self.bin_config['by_cols'], \
                f"load_by value: {lb} is not in bin by_cols: {self.bin_config['by_cols']}"

        print(f"load_by: {load_by}")
        # storer = store.get_storer(table)

        # --
        # determine which columns are required to get the unique load_by
        # --

        # if all the load_by columns are not in original data - determine how to get them
        # try:
        #     missing_load_by = np.array(load_by)[~np.in1d(load_by, storer.attrs['values_cols'])]
        #     assert len(missing_load_by) == 0, f"the following load_by columns: {missing_load_by} are missing in data, " \
        #                                      f"they are required to used 'batch=True'"
        # except KeyError as e:
        #     pass # if missing_load_by should error later
        # if len(missing_load_by):
        #     get_cols += get_cols_from_col_funcs(col_funcs)
        #
        #     get_cols = list(set(get_cols))

        # ---
        # find unique load_by values in data
        # ---

        source = self.input.get('file', self.input.get('source', None))
        assert source is not None, "input does not contain 'file' or 'source', needed"
        table = self.input.get("table", "data")
        where = self.input.get("where", [])
        # TODO: here could just read in load_by columns (plus columns needed in col_funcs)
        #  - use load(), providing the needed columns, then drop duplicates of load_by

        # TODO: check source is valid h5
        # TODO: allow for dataframes to be provide directly (skip the df_iter go right to df[load_by].drop_duplicates)
        store = pd.HDFStore(source, mode="r")

        # get an iterator - more memory efficient
        cprint(f"creating iterator to read data in chunks, to determine unique load_by values ({load_by})", c='OKBLUE')
        df_iter = DataLoader.data_select(obj=store,
                                         table=table,
                                         where=where,
                                         # columns=get_cols,
                                         iterator=True,
                                         chunksize=chunksize)

        # df_iter = store.select(table, where=where, columns=load_by, iterator=True, chunksize=5000000)

        # get the unique values to bin by
        # TODO: here determine ahead of time the number of rows, and then number of iterations to go through
        # TODO: make this into a function, add timer
        t0 = time.time()
        unique_load_bys = []

        for idx, df in enumerate(df_iter):
            # TODO: could just get everything, apply column funcs and take what is needed
            #  - shouldn't be a memory issue if chuncksize isn't too big
            # TODO: should only apply relevant column funcs
            # NOTE: might be applying too many col funcs
            DataLoader.add_cols(df, col_func_dict=self.input.get("col_funcs", None))

            # add the unique load_bys found
            unique_load_bys.append(df[load_by].drop_duplicates())

        store.close()

        # combine and drop duplicates
        unique_load_bys = pd.concat(unique_load_bys)
        unique_load_bys.drop_duplicates(inplace=True)
        t1 = time.time()

        cprint(f"time to get unique load_by cols ({load_by}):\n{t1-t0:.2f} seconds", c="OKGREEN")

        unique_load_bys.sort_values(load_by, inplace=True)

        # read the data in chunks
        # TODO: allow for
        df_bin_all = []
        stats_all = []
        idx_count = 0
        for idx, row in unique_load_bys.iterrows():

            cprint("-"*10,c="OKBLUE")
            cprint("loading by:", c="OKBLUE")
            print(row)
            idx_count += 1
            cprint(f"{idx_count}/{len(unique_load_bys)}", c="OKGREEN")
            # select data - from store, include a where for current load_by values
            # NOTE: 'date' only where selection can be very fast (?)
            row_where = [{"col": k, "comp": "==", "val": v} for k, v in row.to_dict().items()]

            # df = DataLoader.data_select(obj=store,
            #                             where=where + row_where,
            #                             table=table)
            #
            # # add / modify columns
            # DataLoader.add_cols(df, col_func_dict=col_funcs)

            df = DataLoader.load(source=source,
                                 where=where + row_where,
                                 table=table,
                                 col_funcs=self.input.get("col_funcs", None),
                                 row_select=self.input.get("row_select", None),
                                 col_select=self.input.get("col_select", None)
                                 )

            if len(df) == 0:
                print("NO DATA FOUND, SKIPPING")
                continue

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

            print("---")
            print("head of data to be binned")
            print(df.head(2))
            print("---")

            cprint(f"binning by columns: {self.bin_config['by_cols']}", c="HEADER")

            ds_bin, stats_df = self.bin_wrapper(df, col_funcs=None, print_stats=False, **self.bin_config)

            # convert dataset to DataFrame
            df_bin = ds_bin.to_dataframe().dropna().reset_index()

            # merge on the bin_by info to the stats
            stats_df = stats_df.T
            row_df = row.to_frame().T
            stats_df.set_index(row_df.index, inplace=True)
            stats_all.append(pd.concat([row_df, stats_df], axis=1))

            # TODO: allow for merging on more columns to output
            DataLoader.add_cols(df_bin,
                                col_func_dict=self.add_output_cols)

            # TODO: after each batch write values to table/file - write which batches have complete as we
            #  - if restarting skip those batches already completed

            df_bin_all.append(df_bin)

        # store.close()
        out = pd.concat(df_bin_all)

        stats_all = pd.concat(stats_all)

        return out, stats_all

    def bin_data(self, chunksize=5000000):

        batch = self.input.get("batch", False)
        if batch:
            cprint("will bin data by batch", c="HEADER")
            df_bin, stats = self.bin_data_by_batch(chunksize=chunksize)
        else:
            cprint("will bin data all at once", c="HEADER")
            df_bin, stats = self.bin_data_all_at_once()

        return df_bin, stats

    def write_dataframe_to_table(self, df_bin):


        cprint(f"writing results to hdf5 file:\n{self.output_file}", c="OKGREEN")
        with pd.HDFStore(self.output_file, mode="w") as store_out:

            out_table = self.get("table", self.bin_config['val_col'])
            print(f"writing to table: {out_table}")
            store_out.put(key=out_table,
                          value=df_bin,
                          append=True,
                          format='table',
                          data_columns=True)

            store_attrs = store_out.get_storer(out_table).attrs

            # include configs
            store_attrs['raw_data_config'] = self.raw_data_config
            store_attrs['bin_config'] = self.config
            store_attrs['run_info'] = self.run_info


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
        print(f"there were too many points {len(plt_df)}\n"
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


def get_bin_data_config():

    # read json file provided as first argument
    config = get_config_from_sysargv()

    # if not json file provide, config will None, get defaults
    if config is None:
        config_file = get_parent_path("configs", "example_bin_raw_data.json")
        warnings.warn(f"\nconfig is empty / not provided, will just use an example config:\n{config_file}")
        with open(config_file, "r") as f:
            config = nested_dict_literal_eval(json.load(f))

        # override the defaults
        config['input']['file'] = get_parent_path("data", "example", "ABC.h5")
        config['output']['file'] = get_parent_path("data", "example", "ABC_binned.h5")

        assert os.path.exists(config['input']['file']), \
            f"config['input']['file']:\n{config['input']['file']}\ndoes not exists. " \
            f"to create run: python -m PyOptimalInterpolation.read_and_store"

    return config


if __name__ == "__main__":

    # TODO: extend comment in default / example config
    # TODO: move/merge class into DataPrep(?)
    # TODO: review / refactor how the output statistics are printed to the screen - particularly for batch

    # ---
    # Config / Parameters
    # ---

    # read in config
    config = get_bin_data_config()

    # ---
    # initialise
    # ---

    bd = BinData(**config)

    # ---
    # bin data
    # ---

    bin_df, stats = bd.bin_data()

    # ---
    # write to file
    # ---

    bd.write_dataframe_to_table(bin_df)
