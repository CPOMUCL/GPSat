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
import pyarrow.parquet as pq
import time
from dataclasses import dataclass
from typing import List, Dict, Tuple, Union, Type

import matplotlib.pyplot as plt
import cartopy.crs as ccrs

from IPython.display import display
from GPSat import get_data_path
from GPSat.dataprepper import DataPrep
from GPSat.dataloader import DataLoader
from GPSat.utils import stats_on_vals, cprint, json_serializable, nested_dict_literal_eval, \
    _method_inputs_to_config
from GPSat.plot_utils import plot_pcolormesh, plot_hist
from GPSat.decorators import timer

from GPSat import get_parent_path
import re

from GPSat.utils import get_config_from_sysargv

pd.set_option('display.max_columns', 200)


class BinData:

    def __init__(self,
                 # input: Union[dict, None] = None,
                 # file=None,
                 # source=None,
                 # table=None,
                 # bin_config: Union[dict, None] = None,
                 # output: Union[dict, None] = None,
                 # comment: Union[str, None] = None,
                 # add_output_cols: Union[dict, None] = None
                 ):
        """
        Class for binning data and storing the results.

        all input parameters are stored as attributes with the same name

        input, bin_config and output must be supplied

        Parameters
        ----------
        input : dict or None, optional
            The input dictionary containing necessary configurations.
        bin_config : dict or None, optional
            The binning configuration dictionary.
        output : dict or None, optional
            The output dictionary containing necessary configurations.
        comment : str or None, optional
            Optional comment to be printed.
        add_output_cols : dict or None, optional
            Dictionary with additional columns to be added in the output.
        """
        # attributes to store raw_data_config, bin config and run_info
        # - these will be written to the output file when write_dataframe_to_table() is called
        self.raw_data_config = None
        self.config = None
        self.run_info = None


    def _get_source(self, file=None, source=None):

        # source = source if source is not None else self.source
        if source is None:
            source = file
        assert source is not None, "'source' (and 'file') are both None, please provide a valid 'source' value "
        return source

    @staticmethod
    def bin_wrapper(df, col_funcs=None, print_stats=True, **bin_config):
        """
        Perform binning on a DataFrame with optional statistics printing and column modifications.

        This function wraps the binning process, allowing for optional statistics on the data before
        binning, dynamic column additions or modifications, and the application of various binning configurations.

        Parameters
        ----------
        df : pandas.DataFrame
            The DataFrame to be binned.
        col_funcs : dict, optional
            A dictionary where keys are column names to add or modify, and values are functions that
            take a pandas Series and return a modified Series. This allows for the dynamic addition
            or modification of columns before binning. Defaults to None.
        print_stats : bool, optional
            If True, prints basic statistics of the DataFrame before binning. Useful for a preliminary
            examination of the data. Defaults to True.
        **bin_config : dict
            Arbitrary keyword arguments defining the binning configuration. These configurations
            dictate how binning is performed and include parameters such as bin sizes, binning method,
            criteria for binning, etc.

        Returns
        -------
        ds_bin : xarray.Dataset
            The binned data as an xarray Dataset. Contains the result of binning the input DataFrame
            according to the specified configurations.
        stats_df : pandas.DataFrame
            A DataFrame containing statistics of the input DataFrame after any column additions or
            modifications and before binning. Provides insights into the data distribution and can
            inform decisions on binning parameters or data preprocessing.

        Notes
        -----
        The actual structure and contents of the `ds_bin` xarray Dataset will depend on the binning
        configurations specified in `**bin_config`. Similarly, the `stats_df` DataFrame provides a
        summary of the data's distribution based on the column specified in the binning configuration
        and can vary widely in its specifics.

        The binning process may be adjusted significantly through the `**bin_config` parameters,
        allowing for a wide range of binning behaviors and outcomes. For detailed configuration
        options, refer to the documentation of the specific binning functions used within this
        wrapper.
        """

        if print_stats:
            print("head of data:")
            print(df.head(3))

        if bin_config.get("verbose", False) >1:
            cprint("bin config provided:", c="OKBLUE")
            cprint(json.dumps(bin_config, indent=4), c="OKGREEN")

        # --
        # (optionally) add columns

        if col_funcs is not None:
            if bin_config.get("verbose", False):
                cprint("adding / modifying columns with:", c="OKBLUE")
            cprint(json.dumps(col_funcs, indent=4), c="OKGREEN")
            DataLoader.add_cols(df, col_func_dict=col_funcs, verbose=4)

        # --
        # stats on values
        # --

        val_col = bin_config['val_col']
        vals = df[val_col].values

        if np.isnan(vals).all():
            cprint("NaN values found: returning None, None", c="FAIL")
            return None, None

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

    @timer
    def bin_data_all_at_once(self,
                             file=None,
                             source=None,
                             table=None,
                             where=None,
                             add_output_cols=None,
                             bin_config=None,
                             **data_load_kwargs):
        """
        Reads the entire dataset, applies binning, and returns binned data along with statistics.

        This method handles the entire binning process in a single pass, making it suitable for
        datasets that can fit into memory. It allows for preprocessing of data through column
        functions, selection of specific rows and columns, and the addition of output columns
        after binning based on provided configurations.

        Parameters
        ----------
        file : str, optional
            Path to the source file containing the dataset if `source` is not specified. Defaults to None.
        source : str, optional
            An alternative specification of the data source. This could be a path to a file or another
            identifier depending on the context. If both `file` and `source` are provided, `source` takes precedence.
            Defaults to None.
        table : str, optional
            The name of the table within the data source to apply binning. Defaults to None.
        where : list of dict, optional
            Conditions for filtering rows before binning, expressed as a list of dictionaries
            representing SQL-like where clauses. Defaults to None.
       add_output_cols : dict, optional
            Dictionary mapping new column names to functions that define their values, for adding
            columns to the output DataFrame after binning. Defaults to None.
        bin_config : dict
            Configuration for the binning process, including parameters such as bin sizes, binning
            method, and criteria for binning. This parameter is required.
        **data_load_kwargs : dict, optional
            Additional keyword arguments to be passed into DataLoader.load
            see :func:`load <GPSat.dataloader.DataLoader.load>`


        Returns
        -------
        df_bin : pandas.DataFrame
            A DataFrame containing the binned data.
        stats_df : pandas.DataFrame
            A DataFrame containing statistics of the binned data, useful for analyzing the distribution
            and quality of the binned data.

        Raises
        ------
        AssertionError
            If `bin_config` is not provided or is not a dictionary.

        Notes
        -----
        This method is designed to handle datasets that can be loaded entirely into memory.
        For very large datasets, consider using the `bin_data_by_batch` method to process
        the data in chunks and avoid memory issues.

        The `add_output_cols` parameter allows for the dynamic addition of columns to the
        binned dataset based on custom logic, which can be useful for enriching the dataset
        with additional metrics or categorizations derived from the binned data.
        """

        assert bin_config is not None, "bin_config must be supplied"
        assert isinstance(bin_config, dict), "bin_config must be a dict - see input parameters to DataPrep.bin_data_by"

        # get source
        source = self._get_source(file, source)

        # read in all the data
        # table = self.input_info.get("table", "data")
        # table = table if table is not None else self.table
        # assert table is not None, ""

        df = DataLoader.load(source=source,
                             table=table,
                             where=where,
                             **data_load_kwargs)

        # bin the data
        ds_bin, stats_df = self.bin_wrapper(df, col_funcs=None, print_stats=False, **bin_config)

        # convert to DataFrame, if not already
        if isinstance(ds_bin, pd.DataFrame):
            df_bin = ds_bin
        else:
            df_bin = ds_bin.to_dataframe()
            del ds_bin

        df_bin = df_bin.dropna(how="any").reset_index()

        # add any additional extra columns
        DataLoader.add_cols(df_bin,
                            col_func_dict=add_output_cols)

        return df_bin, stats_df

    @timer
    def bin_data_by_batch(self,
                          file=None,
                          source=None,
                          load_by=None,
                          table=None,
                          where=None,
                          add_output_cols=None,
                          chunksize=5000000,
                          bin_config=None,
                          **data_load_kwargs):
        """
        Bins the data in chunks based on unique values of specified columns and returns the aggregated binned data and statistics.

        This method is particularly useful for very large datasets that cannot fit into memory. It reads
        the data in batches, applies binning to each batch based on the unique values of the specified
        `load_by` columns, and aggregates the results. This approach helps manage memory usage while
        allowing for comprehensive data analysis and binning.

        Parameters
        ----------
        file : str, optional
            Path to the source file containing the dataset if `source` is not specified. Defaults to None.
        source : str, optional
            An alternative specification of the data source. This could be a path to a file or another
            identifier depending on the context. If both `file` and `source` are provided, `source` takes precedence.
            Defaults to None.
        load_by : list of str
            List of column names based on which data will be loaded and binned in batches. Each unique
            combination of values in these columns defines a batch.
        table : str, optional
            The name of the table within the data source from which to load the data. Defaults to None.
        where : list of dict, optional
            Conditions for filtering rows from the source, expressed as a list of dictionaries
            representing SQL-like where clauses. Defaults to None.
        add_output_cols : dict, optional
            Dictionary mapping new column names to functions that define their values, for adding
            columns to the output DataFrame after binning. Defaults to None.
        chunksize : int, optional
            The number of rows to read into memory and process at a time. Defaults to 5,000,000.
        bin_config : dict
            Configuration for the binning process, including parameters such as bin sizes, binning
            method, and criteria for binning. This parameter is required.
        **data_load_kwargs : dict, optional
            Additional keyword arguments to be passed into DataLoader.load
            see :func:`load <GPSat.dataloader.DataLoader.load>`

        Returns
        -------
        df_bin : pandas.DataFrame
            A DataFrame containing the aggregated binned data from all batches.
        stats_all : pandas.DataFrame
            A DataFrame containing aggregated statistics of the binned data from all batches,
            useful for analyzing the distribution and quality of the binned data.

        Raises
        ------
        AssertionError
            If `bin_config` is not provided or is not a dictionary.

        Notes
        -----
        The `bin_data_by_batch` method is designed to handle large datasets by processing them in
        manageable chunks. It requires specifying `load_by` columns to define how the dataset is
        divided into batches for individual binning operations. This method ensures efficient memory
        usage while allowing for complex data binning and analysis tasks on large datasets.

        The `add_output_cols` parameter enables the dynamic addition of columns to the output dataset
        based on custom logic applied after binning, which can be used to enrich the dataset with
        additional insights or metrics derived from the binned data.
        """

        assert bin_config is not None, "bin_config must be supplied"
        assert isinstance(bin_config, dict), "bin_config must be a dict - see input parameters to DataPrep.bin_data_by"


        cprint("reading data in by batches", c="OKCYAN")

        # load_by and bin by can difference
        # e.g. could load by date and bin by track
        load_by = load_by if load_by is not None else bin_config['by_cols']
        # load_by = self.input.get("load_by", self.bin_config['by_cols'])
        if isinstance(load_by, str):
            load_by = [load_by]

        # require the load_by values are all in bin by_cols
        for lb in load_by:
            assert lb in bin_config['by_cols'], \
                f"load_by value: {lb} is not in bin by_cols: {bin_config['by_cols']}"

        print(f"load_by: {load_by}")
        # storer = store.get_storer(table)

        if where is None:
            where = []


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

        # get source
        source = self._get_source(file, source)

        # source = self.input.get('file', self.input.get('source', None))
        assert source is not None, "input does not contain 'file' or 'source', needed"

        # TODO: here could just read in load_by columns (plus columns needed in col_funcs)
        #  - use load(), providing the needed columns, then drop duplicates of load_by
        # TODO: check source is valid h5
        # TODO: allow for dataframes to be provide directly (skip the df_iter go right to df[load_by].drop_duplicates)

        # -------HACK-----------
        # - an attempt to read by batch use parquet using the same method as hdf5
        # - remove this, instead read large chunks at a time using parquet + filters

        # try:

        store = pd.HDFStore(source, mode="r")


        # if isinstance(source, str):
        # store = DataLoader._get_source_from_str(source=source)

        # get an iterator - more memory efficient
        cprint(f"creating iterator to read data in chunks, to determine unique load_by values:1 ({load_by})", c='OKBLUE')
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

        cprint(f"iterating over all of the data to then determine how to block it up - this could be slow", "BOLD")
        for idx, df in enumerate(df_iter):
            # TODO: should only apply relevant column funcs
            # NOTE: might be applying too many (some not needed) col funcs
            DataLoader.add_cols(df, col_func_dict=data_load_kwargs.get('col_funcs'))

            # add the unique load_bys found
            unique_load_bys.append(df[load_by].drop_duplicates())

        store.close()

        # except Exception as e:
        #
        #     parquet_file = pq.ParquetFile(source)
        #
        #     # for batch in parquet_file.iter_batches():
        #     #     print("RecordBatch")
        #     #     batch_df = batch.to_pandas()
        #     #     print("batch_df:", batch_df)
        #
        #     t0 = time.time()
        #     unique_load_bys = []
        #
        #     cprint(f"iterating over all of the data to then determine how to block it up - this could be slow", "BOLD")
        #     # TODO: load_by may not be present
        #
        #     try:
        #         for batch in parquet_file.iter_batches(batch_size=chunksize, columns=load_by):
        #
        #             df = batch.to_pandas()
        #
        #             # TODO: should only apply relevant column funcs
        #             # NOTE: might be applying too many (some not needed) col funcs
        #             # DataLoader.add_cols(df, col_func_dict=data_load_kwargs.get('col_funcs'))
        #
        #             # add the unique load_bys found
        #             unique_load_bys.append(df[load_by].drop_duplicates())
        #
        #     except Exception as e:
        #
        #         for batch in parquet_file.iter_batches(batch_size=chunksize):
        #             df = batch.to_pandas()
        #
        #             # TODO: should only apply relevant column funcs
        #             # NOTE: might be applying too many (some not needed) col funcs
        #             DataLoader.add_cols(df, col_func_dict=data_load_kwargs.get('col_funcs'))
        #
        #             # add the unique load_bys found
        #             unique_load_bys.append(df[load_by].drop_duplicates())

        # ----------------------

        # combine and drop duplicates
        unique_load_bys = pd.concat(unique_load_bys)
        unique_load_bys.drop_duplicates(inplace=True)
        t1 = time.time()

        cprint(f"time to get unique load_by cols ({load_by}):\n{t1 - t0:.2f} seconds (using these to batch)", c="OKGREEN")

        unique_load_bys.sort_values(load_by, inplace=True)

        # read the data in chunks
        # TODO: allow for
        df_bin_all = []
        stats_all = []
        idx_count = 0
        for idx, row in unique_load_bys.iterrows():

            cprint("-" * 10, c="OKBLUE")
            cprint(f"{idx_count}/{len(unique_load_bys)}", c="OKGREEN")
            cprint("loading data by:", c="OKBLUE")
            print(row)
            idx_count += 1

            # select data - from store, include a where for current load_by values
            # NOTE: 'date' only where selection can be very fast (?)
            row_where = [{"col": k, "comp": "==", "val": v} for k, v in row.to_dict().items()]

            # load data, add additional where conditions (filter before reading into memory)
            df = DataLoader.load(source=source,
                                 where=where + row_where,
                                 table=table,
                                 # source_kwargs={"engine": "pyarrow"},
                                 **data_load_kwargs)

            if len(df) == 0:
                print("NO DATA FOUND, SKIPPING")
                continue

            if bin_config.get('verbose', False):
                print("---")
                cprint("head of data to be binned:", c="BOLD")
                print(df.head(2))
                print("---")

            cprint(f"binning by columns: {bin_config['by_cols']}", c="HEADER")

            ds_bin, stats_df = self.bin_wrapper(df,
                                                col_funcs=None,
                                                print_stats=False,
                                                **bin_config)

            if ds_bin is None:
                print("DATA WAS ALL NAN, SKIPPING")
                continue

            # convert to DataFrame, if not already
            if isinstance(ds_bin, pd.DataFrame):
                df_bin = ds_bin
            else:
                df_bin = ds_bin.to_dataframe()
                del ds_bin

            df_bin = df_bin.dropna(how="any").reset_index()

            # merge on the bin_by info to the stats
            stats_df = stats_df.T
            row_df = row.to_frame().T
            stats_df.set_index(row_df.index, inplace=True)
            stats_all.append(pd.concat([row_df, stats_df], axis=1))

            # TODO: allow for merging on more columns to output
            DataLoader.add_cols(df_bin,
                                col_func_dict=add_output_cols)

            # TODO: after each batch write values to table/file - write which batches have complete as we
            #  - if restarting skip those batches already completed

            df_bin_all.append(df_bin)

        # store.close()
        out = pd.concat(df_bin_all)

        stats_all = pd.concat(stats_all)

        return out, stats_all


    def bin_data(self,
                 file=None,
                 source=None,
                 load_by=None,
                 table=None,
                 where=None,
                 batch=False,
                 add_output_cols=None,
                 bin_config=None,
                 chunksize=5000000,
                 **data_load_kwargs):
        """
        Bins the dataset, either in a single pass or in batches, based on the provided configuration.

        This method decides between processing the entire dataset at once or in chunks based on the `batch`
        parameter. It applies binning according to the specified `bin_config`, along with any preprocessing
        defined by `col_funcs`, `col_select`, and `row_select`. Additional columns can be added to the
        output dataset using `add_output_cols`. The method is capable of handling both small and very
        large datasets efficiently.

        Parameters
        ----------
        file : str, optional
            Path to the source file containing the dataset if `source` is not specified. Defaults to None.
        source : str, optional
            An alternative specification of the data source. This could be a path to a file or another
            identifier depending on the context. If both `file` and `source` are provided, `source` takes precedence.
            Defaults to None.
        load_by : list of str, optional
            List of column names based on which data will be loaded and binned in batches if `batch` is True.
            Each unique combination of values in these columns defines a batch. Defaults to None.
        table : str, optional
            The name of the table within the data source from which to load the data. Defaults to None.
        where : list of dict, optional
            Conditions for filtering rows from the source, expressed as a list of dictionaries
            representing SQL-like where clauses. Defaults to None.
        batch : bool, optional
            If True, the data is processed in chunks based on `load_by` columns. If False, the entire
            dataset is processed at once. Defaults to False.
        add_output_cols : dict, optional
            Dictionary mapping new column names to functions that define their values, for adding
            columns to the output DataFrame after binning. Defaults to None.
        bin_config : dict
            Configuration for the binning process, including parameters such as bin sizes, binning
            method, and criteria for binning. This parameter is required.
        chunksize : int, optional
            The number of rows to read into memory and process at a time, applicable when `batch` is True.
            Defaults to 5,000,000.
        **data_load_kwargs : dict, optional
            Additional keyword arguments to be passed into DataLoader.load
            see :func:`load <GPSat.dataloader.DataLoader.load>`


        Returns
        -------
        df_bin : pandas.DataFrame
            A DataFrame containing the binned data.
        stats : pandas.DataFrame
            A DataFrame containing statistics of the binned data, useful for analyzing the distribution
            and quality of the binned data.

        Raises
        ------
        AssertionError
            If `bin_config` is not provided or is not a dictionary.

        Notes
        -----
        The `bin_data` method offers flexibility in processing datasets of various sizes by allowing
        for both batch processing and single-pass processing. The choice between these modes is controlled
        by the `batch` parameter, making it suitable for scenarios ranging from small datasets that fit
        easily into memory to very large datasets requiring chunked processing to manage memory usage effectively.

        The additional parameters for row and column selection and the ability to add new columns after
        binning allow for significant customization of the binning process, enabling users to tailor the
        method to their specific data processing and analysis needs.
        """
        source = self._get_source(file, source)

        # --
        # store configs -
        # --

        # to have audit trail on how it was created
        # NOTE: timer decorator causes issue with extracting config like this
        self.config = _method_inputs_to_config(locs=locals(),
                                               code_obj=self.bin_data.__code__)

        try:
            run_info = DataLoader.get_run_info(script_path=__file__)
        except NameError as e:
            run_info = DataLoader.get_run_info()
        self.run_info = run_info

        try:
            # currently this works for hdf5 files
            # - can parquet store attributes?
            self.raw_data_config = DataLoader.get_attribute_from_table(source=source,
                                                                       table=table,
                                                                       attribute_name='config')
        except Exception as e: # tables.exceptions.HDF5ExtError:
            pass

        assert batch is not None
        assert isinstance(batch, bool), f"batch is expected to be a bool value, got: {batch}"

        if batch:
            cprint("will bin data by batch", c="HEADER")
            df_bin, stats = self.bin_data_by_batch(source=source,
                                                   load_by=load_by,
                                                   table=table,
                                                   where=where,
                                                   chunksize=chunksize,
                                                   bin_config=bin_config,
                                                   add_output_cols=add_output_cols,
                                                   **data_load_kwargs)
        else:
            cprint("will bin data all at once", c="HEADER")
            df_bin, stats = self.bin_data_all_at_once(source=source,
                                                      table=table,
                                                      where=where,
                                                      bin_config=bin_config,
                                                      add_output_cols=add_output_cols,
                                                      **data_load_kwargs)

        return df_bin, stats

    def write_dataframe_to_table(self,
                                 df_bin,
                                 file=None,
                                 table=None):
        """
        Writes the binned DataFrame to a specified table in an HDF5 file.

        This method saves the binned data, represented by a DataFrame, into a table within an HDF5 file.
        The method assumes that the HDF5 file is accessible and writable. It allows for the efficient
        storage of large datasets and facilitates easy retrieval for further analysis or processing.

        Parameters
        ----------
        df_bin : pandas.DataFrame
            The DataFrame containing the binned data to be written to the file. This DataFrame should
            already be processed and contain the final form of the data to be saved.
        file : str
            The path to the HDF5 file where the DataFrame will be written. If the file does not exist,
            it will be created. If the file exists, the method will write the DataFrame to the specified
            table within the file.
        table : str
            The name of the table within the HDF5 file where the DataFrame will be stored. If the table
            already exists, the new data will be appended to it.

        Raises
        ------
        AssertionError
            If either `file` or `table` is not specified.

        Notes
        -----
        The HDF5 file format is a versatile data storage format that can efficiently store large datasets.
        It is particularly useful in contexts where data needs to be retrieved for analysis, as it supports
        complex queries and data slicing. This method leverages the pandas HDFStore mechanism for storing
        DataFrames, which abstracts away many of the complexities of working directly with HDF5 files.

        This method also includes the `raw_data_config`, `config` (the binning configuration), and `run_info`
        as attributes of the stored table, providing a comprehensive audit trail of how the binned data was
        generated. This can be crucial for reproducibility and understanding the context of the stored data.
        """

        assert file is not None
        assert table is not None

        cprint("-"*20, c="OKGREEN")
        cprint(f"writing results to hdf5 file:\n{file}", c="OKGREEN")
        with pd.HDFStore(file, mode="w") as store_out:
            # out_table = output.get("table", self.bin_config['val_col'])
            cprint(f"writing to table: '{table}'", c="OKGREEN")
            store_out.put(key=table,
                          value=df_bin,
                          append=True,
                          format='table',
                          data_columns=True)

            store_attrs = store_out.get_storer(table).attrs

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
                f"min datetime {str(plt_df[date_col].min())}, " \
                f"max datetime: {str(plt_df[date_col].max())} \n" \
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
    # cprint('trying to read in configuration from argument', c="OKCYAN")
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
            f"to create run: python -m GPSat.read_and_store"

    return config


if __name__ == "__main__":
    # TODO: review docstrings
    # TODO: remove col_funcs, row_select, col_select - just provide as DataLoader.load kwargs
    # TODO: extend comment in default / example config
    # TODO: move/merge class into DataPrep(?)
    # TODO: review / refactor how the output statistics are printed to the screen - particularly for batch
    # TODO: all for binning of directly provide DataFrame

    cprint('-' * 60, c="BOLD")
    cprint('-' * 60, c="BOLD")
    cprint("running bin_data, expect configuration (JSON) file to be provide as argument", c="OKBLUE")

    # ---
    # Config / Parameters
    # ---

    # read in config
    config = get_bin_data_config()

    # ---
    # extract contents from config
    # ---

    cprint("-" * 30, c="BOLD")
    cprint("will attempt to bin data using the following config:", c="OKCYAN")
    cprint(json.dumps(json_serializable(config), indent=4), c="HEADER")

    input = config.get("input", {})
    bin_config = config.get("bin_config", {})
    output = config.get("output", {})

    # TODO: refactor this to be taken from the output config
    add_output_cols = config.get("add_output_cols", {})

    # ---
    # initialise
    # ---

    bd = BinData()

    # ---
    # bin data
    # ---

    bin_df, stats = bd.bin_data(source=input.get("source"),
                                file=input.get("file"),
                                load_by=input.get("load_by"),
                                table=input.get("table"),
                                where=input.get("where"),
                                col_funcs=input.get("col_funcs"),
                                col_select=input.get("col_select"),
                                row_select=input.get("row_select"),
                                add_output_cols=add_output_cols,
                                batch=input.get("batch", True),
                                bin_config=bin_config)

    # ---
    # write to file
    # ---

    bd.write_dataframe_to_table(bin_df,
                                file=output['file'],
                                table=output['table'])
