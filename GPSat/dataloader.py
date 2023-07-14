# DataLoader class is defined below
# TODO: remove unused methods
import datetime
import os
import re
import sys
import warnings
import pickle
import types

import pandas as pd
import numpy as np
import xarray as xr
import scipy.stats as scst
from scipy.spatial import KDTree

from functools import reduce
from GPSat.utils import config_func, get_git_information, sparse_true_array, pandas_to_dict
from GPSat.decorators import timer


class DataLoader:


    file_suffix_engine_map = {
        "csv": "read_csv",
        "tsv": "read_csv",
        "h5": "HDFStore",
        "zarr": "zarr",
        "nc": "netcdf4"
    }

    # TODO: add docstring for class and methods
    # TODO: need to make row select options consistent
    #  - those that use config_func and those used in _bool_xarray_from_where
    #  - could just check keys of provided dict
    def __init__(self, hdf_store=None, dataset=None):

        self.connect_to_hdf_store(hdf_store)

    @staticmethod
    def add_cols(df, col_func_dict=None, filename=None, verbose=False):
        """
        Adds new columns to a given DataFrame based on the provided dictionary of column-function pairs.

        This function allows the user to add new columns to a DataFrame using a dictionary
        that maps new column names to functions that compute the column values. The functions
        can be provided as values in the dictionary, and the new columns can be added to the
        DataFrame in a single call to this function.

        If a tuple is provided as a key in the dictionary, it is assumed that the corresponding
        function will return multiple columns. The length of the returned columns should match
        the length of the tuple.

        Parameters
        ----------
        df : pandas.DataFrame
            The input DataFrame to which new columns will be added.
        col_func_dict : dict, optional
            A dictionary that maps new column names (keys) to functions (values) that compute
            the column values. If a tuple is provided as a key, it is assumed that the corresponding
            function will return multiple columns. The length of the returned columns should match
            the length of the tuple. If None, an empty dictionary will be used. Default is None.
        filename : str, optional
            The name of the file from which the DataFrame was read. This parameter will be passed
            to the functions provided in the col_func_dict. Default is None.
        verbose : int or bool, optional
            Determines the level of verbosity of the function. If verbose is 3 or higher, the function
            will print messages about the columns being added. Default is False.

        Returns
        -------
        None

        Notes
        -----
        DataFrame is manipulated inplace.
        If a single value is returned by the function, it will be assigned to a column with the name specified in the key.
        See help(utils.config_func) for more details.

        Raises
        ------
        AssertionError
            If the length of the new columns returned by the function does not match the length of
            the tuple key in the col_func_dict.

        Examples
        --------
        >>> import pandas as pd
        >>> from GPSat.dataloader import DataLoader
        >>> def add_one(df, filename=None):
        ...     return df['A'] + 1
        >>> df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
        >>> DataLoader.add_cols(df, col_func_dict={'C': {'func': add_one, "args": df}})
        >>> print(df)
           A  B  C
        0  1  4  2
        1  2  5  3
        2  3  6  4

        """

        # TODO: replace filename with **kwargs
        if col_func_dict is None:
            col_func_dict = {}

        for new_col, col_fun in col_func_dict.items():

            # add new column
            if verbose >= 3:
                print(f"adding new_col: {new_col}")

            # allow for multiple columns to be assigned at once
            # dict can have tuple as keys - need to be converted to list
            if isinstance(new_col, tuple):
                new_col = list(new_col)
                _ = config_func(df=df,
                                filename=filename,
                                **col_fun)
                assert len(_) == len(new_col), f"columns: {new_col} have length: {len(new_col)}" \
                                               f"but function returned from kwargs {col_fun}\n" \
                                               f"only returned: {len(_)} values"
                for i, vals in enumerate(_):
                    df[new_col[i]] = vals
            # otherwise just assign a single value
            else:
                df[new_col] = config_func(df=df,
                                          filename=filename,
                                              **col_fun)

    @classmethod
    def row_select_bool(cls, df, row_select=None, verbose=False, **kwargs):

        if row_select is None:
            row_select = [{}]
        elif isinstance(row_select, dict):
            row_select = [row_select]

        assert isinstance(row_select, list), f"expect row_select to be a list (of dict), is type: {type(row_select)}"
        for i, rs in enumerate(row_select):
            assert isinstance(rs, dict), f"index element: {i} of row_select was type: {type(rs)}, rather than dict"

        select = np.ones(len(df), dtype=bool)

        for sl in row_select:
            # print(sl)
            # skip empty list
            if len(sl) == 0:
                continue
            if verbose >= 3:
                print("selecting rows")
            # select &= config_func(df=df, **{**kwargs, **sl})
            select &= cls._bool_numpy_from_where(df, {**kwargs, **sl.copy()})

        return select


    @classmethod
    @timer
    def read_from_multiple_files(cls,
                                 file_dirs, file_regex,
                                 read_engine="csv",
                                 sub_dirs=None,
                                 col_funcs=None,
                                 row_select=None,
                                 col_select=None,
                                 new_column_names=None,
                                 strict=True,
                                 read_kwargs=None,
                                 read_csv_kwargs=None,
                                 verbose=False):

        # --
        # check inputs
        # --

        valid_read_engine = ['csv', 'nc', 'netcdf', 'xarray']
        assert read_engine in valid_read_engine, f"read_engine: {read_engine} is not valid, " \
                                                 f"must be one of: {valid_read_engine}"
        if verbose:
            print(f"using read_engine: {read_engine}")

        if read_csv_kwargs is not None:
            warnings.warn("read_csv_kwargs was provided, will set read_kwargs to these values. "
                          "In future provide as read_kwargs instead", DeprecationWarning)

        # additional kwargs for pd.read_csv, or xr.open_dataset
        if read_kwargs is None:
            read_kwargs = {}
        assert isinstance(read_kwargs,
                          dict), f"expect read_csv_kwargs to be a dict, is type: {type(read_kwargs)}"

        # functions used to generate column values
        if col_funcs is None:
            col_funcs = {}
        assert isinstance(col_funcs, dict), f"expect col_funcs to be a dict, is type: {type(col_funcs)}"

        if col_select is None:
            if verbose:
                print("col_select is None, will take all")
            col_select = slice(None)

        if isinstance(file_dirs, str):
            file_dirs = [file_dirs]

        if sub_dirs is None:
            sub_dirs = [""]
        elif isinstance(sub_dirs, str):
            sub_dirs = [sub_dirs]

        # if sub directories provided, join paths with each file_dir and use those
        if len(sub_dirs):
            sub_file_dirs = []
            for fd in file_dirs:
                for sd in sub_dirs:
                    sub_file_dirs += [os.path.join(fd, sd)]
            file_dirs = sub_file_dirs

        # check file_dirs exists
        for file_dir in file_dirs:
            if strict:
                assert os.path.exists(file_dir), f"file_dir:\n{file_dir}\ndoes not exist"
            elif not os.path.exists(file_dir):
                warnings.warn(f"file_dir:\n{file_dir}\nwas provide but does not exist")
        file_dirs = [f for f in file_dirs if os.path.exists(f)]

        # for each file, read in
        # - store results in a list
        res = []
        for file_dir in file_dirs:
            print("-" * 100)
            print(f"reading files from:\n{file_dir}\nthat match regular expression: {file_regex}")
            # get all files in file_dir matching expression
            files = [os.path.join(file_dir, _)
                     for _ in os.listdir(file_dir)
                     if re.search(file_regex, _)]

            # ---
            # read in files
            # ---

            # NOTE: multiple netcdf files can be read at once - would it be faster to do that?

            # increment over each file
            for f_count, f in enumerate(files):

                if verbose >= 2:
                    print(f"reading file: {f_count + 1}/{len(files)}")

                # TODO: replace the following with load()

                # read_csv
                if read_engine == "csv":
                    df = pd.read_csv(f, **read_csv_kwargs)
                # read from netcdf
                elif read_engine in ['nc', 'netcdf', 'xarray']:
                    ds = xr.open_dataset(f, **read_kwargs)
                    # NOTE: would be more memory efficient if only got the required rows and columns
                    # TODO: determine if it would be faster to read in multiple files at once with open_mfdataset
                    df = ds.to_dataframe()

                if verbose >= 3:
                    print(f"read in: {f}\nhead of dataframe:\n{df.head(3)}")

                # ---
                # apply column functions - used to add new columns

                cls.add_cols(df,
                             col_func_dict=col_funcs,
                             verbose=verbose,
                             filename=f)

                # ----
                # select rows

                select = cls.row_select_bool(df,
                                             row_select=row_select,
                                             verbose=verbose,
                                             filename=f)

                # select subset of data
                if verbose >= 3:
                    print(f"selecting {select.sum()}/{len(select)} rows")
                df = df.loc[select, :]

                # ----
                # select columns

                # TODO: add more checks around this
                df = df.loc[:, col_select]

                if verbose >= 2:
                    print(f"adding data with shape: {df.shape}")

                # change column names
                if new_column_names is not None:
                    assert len(new_column_names) == df.shape[1], "new_col_names were provided " \
                                                                 f"but have length: {len(new_column_names)}, " \
                                                                 f"which does not match df.shape[1]: {df.shape[1]}"
                    df.columns = new_column_names

                # -----
                # store results
                res += [df]

        # ----
        # concat all
        out = pd.concat(res)

        return out


    @classmethod
    def read_flat_files(cls, file_dirs, file_regex,
                        sub_dirs=None,
                        read_csv_kwargs=None,
                        col_funcs=None,
                        row_select=None,
                        col_select=None,
                        new_column_names=None,
                        strict=True,
                        verbose=False):
        """
        read flat files (csv, tsv, etc) from file system
        return dataframe

        Parameters
        ----------
        file_dirs
        file_regex
        sub_dirs
        read_csv_kwargs
        col_funcs
        row_select
        col_select
        new_column_names
        strict
        verbose

        Returns
        -------
        pd.DataFrame
        """

        # TODO: review the verbose levels
        # TODO: finish doc string
        # TODO: provide option to use os.walk?
        # TODO: check contents of row_select, col_funcs - make sure only valid keys are provided

        out = cls.read_from_multiple_files(
            file_dirs=file_dirs,
            file_regex=file_regex,
            read_engine="csv",
            sub_dirs=sub_dirs,
            col_funcs=col_funcs,
            row_select=row_select,
            col_select=col_select,
            new_column_names=new_column_names,
            strict=strict,
            read_kwargs=read_csv_kwargs,
            verbose=verbose
        )

        return out

    @staticmethod
    def read_hdf(table, store=None, path=None, close=True, **select_kwargs):
        # read (table) data from hdf5 (e.g. .h5) file, possibly selecting only a subset of data
        # return DataFrame
        assert not ((store is None) & (path is None)), f"store and file are None, provide either"

        if store is not None:
            # print("store is provide, using it")
            assert isinstance(store, pd.io.pytables.HDFStore), f"store provide is wrong type, got: {type(store)}"
        elif path is not None:
            # print("using provided path")
            store = pd.HDFStore(path=path, mode="r")
            close = True

        # select data
        df = store.select(key=table, auto_close=close, **select_kwargs)

        return df

    @staticmethod
    def write_to_hdf(df, store,
                     table=None,
                     append=False,
                     config=None,
                     run_info=None):

        # write table
        store.put(key=table,
                  value=df,
                  append=append,
                  format='table',
                  data_columns=True)
        # ---
        # add meta-data / attributes
        # ---

        # configuration - used to generate df (if applicable)
        if config is None:
            print("config is None, will not assign as attr")
        else:
            # store config
            store_attrs = store.get_storer(table).attrs

            # store config and run information
            # if already has 'config' as attribute, then extend (in list)
            # TODO: when storing config do keys get sorted?
            #  - that could affect regression (re-running) if order matters - i.e. with col_funcs
            if hasattr(store_attrs, 'config') & append:
                prev_config = store_attrs.config
                prev_config = prev_config if isinstance(prev_config, list) else [prev_config]
                store_attrs.config = prev_config + [config]
            # store config
            else:
                store_attrs.config = config

        # run information - information data was generated
        if run_info is None:
            print("run_info is None, will not assign as attr")
        else:
            if hasattr(store_attrs, 'run_info') & append:
                prev_run_info = store_attrs.run_info
                prev_run_info = prev_run_info if isinstance(prev_run_info, list) else [prev_run_info]
                store_attrs.run_info = prev_run_info + [run_info]
            else:
                store_attrs.run_info = run_info

    def connect_to_hdf_store(self, store, table=None, mode='r'):
        # connect to hdf file via pd.HDFStore, return store object
        if store is None:
            self.hdf_store = None
        elif isinstance(store, str):
            # if str then expected to be a file path, check if exist
            self.hdf_store = None
        else:
            self.hdf_store = None

    @staticmethod
    def read_netcdf(ds=None, path=None, **kwargs):
        # TODO: remove this this method
        # read data from netcdf (.nc) using xarray either by connecting to a .nc file
        # or by using an open dataset connection
        # use where conditions to select subset of data
        # return either a DataFrame or a xarray object (DataArray or Dataset?)

        # assert not ((ds is None) & (path is None))
        #
        # if path is not None:
        #     ds = xr.open_dataset(filename_or_obj=path, **kwargs)
        pass

    @staticmethod
    def write_to_netcdf(ds, path, mode="w", **to_netcdf_kwargs):
        # given a xr.Dataset object, write to file
        # - simple rapper ds.to_netcdf
        assert isinstance(ds, xr.core.dataset.Dataset), f'ds must be Dataset object, got: {type(ds)}'
        ds.to_netcdf(path=path, mode=mode, **to_netcdf_kwargs)

    @staticmethod
    def read_from_pkl_dict(pkl_files,
                           pkl_dir=None,
                           default_name="obs",
                           strict=True,
                           dim_names=None):

        # TODO: test if pkl_files as str, list of str will work

        # HARDCODED: function to convert keys to dates
        key_to_date = lambda x: np.datetime64(f"{x[0:4]}-{x[4:6]}-{x[6:8]}")

        if isinstance(pkl_files, str):
            pkl_files = {default_name: [pkl_files]}
        elif isinstance(pkl_files, list):
            pkl_files = {default_name: pkl_files}

        assert isinstance(pkl_files, dict), f"pkl_files expected to be dict"

        if pkl_dir is None:
            pkl_dir = ""

        # store data in dict
        # data_vars = {}
        # if just getting dataframes directly store in list
        df_data_vars = []
        # increment over files
        for name, files in pkl_files.items():
            print("*" * 10)
            print(name)

            if isinstance(files, str):
                files = [files]

            # xa_list = []
            df_list = []
            for f in files:
                path = os.path.join(pkl_dir, f)

                if strict:
                    assert os.path.exists(path), f"path: {path}\ndoes not exist"
                else:
                    if not os.path.exists(path):
                        print(f"path: {path}\ndoes not exist, skipping")
                        continue
                # read in data
                with open(path, "rb") as f:
                    _ = pickle.load(f)

                # convert to DataArray

                # - the following is specific for legacy binned obs
                obs = np.concatenate([v[..., None] for k, v in _.items()], axis=-1)
                dates = [key_to_date(k) for k, v in _.items()]

                # store in DataArray
                # xa = xr.DataArray(obs, coords={"date": dates}, name=name, dims=dim_names)
                # #  xadf = xa.to_dataframe().dropna()

                # store in DataFrame
                midx = pd.MultiIndex.from_product([
                    np.arange(obs.shape[0]),
                    np.arange(obs.shape[1]),
                    dates],
                names=['idx0', 'idx1', 'date'])
                df = pd.DataFrame(obs.flat, index=midx, columns=[default_name]).dropna()
                df['source'] = name

                # very slow
                # NOTE: for asserting frames are equal use names=['y', 'x', 'date'] in above midx
                # pd.testing.assert_frame_equal(df, xadf)

                # xa_list += [xa]
                df_list += [df]

            # combine the obs
            # - test for this doing the correct thing - see commented below
            # data_vars[name] = xr.merge(xa_list, compat="override")[name]

            # pandas
            df_data_vars.append(pd.concat(df_list))

            # TODO: put this in a test?
            # # check merge data is as expected
            # date = "2018-12-01"
            # d0 = ds[name].sel(date=date).data
            #
            # data_list = [_.sel(date=date).data for _ in xa_list]
            #
            # from functools import reduce
            # def overwrite(a, b):
            #     a[~np.isnan(b)] = b[~np.isnan(b)]
            #     return a
            #
            # d1 = reduce(overwrite, data_list)
            #
            # dif = (d0 - d1)
            # # where dif is not nan, d0 or d1 are not nan
            # assert (~np.isnan(dif) == (~np.isnan(d0) | ~np.isnan(d1))).all()
            # # where not nan the difference is zero
            # assert (dif[~np.isnan(dif)] == 0).all()

        # out = xr.Dataset(data_vars)
        out = pd.concat(df_data_vars)

        return out

    @staticmethod
    def read_from_npy(npy_files, npy_dir, dims=None, flatten_xy=True, return_xarray=True):
        """
        Read NumPy array(s) from the specified .npy file(s) and return as xarray DataArray(s).

        This function reads one or more .npy files from the specified directory and returns them as xarray DataArray(s).
        The input can be a single file, a list of files, or a dictionary of files with the desired keys.
        The returned dictionary contains the xarray DataArray(s) with the corresponding keys.

        Parameters
        ----------
        npy_files : str, list, or dict
            The .npy file(s) to be read. It can be a single file (str), a list of files, or a dictionary of files.
        npy_dir : str
            The directory containing the .npy file(s).
        dims : list or tuple, optional
            The dimensions for the xarray DataArray(s), by default None.
        flatten_xy : bool, optional
            If True, flatten the x and y arrays by taking the first row and first column, respectively, by default True.
        return_xarray: bool, default True
            If True will convert numpy arrays to DataArray, otherwise will return dict of numpy arrays
        Returns
        -------
        dict
            A dictionary containing xarray DataArray(s) with keys corresponding to the input files.

        Examples
        --------
        >>> read_from_npy(npy_files="data.npy", npy_dir="./data")
        {'obs': <xarray.DataArray (shape)>

        >>> read_from_npy(npy_files=["data1.npy", "data2.npy"], npy_dir="./data")
        {'obs': [<xarray.DataArray (shape1)>, <xarray.DataArray (shape2)>]}

        >>> read_from_npy(npy_files={"x": "data_x.npy", "y": "data_y.npy"}, npy_dir="./data")
        {'x': <xarray.DataArray (shape_x)>, 'y': <xarray.DataArray (shape_y)>}

        """
        # TODO: docstring needs to be reviewed
        # TODO: review this function - was used for loading legacy data?

        if isinstance(npy_files, str):
            npy_files = {'obs': [npy_files]}
        elif isinstance(npy_files, list):
            npy_files = {'obs': npy_files}

        assert isinstance(npy_files, dict), f"npy_files expected to be dict"

        if npy_dir is None:
            npy_dir = ""

        coord_arrays = {}
        for name, f in npy_files.items():
            try:
                coord_arrays[name] = np.load(os.path.join(npy_dir, f))
            except Exception as e:
                print(f"issue reading aux data with prefix: {name}\nError: {e}")

            if return_xarray:
                coord_arrays[name] = xr.DataArray(coord_arrays[name],
                                                  dims=dims,
                                                  name=name)

        # NOTE: this isn't really flattening, it's just taking first row/column
        if ('x' in npy_files) & ('y' in npy_files) & flatten_xy:
            if return_xarray:
                coord_arrays['x'] = coord_arrays['x'].isel(y=0)
                coord_arrays['y'] = coord_arrays['y'].isel(x=0)
            else:
                coord_arrays['x'] = coord_arrays['x'][0, :]
                coord_arrays['y'] = coord_arrays['y'][:, 0]

        return coord_arrays

    @classmethod
    @timer
    def data_select(cls,
                    obj,
                    where=None,
                    table=None,
                    return_df=True,
                    reset_index=False,
                    drop=True,
                    copy=True,
                    columns=None,
                    close=False,
                    **kwargs):
        """
        Selects data from an input object (DataFrame, HDFStore, or xarray) based on filtering conditions.

        This function filters data from various types of input objects based on the provided conditions
        specified in the 'where' parameter. It also supports selecting specific columns, resetting the index,
        and returning the output as a DataFrame.

        Parameters
        ----------
        obj : pandas.DataFrame, pandas.Series, pandas.HDFStore, xarray.DataArray, or xarray.Dataset
            The input object from which data will be selected.
        where : dict, list of dict or None, default is None
            Filtering conditions to be applied to the input object. It can be a single dictionary or a list
            of dictionaries. Each dictionary should have keys: "col", "comp", "val",
            e.g. {"col": "t", "comp": "<=", "val": 4}. The "col" value specifies the column, "comp" specifies
            the comparison to be performed (>, >=, ==, !=, <=, <) and "val" is the value to be compared against.
            If None then selects all data. Specifying 'where' parameter can avoid reading all data in from
            filesystem when obj is pandas.HDFStore or xarray.Dataset
        table : str, default is None
            The table name to select from when using an HDFStore object.
            If obj is pandas.HDFStore then table must be supplied
        return_df : bool, default True
            If True, the output will be returned as a pandas.DataFrame
        reset_index : bool, default is False
            If True, the index of the output DataFrame will be reset.
        drop : bool, default True
            If True, the output will have the filtered-out values removed. Applicable only for xarray objects.
            Default is True.
        copy : bool, default True
            If True, the output will be a copy of the selected data. Applicable only for DataFrame objects.
        columns : list or None, default is None
            A list of column names to be selected from the input object. If None, which selects all columns.
        close : bool, default is False
            If True, and obj is pandas.HDFStore it will be closed after selecting data
        kwargs : any
            Additional keyword arguments to be passed to the obj.select method when using an HDFStore object.

        Returns
        -------
        out : pandas.DataFrame, pandas.Series, or xarray.DataArray
            The filtered data as a DataFrame, Series, or DataArray, based on the input object type and return_df parameter.

        Raises
        ------
        AssertionError
            If the table parameter is not provided when using an HDFStore object.
        AssertionError
            If the provided columns are not found in the input object when using a DataFrame object.

        Examples
        --------
        >>> import pandas as pd
        >>> import xarray as xr
        >>> from GPSat.dataloader import DataLoader
        >>> df = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})

        # Select data from a DataFrame with a filtering condition
        >>> selected_df = DataLoader.data_select(df, where={"col": "A", "comp": ">=", "val": 2})
        >>> print(selected_df)
           A  B
        1  2  5
        2  3  6

        """

        # TODO: provide doc string
        # TODO: specify how kwargs can work - depends on obj type
        # TODO: this method needs to be unit tested
        #  - check get similar type of results for different obj typs
        # select data from dataframe based off of where conditions
        # - extend to include local expert center, local expert inclusion (radius/select)

        if isinstance(where, dict):
            where = [where]

        # to handle empty list
        if isinstance(where, list):
            if len(where) == 0:
                where = None

        # is where a list of dicts?
        # - will require converting to a more specific where
        is_list_of_dict = cls.is_list_of_dict(where)

        # xr.DataArray, xr.DataSet
        if isinstance(obj, (xr.core.dataarray.DataArray, xr.core.dataarray.Dataset)):
            # TODO: for xarray allow for columns to be used
            if columns is not None:
                warnings.warn(f"columns were provided, but currently not implemented for obj type: {type(obj)}")

            # convert list of dict to bool DataArray
            if is_list_of_dict:
                tmp = [cls._bool_xarray_from_where(obj, wd) for wd in where]
                # combine (bool xarrays) using &
                where = reduce(lambda x, y: x & y, tmp)

            # TODO: should check where for type here - what is valid? DataArray, np.array?
            if where is None:
                out = obj
            else:
                out = obj.where(where, drop=drop)

            # return DataFrame ?
            if return_df:
                out = out.to_dataframe().dropna()
                # copy attributes - if possible
                try:
                    out.attrs = obj.attrs
                except AttributeError:
                    pass

            # TODO: should reset_index be default?
            if reset_index:
                out.reset_index(inplace=True)

        # pd.HDFStore
        elif isinstance(obj, pd.io.pytables.HDFStore):
            # TODO: determine if it is always the case
            assert table is not None, "\n\nobj is HDFStore, however table is None, needs to be provided\n\n"

            if is_list_of_dict:
                where = [cls._hdfstore_where_from_dict(wd) for wd in where]

            try:
                out = obj.select(key=table, where=where, columns=columns, **kwargs)
            except KeyError as e:
                print(f"exception occurred: {e}\nwill now close object")
                if close:
                    obj.close()
                raise KeyError

            if reset_index:
                out.reset_index(inplace=True)

            # close the HDFStore object?
            if close:
                print("closing")
                obj.close()

        # pd.DataFrame
        elif isinstance(obj, (pd.DataFrame, pd.Series)):
            # TODO: where selection should be able to select from multi index

            if is_list_of_dict:
                tmp = [cls._bool_numpy_from_where(obj, wd) for wd in where]
                where = reduce(lambda x, y: x & y, tmp)

            # if where is None - take all (using slice)
            if where is None:
                where = slice(where)
            assert isinstance(where, (np.ndarray, pd.Series, slice))

            if columns is None:
                columns = slice(None)
            else:
                missing_columns = []
                for c in columns:
                    if c not in obj:
                        missing_columns.append(c)
                assert len(missing_columns) == 0, f"columns were provide, but {missing_columns} are not in obj (dataframe)"
            out = obj.loc[where, columns]

            if copy:
                out = out.copy()

        else:
            warnings.warn(f"type(obj): {type(obj)} was not understood, returning None")
            out = None

        # TODO: allow for col_funcs to be applied?

        return out

    @classmethod
    def _get_source_from_str(cls, source, engine=None, verbose=False, **kwargs):

        # TODO: add doc string

        # do nothing if source is not str
        if not isinstance(source, str):
            print(f"source provided to _get_source_from_str(...) is not a str, got type: {type(source)}\n"
                  f"return source as is")

            return source

        # TODO: allow engine to not be case sensitive
        # TODO: allow for files to be handled by DataLoader.read_flat_files()
        #  - i.e. let file be a dict to be unpacked into read_flat_files, set engine = "read_flat_files"
        # TODO: add verbose statements

        # given a string get the corresponding data source
        # i.e. DataFrame, Dataset, HDFStore

        # if engine is None then infer from file name
        if (engine is None) & isinstance(source, str):
            # from the beginning (^) match any character (.) zero
            # or more times (*) until last (. - require escape with \)
            file_suffix = re.sub("^.*\.", "", source)

            assert file_suffix in cls.file_suffix_engine_map, \
                f"file_suffix: {file_suffix} not in file_suffix_engine_map: {cls.file_suffix_engine_map}"

            engine = cls.file_suffix_engine_map[file_suffix]

            if verbose:
                print(f"engine not provide, inferred '{engine}' from file suffix '{file_suffix}'")

        # connect / read in data

        # available pandas read method
        pandas_read_methods = [i for i in dir(pd) if re.search("^read", i)]

        # xr.open_dataset engines
        xr_dataset_engine = ["netcdf4", "scipy", "pydap", "h5netcdf", "pynio", "cfgrib", \
                             "pseudonetcdf", "zarr"]

        # self.data_source = None
        # read in via pandas
        if engine in pandas_read_methods:
            source = getattr(pd, engine)(source, **kwargs)
        # xarray open_dataset
        elif engine in xr_dataset_engine:
            source = xr.open_dataset(source, engine=engine, **kwargs)
        # or hdfstore
        elif engine == "HDFStore":
            source = pd.HDFStore(source, mode="r", **kwargs)
        else:
            warnings.warn(f"file: {source} was not read in as\n"
                          f"engine: {engine}\n was not understood. "
                          f"source has not been changed")

        return source

    @staticmethod
    def add_data_to_col(df, add_data_to_col=None, verbose=False):
        """
        Adds new data to an existing column or creates a new column with the provided data in a DataFrame.

        This function takes a DataFrame and a dictionary with the column name as the key and the data to be
        added as the value. It can handle scalar values or lists of values, and will replicate the DataFrame
        rows for each value in the list.

        Parameters
        ----------
        df : pandas.DataFrame
            The input DataFrame to which data will be added or updated.
        add_data_to_col : dict, optional
            A dictionary with the column name (key) and data to be added (value). The data can be a scalar value
            or a list of values. If a list of values is provided, the DataFrame rows will be replicated for each
            value in the list. If None, an empty dictionary will be used. Default is None.
        verbose : bool, default False.
            If True, the function will print progress messages.

        Returns
        -------
        df : pandas.DataFrame
            The DataFrame with the updated or added columns.

        Raises
        ------
        AssertionError
            If the add_data_to_col parameter is not a dictionary.

        Notes
        -----
        This method adds data to a specified column in a pandas DataFrame repeatedly.
        The method creates a copy of the DataFrame for each entry in the data to be added,
        and concatenates them to create a new DataFrame with the added data.

        Examples
        --------
        >>> import pandas as pd
        >>> from GPSat.dataloader import DataLoader
        >>> df = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})
        >>> updated_df = DataLoader.add_data_to_col(df, add_data_to_col={"C": [7, 8]})
        >>> print(updated_df)
           A  B  C
        0  1  4  7
        1  2  5  7
        2  3  6  7
        0  1  4  8
        1  2  5  8
        2  3  6  8

        >>> len(df)
        3
        >>> out = DataLoader.add_data_to_col(df, add_data_to_col={"a": [1,2,3,4]})
        >>> len(out)
        12
        >>> out = DataLoader.add_data_to_col(df, add_data_to_col={"a": [1,2,3,4], "b": [5,6,7,8]})
        >>> len(out)
        48
        """
        # add columns - repeatedly (e.g. dates)
        if add_data_to_col is None:
            add_data_to_col = {}

        assert isinstance(add_data_to_col, dict), f"add_cols expected to be dict, got: {type(add_data_to_col)}"

        # for each element in add_data_to_col will copy location data
        # TODO: is there a better way of doing this?

        for k, v in add_data_to_col.items():
            tmp = []
            if isinstance(v, (int, str, float)):
                v = [v]
            if verbose:
                print(f"adding column: {k}, which has {len(v)} entries\n"
                      f" current df size: {len(df)} -> new df size: {len(df) * len(v)}")

            for vv in v:
                _ = df.copy(True)
                _[k] = vv
                tmp += [_]
            df = pd.concat(tmp, axis=0)

        return df


    @classmethod
    @timer
    def load(cls, source,
             where=None,
             engine=None,
             table=None,
             source_kwargs=None,
             col_funcs=None,
             row_select=None,
             col_select=None,
             filename=None,
             reset_index=False,
             add_data_to_col=None,
             close=False,
             verbose=False,
             **kwargs):
        """
        Load data from various sources and (optionally)
        apply selection of columns / rows and add / modify columns


        Parameters
        ----------
        source: str, pd.DataFrame, pd.Series, pd.HDFStore, xr.dataset, default None. If str will try to convert to other types

        where: dict or list of dict, default None: Used when querying HDFStore, DataSet, Dataarray
            can use list of one or more dict, each containing "col", "comp", "val"
            "col" - refers to a 'column' (or variable for xarray objects)
            "comp" - is type of comparison to apply e.g. ==, !=, >=, >, <=, <
            "val" - value to be compare with
            e.g. where = [ {"col": "A", "comp": ">=", "val": 0}] will select entries where columns "A" is greater than 0
            NOTE: think of this as a database query, with the where used to read data from the file system into memory
        engine: str or None, default None. Specify the type of 'engine' to use to read in data.
            If not supplied will be inferred by source if source is string.
            Valid values: "HDFStore", "netcdf4", "scipy", "pydap", "h5netcdf", "pynio", "cfgrib",
            "pseudonetcdf", "zarr" or any of Pandas "read_*"
        table: str or None, default None. Used only if source is HDFStore (or is converted to one) and is required if so.
            Should be a valide key/table in HDFStore.
        source_kwargs: dict or None, default None. Additional arguments to be provided to _get_source_from_str
        col_funcs: dict or None. If dict will be provide to add_cols method to add or modify columns.
        row_select: dict, list of dict, or None, default None. Used to select a subset of data AFTER
            data initially read into memory. Can be same type of input as 'where' i.e.
            {"col": "A", "comp": ">=", "val": 0} or use col_funcs that return bool array
            e.g. {"func": "lambda x: ~np.isnan(x)", "col_args": 1}, see help(utils.config_func) for more details
        col_select: list of str or None, default None. If list of str will return a subset of columns using col_select,
            all values must be valid. If None all columns will be returned.
        filename: str or None, default None. Used by add_cols method
        reset_index: bool, default True. apply reset_index(inplace=True) before returning?
        add_data_to_col:
        close: bool, default False
            see DataLoader.data_select for details
        verbose: bool, default False
        kwargs: Additional arguments to be provided to data_select method

        Returns
        -------
        pd.DataFrame

        """
        # TODO: doc string needs more work

        # given a source: DataFrame,Series,Dataset,HDFStore or str
        # - read in data (possible using where), add columns, select subset of rows and columns

        # if the source is a string - process to get valid source: DataFrame, DataSet, HDFStore
        if isinstance(source, str):
            if source_kwargs is None:
                source_kwargs = {}
            # if provide string as a source then set close to True (used for HDFStore)
            close = True
            source = cls._get_source_from_str(source, engine=engine, **source_kwargs)

        # --
        # load data
        # --

        # TODO: review some of these hardcoded defaults below - should they be options? - specifically close
        df = cls.data_select(obj=source,
                             where=where,
                             table=table,
                             return_df=True,
                             reset_index=reset_index,
                             drop=True,
                             copy=True,
                             close=close,
                             **kwargs)

        # ---
        # modify dataframe: add columns, select rows/cols
        # ---

        df = cls._modify_df(df,
                            col_funcs=col_funcs,
                            filename=filename,
                            row_select=row_select,
                            col_select=col_select,
                            add_data_to_col=add_data_to_col,
                            verbose=verbose)
        return df

    @classmethod
    def _modify_df(cls, df,
                   col_funcs=None,
                   filename=None,
                   row_select=None,
                   col_select=None,
                   add_data_to_col=None,
                   verbose=False):

        # ---
        # add any new columns with specified values (repeatedly)
        # ---

        df = cls.add_data_to_col(df,
                                 add_data_to_col=add_data_to_col,
                                 verbose=verbose)

        # ----
        # apply column functions - used to add new columns
        # ----

        cls.add_cols(df,
                     col_func_dict=col_funcs,
                     verbose=verbose,
                     filename=filename)

        # ----
        # select rows - similar to where but done after data loaded into memory
        # ----

        select = cls.row_select_bool(df,
                                     row_select=row_select,
                                     verbose=verbose,
                                     filename=filename)

        # select subset of data
        if verbose >= 3:
            print(f"selecting {select.sum()}/{len(select)} rows")
        df = df.loc[select, :]

        # ----
        # select columns
        # ----

        if col_select is None:
            col_select = slice(None)
        else:
            missing_columns = []
            for c in col_select:
                if c not in df:
                    missing_columns.append(c)
            assert len(missing_columns) == 0, f"columns were provide, but {missing_columns} are not in obj (dataframe)"

        df = df.loc[:, col_select]

        return df

    @staticmethod
    def is_list_of_dict(lst):
        """
        Checks if the given input is a list of dictionaries.

        This utility function tests if the input is a list where all elements are instances of the `dict` type.

        Parameters
        ----------
        lst : list
            The input list to be checked for containing only dictionaries.

        Returns
        -------
        bool
            True if the input is a list of dictionaries, False otherwise.

        Examples
        --------
        >>> from GPSat.dataloader import DataLoader
        >>> DataLoader.is_list_of_dict([{"col": "t", "comp": "==", "val": 1}])
        True

        >>> DataLoader.is_list_of_dict([{"a": 1, "b": 2}, {"c": 3, "d": 4}])
        True

        >>> DataLoader.is_list_of_dict([1, 2, 3])
        False

        >>> DataLoader.is_list_of_dict("not a list")
        False

        """
        if isinstance(lst, list):
            return all([isinstance(_, dict) for _ in lst])
        else:
            return False

    @staticmethod
    def _hdfstore_where_from_dict(wd):
        col = wd['col']
        comp = wd['comp']
        val = wd['val']
        if isinstance(val, str):
            val = f'"{val}"'
        elif isinstance(val, (int, float, bool, list)):
            val = str(val)
        elif isinstance(val, (np.datetime64, pd._libs.tslibs.timestamps.Timestamp)):
            val = f'"{val}"'
        return "".join([col, comp, val])

    @staticmethod
    def _bool_xarray_from_where(obj, wd):

        # TODO: check negating the results works expected
        wd = wd.copy()
        negate = wd.pop("negate", False)

        # unpack values
        # - allow col to also be coord?
        col = wd['col']
        comp = wd['comp']
        val = wd['val']

        # checks
        assert isinstance(obj, (xr.core.dataarray.DataArray, xr.core.dataarray.Dataset))
        assert col in obj.coords, f"'col': {col} is not in coords: {obj.coords._names}"
        assert comp in [">=", ">", "==", "<", "<="], f"comp: {comp} is not valid"

        # check dtype for datetime
        if np.issubdtype(obj.coords[wd['col']], np.datetime64):
            if isinstance(val, str):
                val = np.datetime64(val)
            # check if int or float -

        # create a function for comparison
        tmp_fun = lambda x, y: eval(f"x {comp} y")

        out = tmp_fun(obj.coords[col], val)

        if negate:
            out = ~out

        return out

    @staticmethod
    def _bool_numpy_from_where(obj, wd):
        """

        Parameters
        ----------
        obj: DataFrame or Series
        wd: dict containing where conditions (?)

        Returns
        -------

        """
        # perform simple comparison?
        # wd - dict with 'col', 'comp', 'val'
        # e.g. {"col": "t", "comp": "<=", "val": 4}
        simple_compare = all([i in wd for i in ['col', 'comp', 'val']])

        # make a copy of 'where' dict, just so can pop out 'negate' without affecting original
        wd = wd.copy()
        negate = wd.pop("negate", False)

        if simple_compare:

            # unpack values
            # - allow col to also be coord?
            col = wd['col']
            comp = wd['comp']
            val = wd['val']

            # checks
            assert isinstance(obj, (pd.Series, pd.DataFrame))
            assert col in obj.columns, f"col: '{col}' is not in coords: {obj.columns}"
            assert comp in [">=", ">", "==", "<", "<="], f"comp: {comp} is not valid"

            # # check dtype for datetime - not needed if using a Series
            # if np.issubdtype(obj.coords[wd['col']], np.datetime64):
            #     if isinstance(val, str):
            #         val = np.datetime64(val)
            #     # check if int or float -

            # create a function for comparison
            tmp_fun = lambda x, y: eval(f"x {comp} y")

            out = tmp_fun(obj[col], val)

        # otherwise  use config_func
        else:
            out = config_func(df=obj, **wd)
            if str(out.dtype) != 'bool':
                warnings.warn("not returning an array with dtype bool")

        # if negate is True then flip the array
        if negate:
            out = ~out
            # this can be slightly faster, does not require creation of new array
            # np.invert(out, out=out)

        return out

    @classmethod
    def download_data(cls, id_files=None, id=None, file=None, unzip=False):
        # wrapper for downloading data from good drive using
        pass

    @staticmethod
    def get_run_info(script_path=None):
        """
        Retrieves information about the current Python script execution environment,
        including run time, Python executable path, and Git information.

        This function collects information about the current script execution environment,
        such as the date and time when the script is executed, the path of the Python interpreter,
        the script's file path, and Git information (if available).

        Parameters
        ----------
        script_path : str, default None
            The file path of the currently executed script. If None which will try to retrieve the file path
            automatically.

        Returns
        -------
        run_info : dict
            A dictionary containing the following keys:
            - 'run_time': The date and time when the script was executed, formatted as 'YYYY-MM-DD HH:MM:SS'.
            - 'python_executable': The path of the Python interpreter.
            - 'script_path': The absolute file path of the script (if available).
            - Git-related keys: 'git_branch', 'git_commit', 'git_url', and 'git_modified' (if available).

        Examples
        --------
        >>> from GPSat.dataloader import DataLoader
        >>> run_info = DataLoader.get_run_info()
        >>> print(run_info)
        {
            "run_time": "2023-04-28 10:30:00",
            "python_executable": "/usr/local/bin/python3.9",
            "script_path": "/path/to/your/script.py",
            "branch": "main",
            "commit": "123abc",
            "remote": ["https://github.com/user/repo.git" (fetch),"https://github.com/user/repo.git" (push)]
            "details": ['commit 123abc',
              'Author: UserName <username42@gmail.com>',
              'Date:   Fri Apr 28 07:22:31 2023 +0100',
              ':bug: fix ']
            "modified" : ['list_of_files.py', 'modified_since.py', 'last_commit.py']
        }

        """
        # TODO: this method does not really fit with close, move else where?
        run_info = {
            "run_time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "python_executable": sys.executable,
        }
        try:
            run_info['script_path'] = os.path.abspath(script_path)
        except NameError as e:
            pass
        except TypeError as e:
            pass
        # get git information - branch, commit, etc
        try:
            git_info = get_git_information()
        except Exception as e:
            git_info = {}

        run_info = {**run_info, **git_info}

        return run_info

    @classmethod
    @timer
    def bin_data_by(cls,
                    df,
                    by_cols=None, val_col=None,
                    x_col='x', y_col='y',
                    x_range=None, y_range=None,
                    grid_res=None, bin_statistic="mean",
                    limit=10000):

        """
        Bins the input DataFrame df based on the given columns and computes the bin statistics for a specified value column.

        This function takes a DataFrame, filters it based on the unique combinations of the by_cols column values, and
        then bins the data in each filtered DataFrame based on the x_col and y_col column values. It computes the
        bin statistic for the specified val_col and returns the result as an xarray DataArray. The output DataArray
        has dimensions 'y', 'x', and the given by_cols.

        Parameters
        ----------
        df : pandas.DataFrame
            The input DataFrame to be binned.
        by_cols : str or list[str] or tuple[str]
            The column(s) by which the input DataFrame should be filtered.
            Unique combinations of these columns are used to create separate DataFrames for binning.
        val_col : str
            The column in the input DataFrame for which the bin statistics should be computed.
        x_col : str, optional, default='x'
            The column in the input DataFrame to be used for binning along the x-axis.
        y_col : str, optional, default='y'
            The column in the input DataFrame to be used for binning along the y-axis.
        x_range : tuple, optional
            The range of the x-axis values for binning. If None, the minimum and maximum x values are used.
        y_range : tuple, optional
            The range of the y-axis values for binning. If None, the minimum and maximum y values are used.
        grid_res : float, optional
            The resolution of the grid used for binning. If None, the resolution is calculated based on the input data.
        bin_statistic : str, optional, default="mean"
            The statistic to compute for each bin. Supported values are "mean", "median", "sum", "min", "max", and "count".

        limit : int, optional, default=10000
            The maximum number of unique combinations of the by_cols column values allowed.
            Raises an AssertionError if the number of unique combinations exceeds this limit.

        Returns
        -------
        out : xarray.Dataset
            The binned data as an xarray Dataset with dimensions 'y', 'x', and the given by_cols.
            Raises

        Raises
        ------
        DeprecationWarning
        If the deprecated method DataLoader.bin_data_by(...) is used instead of DataPrep.bin_data_by(...).

        AssertionError
        If any of the input parameters do not meet the specified conditions.


        """


        # TODO: this method may be more suitable in a different class - a DataPrep class
        # TODO: add doc string
        # TODO: add print statements (given a verbose level)
        # TODO: grid_res should be in same dimensions as x,y
        # --
        # checks
        # --

        warnings.warn("\nDataLoader.bin_data_by(...) is deprecated, using DataPrep.bin_data_by(...) instead",
                      category=DeprecationWarning)

        assert by_cols is not None, f"by_col needs to be provided"
        if isinstance(by_cols, str):
            by_cols = [by_cols]
        assert isinstance(by_cols, (list, tuple)), f"by_cols must be list or tuple, got type: {type(by_cols)}"

        for bc in by_cols:
            assert bc in df, f"by_cols value: {bc} is not in df.columns: {df.columns}"

        assert val_col in df, f"val_col: {val_col} is not in df.columns: {df.columns}"
        assert x_col in df, f"x_col: {x_col} is not in df.columns: {df.columns}"
        assert y_col in df, f"y_col: {y_col} is not in df.columns: {df.columns}"

        # ----

        # get the common pairs
        bc_pair = df.loc[:, by_cols].drop_duplicates()

        assert len(bc_pair) < limit, f"number unique values of by_cols found in data: {len(bc_pair)} > limit: {limit} " \
                                     f"are you sure you want this many? if so increase limit"

        da_list = []
        for idx, bcp in bc_pair.iterrows():

            # select data
            select = np.ones(len(df), dtype=bool)
            for bc in by_cols:
                select &= (df[bc] == bcp[bc]).values
            df_bin = df.loc[select, :]

            # store the 'by' coords
            by_coords = {bc: [bcp[bc]] for bc in by_cols}

            b, xc, yc = cls.bin_data(df_bin,
                                     x_range=x_range,
                                     y_range=y_range,
                                     grid_res=grid_res,
                                     x_col=x_col,
                                     y_col=y_col,
                                     val_col=val_col,
                                     bin_statistic=bin_statistic,
                                     return_bin_center=True)
            # add extra dimensions to binned data
            b = b.reshape(b.shape + (1,) * len(by_cols))
            # store data in DataArray
            # TODO: review y,x order - here assumes y is first dim. for symmetrical grids it doesn't matter
            coords = {**{'y': yc, 'x': xc}, **by_coords}
            da = xr.DataArray(data=b,
                              dims=['y', 'x'] + by_cols,
                              coords=coords,
                              name=val_col)
            da_list += [da]

        # combine into a single Dataset
        out = xr.combine_by_coords(da_list)
        return out

    @staticmethod
    def bin_data(
            df,
            x_range=None,
            y_range=None,
            grid_res=None,
            x_col="x",
            y_col="y",
            val_col=None,
            bin_statistic="mean",
            return_bin_center=True):
        """
        Bins data from a given DataFrame into a 2D grid, applying the specified statistical function
        to the data in each bin.

        This function takes a DataFrame containing x, y, and value columns and bins the data into a 2D grid.
        It returns the resulting grid, as well as the x and y bin edges or centers,
        depending on the value of return_bin_center.

        Parameters
        ----------
        df : pd.DataFrame
            The input DataFrame containing the data to be binned.
        x_range : list or tuple of floats, optional
            The range of x values, specified as [min, max]. If not provided, a default value of [-4500000.0, 4500000.0]
            will be used.
        y_range : list or tuple of floats, optional
            The range of y values, specified as [min, max]. If not provided, a default value of [-4500000.0, 4500000.0]
            will be used.
        grid_res : float or None.
            The grid resolution, expressed in kilometers. This parameter must be provided.
        x_col : str, default is "x".
            The name of the column in the DataFrame containing the x values.
        y_col : str, default is "y".
            The name of the column in the DataFrame containing the y values.
        val_col : str, optional
            The name of the column in the DataFrame containing the values to be binned. This parameter must be provided.
        bin_statistic : str, default is "mean".
            The statistic to apply to the binned data. Options are 'mean', 'median', 'count', 'sum', 'min', 'max', or
            a custom callable function.
        return_bin_center : bool,  default is True.
            If True, the function will return the bin centers instead of the bin edges.

        Returns
        -------
        binned_data : numpy.ndarray
            The binned data as a 2D grid.
        x_out : numpy.ndarray
            The x bin edges or centers, depending on the value of return_bin_center.
        y_out : numpy.ndarray
            The y bin edges or centers, depending on the value of return_bin_center.

        """
        # TODO: complete doc string
        # TODO: move defaults out of bin_data to bin_data_by?
        # TODO: double check get desired shape, dim alignment if x_range != y_range

        warnings.warn("\nDataLoader.bin_data(...) is deprecated, using DataPrep.bin_data(...) instead",
                      category=DeprecationWarning)

        # ---
        # check inputs, handle defaults

        assert val_col is not None, "val_col - the column containing values to bin cannot be None"
        assert grid_res is not None, "grid_res is None, must be supplied - expressed in km"
        assert len(df) > 0, f"dataframe (df) provide must have len > 0"

        if x_range is None:
            x_range = [-4500000.0, 4500000.0]
            print(f"x_range, not provided, using default: {x_range}")
        assert x_range[0] < x_range[1], f"x_range should be (min, max), got: {x_range}"

        if y_range is None:
            y_range = [-4500000.0, 4500000.0]
            print(f"y_range, not provided, using default: {y_range}")
        assert y_range[0] < y_range[1], f"y_range should be (min, max), got: {y_range}"

        assert len(x_range) == 2, f"x_range expected to be len = 2, got: {len(x_range)}"
        assert len(y_range) == 2, f"y_range expected to be len = 2, got: {len(y_range)}"

        # if grid_res is None:
        #     grid_res = 50
        #     print(f"grid_res, not provided, using default: {grid_res}")

        # bin parameters
        assert x_col in df, f"x_col: {x_col} is not in df columns: {df.columns}"
        assert y_col in df, f"y_col: {y_col} is not in df columns: {df.columns}"
        assert val_col in df, f"val_col: {val_col} is not in df columns: {df.columns}"

        x_min, x_max = x_range[0], x_range[1]
        y_min, y_max = y_range[0], y_range[1]

        # number of bin (edges)
        n_x = ((x_max - x_min) / grid_res) + 1
        n_y = ((y_max - y_min) / grid_res) + 1
        n_x, n_y = int(n_x), int(n_y)

        # NOTE: x will be dim 1, y will be dim 0
        x_edge = np.linspace(x_min, x_max, int(n_x))
        y_edge = np.linspace(y_min, y_max, int(n_y))

        # extract values
        x_in, y_in, vals = df[x_col].values, df[y_col].values, df[val_col].values

        # apply binning
        binned = scst.binned_statistic_2d(x_in, y_in, vals,
                                          statistic=bin_statistic,
                                          bins=[x_edge,
                                                y_edge],
                                          range=[[x_min, x_max], [y_min, y_max]])

        xy_out = x_edge, y_edge
        # return the bin centers, instead of the edges?
        if return_bin_center:
            x_cntr, y_cntr = x_edge[:-1] + np.diff(x_edge) / 2, y_edge[:-1] + np.diff(y_edge) / 2
            xy_out = x_cntr, y_cntr

        # TODO: if output is transpose, should the x,y (edges or centers) be swapped?
        return binned[0].T, xy_out[0], xy_out[1]

    @staticmethod
    @timer
    def kdt_tree_list_for_local_select(df, local_select):
        """

        Pre-calculates a list of KDTree objects for selecting points within a radius based on the local_select input.

        Given a DataFrame and a list of local selection criteria, this function builds a list of KDTree objects that
        can be used for spatially selecting points within specified radii.

        Parameters
        ----------
        df : pd.DataFrame
            The input DataFrame containing the data to be used for KDTree construction.
        local_select : list of dict
            A list of dictionaries containing the selection criteria for each local select. Each dictionary should
            have the following keys:
            - 'col': The name of the column(s) used for spatial selection. Can be a single string or a list of strings.
            - 'comp': The comparison operator, either '<' or '<='. Currently, only less than comparisons are supported
              for multi-dimensional values.

        Returns
        -------
        out : list
            A list of KDTree objects or None values, where each element corresponds to an entry in the local_select
            input. If an entry in local_select has a single string for 'col', the corresponding output element will be
            None. Otherwise, the output element will be a KDTree object built from the specified columns.

        Examples
        --------
        >>> import pandas as pd
        >>> from GPSat.dataloader import DataLoader
        >>> df = pd.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6]})
        >>> local_select = [{"col": ["x", "y"], "comp": "<"}]
        >>> kdt_trees = DataLoader.kdt_tree_list_for_local_select(df, local_select)
        >>> print(kdt_trees)

        """
        # TODO: remove this method if it's being used

        # pre calculate KDTree objects
        out = []
        for idx, ls in enumerate(local_select):
            col = ls['col']
            comp = ls['comp']

            # single (str) entry for column
            if isinstance(col, str):
                out += [None]
            # otherwise use a KD tree to select points within a radius
            else:
                assert comp in ["<", "<="], f"for multi dimensional values only less than comparison handled"
                for c in col:
                    assert c in df
                # creating a kdt tree can take (say) 90ms for 3.7k rows
                kdt = KDTree(df.loc[:, col].values)
                out += [kdt]

        return out

    @classmethod
    @timer
    def local_data_select(cls, df, reference_location, local_select, kdtree=None, verbose=True):

        # use a bool to select values
        select = np.ones(len(df), dtype='bool')

        # convert reference location to dict (if not already)
        reference_location = pandas_to_dict(reference_location)

        # increment over each of the selection criteria
        for idx, ls in enumerate(local_select):
            col = ls['col']
            comp = ls['comp']
            if verbose:
                print(ls)

            # single (str) entry for column
            if isinstance(col, str):
                # TODO: here just use data_select method?
                assert col in df, f"col: {col} is not in data - {df.columns}"
                assert col in reference_location, f"col: {col} is not in reference_location - {reference_location.keys()}"
                assert comp in [">=", ">", "==", "<", "<="], f"comp: {comp} is not valid"

                tmp_fun = lambda x, y: eval(f"x {comp} y")
                _ = tmp_fun(df.loc[:, col], reference_location[col] + ls['val'])
                select &= _
            # otherwise use a KD tree to select points within a radius
            else:
                assert comp in ["<", "<="], f"for multi dimensional values only less than comparison handled"
                for c in col:
                    assert c in df, f"column: {c} is not in df.columns: {df.columns}"
                    assert c in reference_location, f"col: {col} is not in reference_location - {reference_location.keys()}"
                # creating a kdt tree can take (say) 90ms for 3.7k rows
                # - using pre-calculated kd-tree can reduce run time (if being called often)
                if kdtree is not None:
                    if isinstance(kdtree, list):
                        kdt = kdtree[idx]
                    else:
                        kdt = kdtree
                    assert isinstance(kdt, KDTree), f"kdtree did not provide a KDTree, got type: {type(kdt)}"
                else:
                    kdt = KDTree(df.loc[:, col].values)

                in_ids = kdt.query_ball_point(x=[reference_location[c] for c in col],
                                              r=ls['val'])
                # create a bool array of False, then populate locations with True
                _ = np.zeros(len(df), dtype=bool)
                _[in_ids] = True
                select &= _

        # data to be used by a local model
        return df.loc[select, :]

    @staticmethod
    @timer
    def make_multiindex_df(idx_dict, **kwargs):
        """
        Create a multi-indexed DataFrame from the provided index dictionary for each keyword argument supplied

        This function creates a multi-indexed DataFrame, with each row having the same multi-index value
        The index dictionary serves as the levels and labels for the multi-index, while the keyword arguments
        provide the data.

        Parameters
        ----------
        idx_dict : dict or pd.Series
            A dictionary or pandas Series containing the levels and labels for the multi-index.
        **kwargs : dict
            Keyword arguments specifying the data and column names for the resulting DataFrame.
            The data can be of various types: int, float, bool, np.ndarray, pd.DataFrame, dict, or tuple.
            This data will be transformed into a DataFrame, where the multi-index will be added

        Returns
        -------
        dict
            A dictionary containing the multi-indexed DataFrames with keys corresponding to the keys of
            provided keyword arguments.

        Examples
        --------
        >>> import numpy as np
        >>> import pandas as pd
        >>> from GPSat.dataloader import DataLoader
        >>> idx_dict = {"year": 2020, "month": 1}
        >>> data = pd.DataFrame({"x": np.arange(10)})
        >>> df = pd.DataFrame({"y": np.arange(3)})
        >>> DataLoader.make_multiindex_df(idx_dict, data=data, df=df)
        {'data': <pandas.DataFrame (multiindexed) with shape (3, 4)>}
        """

        out = {}
        # TODO: review if there is a better way of providing data to create a multi index with
        if isinstance(idx_dict, pd.Series):
            idx_dict = idx_dict.to_dict()

        assert isinstance(idx_dict, dict), "idx expected to dict, pd.Series"
        for k, v in kwargs.items():

            if isinstance(v, (int, float, bool)):
                df = pd.DataFrame({k: v}, index=[0])
            elif isinstance(v, np.ndarray):
                assert len(v.shape) > 0, "np.array provided but has no shape, provide as (int,float,bool)" \
                                         " or array with shape"
                # move to DataArray -> DataFrame
                dummy_dims = [f"_dim_{i}" for i in range(len(v.shape))]
                da = xr.DataArray(v, name=k, dims=dummy_dims)
                df = da.to_dataframe().reset_index()

            elif isinstance(v, pd.DataFrame):
                df = v
            elif isinstance(v, dict):
                # TODO: allow a dict to be provide with values and dims specified
                print("dict provided, not handled yet, skipping")
                continue
            elif isinstance(v, tuple):
                # if tuple provided expected first entry is data, second is for cords
                if len(v) > 2:
                    warnings.warn("only first two entries are being used")
                da = xr.DataArray(v[0], name=k, coords=v[1])
                df = da.to_dataframe().reset_index()

            # create a multi index - is this the best way to do this?
            tmp_indx = pd.MultiIndex.from_arrays([np.full(len(df), v)
                                                  for k, v in idx_dict.items()],
                                                 names=list(idx_dict.keys()))
            df.index = tmp_indx

            out[k] = df

        return out

    @staticmethod
    def mindex_df_to_mindex_dataarray(df, data_name,
                                      dim_cols=None,
                                      infer_dim_cols=True,
                                      index_name="index"):

        # NOTE: df is manipulated by reference - provide copy if need be
        # TODO: might want to return numpy directly, skip the DataArray bit
        # TODO: should check if index_name arbitrary name conflicts with any dim_names, data_name
        assert data_name in df, f"data_name: {data_name} is not "
        # get the multi-index
        # TODO: ensure / test indices are in correct order
        midx = df.index.unique()

        # dimension columns (in addition to the (multi) index)
        if dim_cols is None:
            if infer_dim_cols:
                # dimension index columns should look like: _dim_#
                # - where # corresponds to the dimension # (starting from zero) and
                # - the column value is the location on that dimension
                dim_cols = [c for c in df.columns if re.search("^_dim_\d", c)]
            else:
                dim_cols = []

        # replace the multi-index with a integer placeholder
        # - do this with a map from tuples to integer
        idx_map = {_: i for i, _ in enumerate(midx)}
        df.index = df.index.map(idx_map)

        # assign an arbitrary name to index
        df.index.name = index_name

        # add dim_cols to multi-index
        df.set_index(dim_cols, append=True, inplace=True)

        # HACK: want to have a dim/coordinate as multi-index
        # da = df.to_xarray()
        da = xr.DataArray.from_series(df[data_name])
        new_coords = {"index": midx, **{k: v for k, v in da.coords.items() if k in dim_cols}}
        da = xr.DataArray(da.data, coords=new_coords)

        return da

    @classmethod
    def generate_local_expert_locations(cls,
                                        loc_dims,
                                        ref_data=None,
                                        format_type=None,
                                        masks=None,
                                        include_col="include",
                                        col_func_dict=None,
                                        row_select=None,
                                        keep_cols=None,
                                        sort_by=None):
        # locations (centers) of local experts
        # loc_dims specify the core dimensions for the local expert locations - more can be added later
        # - loc_dims can either provide a list/arrays or reference col / coord in ref_data
        # ref_data can either be DataFrame or DataArray - can be used to get dimension values
        # - DataFrame should have dimensions in columns (will take unique)
        # - DataArray will have dims in coords

        assert isinstance(loc_dims, dict), f"loc_dims must be a dict"  # ?

        # TODO: should masks be provide or calculated on the fly
        # masks
        masks = [] if masks is None else masks
        masks = masks if isinstance(masks, (list, tuple)) else list(masks)

        # TODO: review the need for format_type and specifying how intermediate steps are handled

        # how should intermediate steps be handled? - review this
        # - either with DataFrame or DataArray
        if format_type is None:
            if isinstance(ref_data, (pd.DataFrame, pd.Series)):
                format_type = "DataFrame"
            elif isinstance(ref_data, (xr.DataArray, xr.Dataset)):
                format_type = "DataArray"

        valid_format_types = ['DataFrame', "DataArray"]
        assert format_type in valid_format_types, f"format_type: {valid_format_types}"
        print(f"using {format_type} for intermediate steps to get expert locations")

        if format_type == "DataArray":
            # create coordinate dict
            coord_dict = {}
            for k, v in loc_dims.items():
                # if dim is a str - then is it expected to dim values from ref_data
                if isinstance(v, str):
                    assert ref_data is not None, f"in loc_dim: key={k} has str value: {v} - " \
                                                 f"expected to get from ref_data but it is None"
                    assert v in ref_data.coords, f"{v} is not in ref_data: {ref_data.columns}"
                    coord_dict[k] = ref_data.coords[v].values
                elif isinstance(v, (list, tuple)):
                    coord_dict[k] = np.array(v)
                elif isinstance(v, np.ndarray):
                    coord_dict[k] = v
                else:
                    warnings.warn(f"key {k} with has value with type: {type(v)} not handled - assigning as is")
                    coord_dict[k] = v

            # create a DataArray - used to specify local expert locations - default will be everywhere
            # need (?) to create an intermediate object
            locs = xr.DataArray(True, coords=coord_dict, dims=list(coord_dict.keys()), name=include_col)

            # apply masks - coords expected to align
            for m in masks:
                locs &= m

            locs = locs.to_dataframe().reset_index()

        elif format_type == "DataFrame":
            coord_dict = {}
            for k, v in loc_dims.items():
                # if dim is a str - then is it expected to dim values from ref_data
                if isinstance(v, str):
                    assert ref_data is not None, f"in loc_dim: key={k} has str value: {v} - " \
                                                 f"expected to get from ref_data but it is None"
                    assert v in ref_data, f"{v} is not in ref_data: {list(ref_data.coords.keys())}"
                    coord_dict[k] = ref_data[v].unique().values

            midx = pd.MultiIndex.from_product([v for v in coord_dict.values()],
                                              names=list(coord_dict.keys()))
            locs = pd.DataFrame(True, index=midx, columns=[include_col]).reset_index()

            # TODO: merge masks onto locs, update include column by &= some reference column
            #  - will need reference column and merge on columns specified
            warnings.warn(f"NOT IMPLEMENTED: applying masks for format_type={format_type}")

        # drop values where include is False
        locs = locs.loc[locs[include_col]]

        # apply column function - to add new columns
        cls.add_cols(locs, col_func_dict)

        # (additional) select rows
        if row_select is not None:
            select = cls.row_select_bool(locs, row_select=row_select)
            locs = locs.loc[select, :]

        # store rows - e.g. by date?
        if sort_by is not None:
            locs.sort_values(by=sort_by, inplace=True)

        # select a subset of columns
        if keep_cols is not None:
            locs = locs.loc[:, keep_cols]

        return locs

    @staticmethod
    def get_masks_for_expert_loc(ref_data, el_masks=None, obs_col=None):
        # TODO: get_masks_for_expert_loc requires more thought and needs cleaning
        #  - allow to read data from file system? provide different reference data?
        #  - let lel_mask be only list of dict?

        # get a list of mask to apply to where local experts should be located
        # ref_data is reference data - should be optional

        if el_masks is None:
            el_masks = []
        el_masks = el_masks if isinstance(el_masks, list) else list(el_masks)

        masks = []
        for m in el_masks:
            if isinstance(m, str):
                if m == "had_obs":
                    # TODO: allow for reference data to be pd.DataFrame
                    assert isinstance(ref_data, (xr.DataArray, xr.Dataset))
                    # create a mask for were to (not to) have a local expert location
                    no_obs = xr.apply_ufunc(np.isnan, ref_data[obs_col])
                    had_obs = xr.apply_ufunc(np.any, ~no_obs, input_core_dims=[['date']], vectorize=True)
                    masks += [had_obs]
                else:
                    print(f"mask: {m} not understood")
            elif isinstance(m, dict):

                if "grid_space" in m:
                    # TODO: allow for reference data to be pd.DataFrame - may want to do it differently
                    assert isinstance(ref_data, (xr.DataArray, xr.Dataset))
                    # create an array with True values regularly spaced - to create a coarse grid local expert locations

                    coord_dict = {i: ref_data.coords[i].values for i in m['dims']}
                    tmp_shape = tuple([len(ref_data.coords[i]) for i in m['dims']])
                    reg_space = sparse_true_array(shape=tmp_shape, grid_space=m['grid_space'])
                    reg_space = xr.DataArray(reg_space, coords=coord_dict)
                    masks += [reg_space]

        return masks

    @staticmethod
    def get_where_list_legacy(read_in_by=None, where=None):
        """
        generate a list (of lists) of where conditions that can be consumed by pd.HDFStore(...).select


        Parameters
        ----------
        read_in_by: dict of dict or None. sub dict must contain 'values', 'how'
        where: str or None. Used if read_in_by is not provided

        Returns
        -------
        list of list containing string where conditions

        """
        # TODO: review / refactor get_where_list_legacy (or just remove?)
        # create a list of 'where' conditions that can be used

        if read_in_by is not None:

            assert isinstance(read_in_by, dict), f"read_in_by provided, expected to be dict, got: {type(read_in_by)}"

            # TODO: wrap the following up into a method - put in DataPrep
            if where is not None:
                warnings.warn("'read_in_by' is specified, as is 'where' in 'input' of config, will ignore 'where'")

            where_dict = {}
            for k, v in read_in_by.items():
                vals = v['values']
                how = v['how']

                if isinstance(vals, dict):
                    func = vals.pop('func')
                    # if func is a str - expect it to be a funciton to evaluate
                    # - currently expects to be lambda function
                    # TODO: allow for func to be non lambda - i.e. imported - see config_func
                    if isinstance(func, str):
                        func = eval(func)
                    vals = func(**vals)

                else:
                    pass

                # force vals to be an array
                if isinstance(vals, (int, float, str)):
                    vals = [vals]
                if not isinstance(vals, np.ndarray):
                    vals = np.array(vals)

                if how == "interval":
                    # awkward way of checking dtype
                    if re.search('int|float', str(vals.dtype)):
                        w = [[f"{k} >= {vals[vi]}", f"{k} < {vals[vi + 1]}"]
                             for vi in range(len(vals) - 1)]
                    # non numbers deserve a single quote (')
                    else:
                        w = [[f"{k} >= '{vals[vi]}'", f"{k} < '{vals[vi + 1]}'"]
                             for vi in range(len(vals) - 1)]
                else:
                    # awkward way of checking dtype
                    if re.search('int|float', str(vals.dtype)):
                        w = [[f"{k} {how} {v}"] for v in vals]
                    # non numbers deserve a single quote (')
                    else:
                        w = [[f"{k} {how} '{v}'"] for v in vals]

                where_dict[k] = w

                # create a where to increment over
                # - this should be a list of lists
                # - with each (sub) list containing where condition to be evaluated
                where_list = reduce(lambda x, y: [xi + yi for xi in x for yi in y],
                                    [v for k, v in where_dict.items()])
        else:
            where_list = where

        if not isinstance(where_list, list):
            where_list = [where_list]

        return where_list

    @staticmethod
    def get_where_list(global_select, local_select=None, ref_loc=None):
        # store results in list
        out = []

        ref_loc = pandas_to_dict(ref_loc)

        for gs in global_select:
            # check if static where
            is_static = all([c in gs for c in ['col', 'comp', 'val']])
            # if it's a static where condition just add
            if is_static:
                out += [gs]
            # otherwise it's 'dynamic' - i.e. a function local_select and reference location
            else:
                # require local_select and ref_loc are provided
                assert local_select is not None, \
                    f"dynamic where provide: {gs}, however local_select is: {type(local_select)}"
                assert ref_loc is not None, \
                    f"dynamic where provide: {gs}, however ref_loc is: {type(ref_loc)}"
                # check required elements are
                assert all([c in gs for c in ['loc_col', 'src_col', 'func']]), \
                    f"dynamic where had keys: {gs.keys()}, must have: ['loc_col', 'src_col', 'func'] "
                # get the location column
                loc_col = gs['loc_col']
                # require location column is reference
                assert loc_col in ref_loc, f"loc_col: {loc_col} not in ref_loc: {ref_loc}"

                func = gs['func']
                if isinstance(func, str):
                    func = eval(func)
                # increment over the local select - will make a selection
                for ls in local_select:
                    # if the location column matchs the local select
                    if loc_col == ls['col']:
                        # create a 'where' dict using comparison and value from local select
                        _ = {
                            "col": gs['src_col'],
                            "comp": ls['comp'],
                            "val": func(ref_loc[loc_col], ls['val'])
                        }
                        out += [_]

        return out

    @staticmethod
    def get_attribute_from_table(source, table, attribute_name):
        # get attribute from a given table in a HDF5 file
        if isinstance(source, str):
            with pd.HDFStore(source, mode='r') as store:
                try:
                    attr = store.get_storer(table).attrs[attribute_name]
                    # print(json.dumps(raw_data_config, indent=4))
                except Exception as e:
                    print(e)
                    warnings.warn(f"issue getting attribute: {attribute_name}\nfrom table: {table}\nin source:{source}")
                    attr = None
        elif isinstance(source, pd.io.pytables.HDFStore):
            try:
                attr = source.get_storer(table).attrs[attribute_name]
                # print(json.dumps(raw_data_config, indent=4))
            except Exception as e:
                print(e)
                warnings.warn(f"issue getting attribute: {attribute_name}\nfrom table: {table}\nin source:{source}")
                attr = None
        else:
            raise NotImplementedError(f"source type: {type(source)} is not implemented")


        return attr

if __name__ == "__main__":

    import pandas as pd
    from GPSat import get_data_path
    from GPSat.utils import WGS84toEASE2_New, EASE2toWGS84_New

    pd.set_option("display.max_columns", 200)

    # ----
    # add_cols
    # ---

    import pandas as pd
    from GPSat.dataloader import DataLoader


    def add_one(x):
        return x['A'] + 1

    df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
    DataLoader.add_cols(df, col_func_dict={'C': {'func': add_one, "args": df}})

    # ---
    # read flat files
    # ---

    # example config
    # - below "func" are given as string, to allow storing as attribute in hdf5 file
    # - they could be python functions
    read_flat_files_config = {
        "file_dirs": [
            get_data_path("RAW")
        ],
        # read all .csv files - probably to aggressive, dial back
        "file_regex": "\.csv$",
        # add/modify columns
        "col_funcs": {
            # convert 'datetime' column to dtype datetime64
            "datetime": {
                "func": "lambda x: x.astype('datetime64')",
                "col_args": "datetime"
            }
        },
        #
        "row_select": [
            # select datetime after '2019-12-01'
            # - the 'col_numpy': False means the 'datetime' column will stay as pd.Series
            # - allowing for comparison with a str e.g. "2019-12-01"
            {"func": ">=", "col_args": "datetime", "args": "2018-12-01", "col_numpy": False},
            # and before '2020-01-31' - done differently to above
            # - NOTE: YYYY-MM-DD will be converted to YYYY-MM-DD 00:00:00 (i.e. mid-night, start of day)
            {
                "func": "lambda dt, val: dt <= val",
                "col_kwargs": {"dt": "datetime"},
                "kwargs": {"val": "2019-01-31 23:59:59"},
                "col_numpy": False,
            }
        ],
        # select all columns
        "col_select": None,
        # increase verbose level to see more details
        "verbose": 3
    }

    # run info - if __file__ does not exist in environment (i.e. when running interactive)
    try:
        run_info = DataLoader.get_run_info(script_path=__file__)
    except NameError as e:
        run_info = DataLoader.get_run_info()

    # --
    # read flat (csv) files
    # --

    df = DataLoader.read_flat_files(**read_flat_files_config)

    # --
    # write to hdf5
    # --

    print("writing to hdf5 file")
    # store = pd.HDFStore(path=get_data_path("RAW", "example.h5"), mode='w')
    with pd.HDFStore(path=get_data_path("RAW", "example.h5"), mode='w') as store:
        DataLoader.write_to_hdf(df,
                                table="data",
                                append=False,
                                store=store,
                                config=read_flat_files_config,
                                run_info=run_info)

    # --
    # read hdf5
    # --

    print("reading from hdf5 files")
    # read by specifying file path
    df = DataLoader.read_hdf(table="data", path=get_data_path("RAW", "example.h5"))

    # read-only store, close when done)
    store_r = pd.HDFStore(path=get_data_path("RAW", "example.h5"), mode='r')
    df = DataLoader.read_hdf(table="data", store=store_r, close=False)
    print(f"store is open: {store_r.is_open}")

    # provide where arguments
    print("select from hdf using 'where' conditions")
    where = ["lon<=81.0", "datetime>='2018-12-03'", "datetime<='2018-12-25'"]
    print(f"where: {where}")
    df2 = DataLoader.read_hdf(table="data", store=store_r,
                              where=where)
    print(f"store is open: {store_r.is_open}")
    print(f"'lon' max: {df2['lon'].max()}")
    print(f"'datetime' max: {df2['datetime'].max()}")
    print(f"'datetime' min: {df2['datetime'].min()}")

    # --
    # bin data
    # --

    print("bin data - in practice one should review the data and remove 'bad' points first")

    # convert (lon, lat) to (x,y) using EASE2.0 projection
    df['x'], df['y'] = WGS84toEASE2_New(df['lon'], df['lat'])
    # convert datetime to date
    df['date'] = df['datetime'].values.astype('datetime64[D]')

    # add fake 'sat' column - not really needed
    df['sat'] = "S3AB"
    df.loc[df['lat'] > 81, 'sat'] = "CS"

    # bind data using a 50x50km grid
    # - grid_res and x/y_range are in meters
    # - returns a xr.Dataset
    ds_bin = DataLoader.bin_data_by(df=df,
                                    by_cols=['sat', 'date'],
                                    val_col='fb',
                                    grid_res=50 * 1000,
                                    x_range=[-4500000.0, 4500000.0],
                                    y_range=[-4500000.0, 4500000.0])

    # --
    # write Dataset to netCDF file
    # --

    nc_file = get_data_path("binned", "example.nc")
    os.makedirs(os.path.dirname(nc_file), exist_ok=True)
    DataLoader.write_to_netcdf(ds_bin, path=nc_file, mode='w')

    # ----
    # read / select data
    # ----

    # --
    # from netCDF file (using xarray)
    # --
    ds = xr.open_dataset(nc_file)

    # select data using a standard xarray 'where' condition
    d0, d1 = '2018-12-03', '2018-12-25'
    where = (ds.coords['date'] >= np.datetime64(d0)) & \
            (ds.coords['date'] <= np.datetime64(d1))

    df0 = DataLoader.data_select(ds, where, return_df=True)

    # using a list of dict
    where = [
        {"col": "date", "comp": ">=", "val": d0},
        {"col": "date", "comp": "<=", "val": d1}
    ]
    df1 = DataLoader.data_select(ds, where, return_df=True)

    # check are equal
    pd.testing.assert_frame_equal(df0, df1)

    # --
    # from ndf5 file (using pd.HDFStore)
    # --

    # using a standard where condition for store.select
    where = [f'datetime>="{d0}"', f'datetime<="{d1}"']
    ndf5_file = get_data_path("RAW", "example.h5")
    store = pd.HDFStore(ndf5_file)

    df0 = DataLoader.data_select(store, where, table="data")

    # using a list of dict
    where = [
        {"col": "datetime", "comp": ">=", "val": d0},
        {"col": "datetime", "comp": "<=", "val": d1}
    ]

    df1 = DataLoader.data_select(store, where, table="data")

    # check are equal
    pd.testing.assert_frame_equal(df0, df1)

    #  ---
    # select subset - points within some distance of a reference point
    # ---

    coords_col = ['x', 'y', 't']

    nc_file = get_data_path("binned", "example.nc")
    ds = xr.open_dataset(nc_file)

    # select data using a standard xarray 'where' condition
    d0, d1 = '2018-12-03', '2018-12-25'
    where = (ds.coords['date'] >= np.datetime64(d0)) & \
            (ds.coords['date'] <= np.datetime64(d1))

    df = DataLoader.data_select(ds, where, return_df=True)
    df.reset_index(inplace=True)

    # data may require a transformation of columns to get
    # coordinates that would be used in OI i.e. ('lon', 'lat', 'datetime') -> ('x', 'y', 't')
    col_func_dict = {
        "t": {
            "func": "lambda x: x.astype('datetime64[D]').astype(int)",
            "col_args": "date"
        }
    }

    # add new columns in place
    DataLoader.add_cols(df, col_func_dict)

    assert all([c in df for c in coords_col]), f"not all coords_col: {coords_col} are in data: {df.columns}"

    # used for selecting a data for a "local expert"

    # a local expert is defined in relation to some reference locations - store in DataFrame
    ref_locs = pd.DataFrame({"date": ["2018-12-14", "2018-12-15"],
                             "x": df['x'].median(),
                             "y": df['y'].median()})

    # - get lon, lat for demonstration purposes
    # lon, lat = EASE2toWGS84_New(df['x'].median(), df['y'].median())
    # ref_locs['lon'], ref_locs['lat'] = EASE2toWGS84_New(df['x'].median(), df['y'].median())
    convert_ref_loc = {
        "t": {
            "func": "lambda x: x.astype('datetime64[D]').astype(int)",
            "col_args": "date"
        }
    }
    DataLoader.add_cols(ref_locs, convert_ref_loc)

    assert all([c in ref_locs for c in coords_col]), f"not all coords_col: {coords_col} are in data: {ref_locs.columns}"

    # for a given reference location (row from ref_locs)
    # - find the data within some radius
    reference_location = ref_locs.iloc[0, :].to_dict()

    # location selection will be made always relative to a reference location
    # - the correspoding reference location will be add to "val" (for 1-d)
    # - for 2-d coordindates (a combination), KDTree.ball_query will be used
    local_select = [
        {"col": "t", "comp": "<=", "val": 4},
        {"col": "t", "comp": ">=", "val": -4},
        {"col": ["x", "y"], "comp": "<", "val": 300 * 1000}
    ]
    verbose = True

    df_sel = DataLoader.local_data_select(df, reference_location, local_select)
