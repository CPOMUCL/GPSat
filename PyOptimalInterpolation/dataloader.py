import datetime
import os
import re
import sys
import warnings
import pickle

import pandas as pd
import numpy as np
import xarray as xr
import scipy.stats as scst
from scipy.spatial import KDTree

from functools import reduce
from PyOptimalInterpolation.utils import config_func, get_git_information, sparse_true_array, pandas_to_dict
from PyOptimalInterpolation.decorators import timer


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
        # self.dataset =


    @staticmethod
    def add_cols(df, col_func_dict=None, filename=None, verbose=False):
        """
        Add columns to a given DataFrame (df) using elements from a dictionary
        NOTE: DataFrame is manipulated inplace

        Parameters
        ----------
        df: pd.DataFrame to have columns added to
        col_func_dict: dict or None. If dict keys will be the new column name
            other values, along with filename, will be passed into utils.config_func
        filename: str or None. provide to each call of utils.config_func
        verbose: bool or int, default False. Print new column being added if >= 3

        Returns
        -------
        None

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

    @staticmethod
    def row_select_bool(df, row_select=None, verbose=False, **kwargs):

        if row_select is None:
            row_select = [{}]
        elif isinstance(row_select, dict):
            row_select = [row_select]

        assert isinstance(row_select, list), f"expect row_select to be a list (of dict), is type: {type(col_funcs)}"
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
            select &= config_func(df=df, **{**kwargs, **sl})

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
        # this is for reading data stored in dict in pickle files
        # - as a result this method is a a bit rigid
        # pkl_files can either be a dict, str or list of str
        # if pkl_dir is not None it will be pre-appended to all pkl_files

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
        data_vars = {}
        # increment over files
        for name, files in pkl_files.items():
            print("*" * 10)
            print(name)

            if isinstance(files, str):
                files = [files]

            xa_list = []
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

                xa = xr.DataArray(obs, coords={"date": dates}, name=name, dims=dim_names)
                xa_list += [xa]

            # combine the obs
            # - test for this doing the correct thing - see commented below
            data_vars[name] = xr.merge(xa_list, compat="override")[name]

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

        ds = xr.Dataset(data_vars)

        return ds

    @staticmethod
    def read_from_npy(npy_files, npy_dir, dims=None, flatten_xy=True):

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

            coord_arrays[name] = xr.DataArray(coord_arrays[name],
                                              dims=dims,
                                              name=name)

        if ('x' in npy_files) & ('y' in npy_files) & flatten_xy:
            coord_arrays['x'] = coord_arrays['x'].isel(y=0)
            coord_arrays['y'] = coord_arrays['y'].isel(x=0)

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
                    close=True,
                    **kwargs):

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
                # TODO: should reset_index be default?
                out = out.to_dataframe().dropna()

            if reset_index:
                out.reset_index(inplace=True)

        # pd.HDFStore
        elif isinstance(obj, pd.io.pytables.HDFStore):
            # TODO: determine if it is always the case
            assert table is not None, "\n\nobj is HDFStore, however table is None, needs to be provided\n\n"

            if is_list_of_dict:
                where = [cls._hdfstore_where_from_dict(wd) for wd in where]
            out = obj.select(key=table, where=where, columns=columns, **kwargs)

            if reset_index:
                out.reset_index(inplace=True)

            # close the HDFStore object?
            if close:
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
             verbose=False):

        # given a source: DataFrame,Series,Dataset,HDFStore or str
        # - read in data (possible using where), add columns, select subset of rows and columns

        # if the source is a string - process to get valid source: DataFrame, DataSet, HDFStore
        if isinstance(source, str):
            if source_kwargs is None:
                source_kwargs = {}
            source = cls._get_source_from_str(source, engine=engine, **source_kwargs)

        # --
        # load data
        # --

        # TODO: review some of these hardcoded defaults below - should they be options?
        df = cls.data_select(obj=source,
                             where=where,
                             table=table,
                             return_df=True,
                             reset_index=False,
                             drop=True,
                             copy=True,
                             close=True,
                             columns=None)

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
        df = df.loc[select, :]#.copy(True)

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

        return tmp_fun(obj.coords[col], val)

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

            return tmp_fun(obj[col], val)

        # otherwise  use config_func
        else:
            out = config_func(df=obj, **wd)
            if str(out.dtype) != 'bool':
                warnings.warn("not returning an array with dtype bool")
            return out

    @classmethod
    def download_data(cls, id_files=None, id=None, file=None, unzip=False):
        # wrapper for downloading data from good drive using
        pass

    @staticmethod
    def get_run_info(script_path=None):
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

        Parameters
        ----------
        df
        x_range
        y_range
        grid_res
        x_col
        y_col
        val_col
        bin_statistic
        return_bin_center

        Returns
        -------

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
                    assert c in df
                    assert c in reference_location
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
    @timer
    def store_to_hdf_table_w_multiindex(idx_dict, out_path, **kwargs):

        raise NotImplementedError
        # store (append) data in table (key) matching the data name (k)
        # with pd.HDFStore(out_path, mode='a') as store:
        #     store.append(key=k, value=df, data_columns=True)
        pass

    @staticmethod
    def mindex_df_to_mindex_dataarray(df, data_name,
                                      dim_cols=None,
                                      infer_dim_cols=True,
                                      index_name="index"):

        # NOTE: df is manipulated by reference - provide copy if need be
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
        # TODO: review / refactor get_where_list_legacy
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


if __name__ == "__main__":

    import pandas as pd
    from PyOptimalInterpolation import get_data_path
    from PyOptimalInterpolation.utils import WGS84toEASE2_New, EASE2toWGS84_New

    pd.set_option("display.max_columns", 200)

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
