import gc
import os
import re
import warnings
import time
import datetime
import pprint

import numpy as np
import pandas as pd
import xarray as xr
from abc import ABC, abstractmethod
from typing import List, Dict, Tuple, Union, Type
from dataclasses import dataclass

import cartopy.crs as ccrs
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from PyOptimalInterpolation.plot_utils import plot_pcolormesh, plot_hist

from PyOptimalInterpolation.decorators import timer
from PyOptimalInterpolation.dataloader import DataLoader
import PyOptimalInterpolation.models as models
from PyOptimalInterpolation.prediction_locations import PredictionLocations
from PyOptimalInterpolation.utils import json_serializable, check_prev_oi_config, get_previous_oi_config, config_func, \
    dict_of_array_to_dict_of_dataframe, pandas_to_dict, to_array


@dataclass
class LocalExpertData:
    # class attributes
    # TODO: fix the type hints below - list of what, other types things can be
    obs_col: Union[str, None] = None
    coords_col: Union[list, None] = None
    global_select: Union[list, None] = None
    local_select: Union[list, None] = None
    col_funcs:  Union[list, None] = None
    table:  Union[str, None] = None
    data_source: Union[str, None] = None
    engine:  Union[str, None] = None
    read_kwargs: Union[dict, None] = None

    file_suffix_engine_map = {
        "csv": "read_csv",
        "tsv": "read_csv",
        "h5": "HDFStore",
        "zarr": "zarr",
        "nc": "netcdf4"
    }

    def set_data_source(self, verbose=False):

        # TODO: replace parts of below with DataLoader._get_source_from_str
        data_source = self.data_source
        engine = self.engine
        kwargs = self.read_kwargs

        if kwargs is None:
            kwargs = {}
        assert isinstance(kwargs, dict), f"expected additional read_kwargs to be dict (or None), got: {type(kwargs)}"


        # TODO: allow engine to not be case sensitive
        # TODO: allow for files to be handled by DataLoader.read_flat_files()
        #  - i.e. let file be a dict to be unpacked into read_flat_files, set engine = "read_flat_files"
        # TODO: add verbose statements

        # read in or connect to data

        # if engine is None then infer from file name
        if (engine is None) & isinstance(data_source, str):
            # from the beginning (^) match any character (.) zero
            # or more times (*) until last (. - require escape with \)
            file_suffix = re.sub("^.*\.", "", data_source)

            assert file_suffix in self.file_suffix_engine_map, \
                f"file_suffix: {file_suffix} not in file_suffix_engine_map: {self.file_suffix_engine_map}"

            engine = self.file_suffix_engine_map[file_suffix]

            if verbose:
                print(f"engine not provide, inferred '{engine}' from file suffix '{file_suffix}'")

        # connect / read in data

        # available pandas read method
        pandas_read_methods = [i for i in dir(pd) if re.search("^read", i)]
        # xr.open_dataset engines
        xr_dataset_engine = ["netcdf4", "scipy", "pydap", "h5netcdf", "pynio", "cfgrib", \
                             "pseudonetcdf", "zarr"]

        # self._data_file = data_source
        self.engine = engine

        # self.data_source = None
        # read in via pandas
        if engine in pandas_read_methods:
            self.data_source = getattr(pd, engine)(data_source, **kwargs)
        # xarray open_dataset
        elif engine in xr_dataset_engine:
            self.data_source = xr.open_dataset(data_source, engine=engine, **kwargs)
        # or hdfstore
        elif engine == "HDFStore":
            self.data_source = pd.HDFStore(data_source, mode="r", **kwargs)
        else:
            warnings.warn(f"file: {data_source} was not read in as\n"
                          f"engine: {engine}\n was not understood. "
                          f"data_source has not been changed")
            self.engine = None


# TODO: change print statements to use logging
class LocalExpertOI:


    # when reading in data
    file_suffix_engine_map = {
        "csv": "read_csv",
        "tsv": "read_csv",
        "h5": "HDFStore",
        "zarr": "zarr",
        "nc": "netcdf4"
    }


    def __init__(self,
                 expert_loc_config: Union[Dict, None]=None,
                 data_config: Union[Dict, None]=None,
                 model_config: Union[Dict, None]=None,
                 pred_loc_config: Union[Dict, None]=None):

        # TODO: make locations, data, model attributes with arbitrary structures
        #  maybe just dicts with their relevant attributes stored within

        self.constraints = None
        self.model_init_params = None
        self.model_load_params = None
        self.model = None
        self.data_table = None

        # data will be set as LocalExpertData instance
        self.data = None
        # config will be used to store the parameters used to set: locations, data, model
        self.config = {}

        # ------
        # Local Expert Locations
        # ------

        locations = self._none_to_dict_check(expert_loc_config)

        self.set_expert_locations(**locations)

        # ------
        # Data (source)
        # ------

        data = self._none_to_dict_check(data_config)

        self.set_data(**data)

        # ------
        # Model
        # ------

        model = self._none_to_dict_check(model_config)
        
        self.set_model(**model)

        # ------
        # Prediction Locations
        # ------

        pred_loc = self._none_to_dict_check(pred_loc_config)

        self.set_pred_loc(**pred_loc)

    def _none_to_dict_check(self, x):
        if x is None:
            x = {}
        assert isinstance(x, dict)
        return x
        
    def _method_inputs_to_config(self, locs, code_obj):
        # TODO: validate this method returns expected values
        # code_obj: e.g. self.<method>.__code__
        # locs: locals
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
                    print(f"KeyError on var: {var}\n", e, "skipping")
        return json_serializable(config)

    def set_pred_loc(self, **kwargs):

        self.config["pred_loc"] = self._method_inputs_to_config(locals(), self.set_pred_loc.__code__)

        # TODO: set ploc as PredictionLocation object, initialised with kwargs
        # - what happens if kwargs is empty?
        self.pred_loc = PredictionLocations(**kwargs)

        # TODO: if check data exists, get coords_col from there
        if isinstance(self.data, LocalExpertData):
            self.pred_loc.coords_col = self.data.coords_col


    def set_data(self,
                 **kwargs
                 ):

        # --
        # store parameters to config
        # --

        # TODO: non JSON serializable objects may cause issues if trying to re-run with later
        # TODO: wrap this into private method, use self.*.__code__, locs as input
        self.config["data"] = self._method_inputs_to_config(locals(), self.set_data.__code__)

        # ---
        # initialise data attribute with key words arguments provided
        # ---

        self.data = LocalExpertData(**kwargs)

        # if data_source was provided - then properly set values (connect to xr.dataset / HDFStore / read_csv)
        if self.data.data_source is not None:
            # if data_source is str try to set to DataFrame, xr.Dataset or HDFStore
            if isinstance(self.data.data_source, str):
                self.data.set_data_source()

            # TODO: check data_source is valid type - do that here (?)

    def set_model(self,
                  oi_model=None,
                  init_params=None,
                  constraints=None,
                  load_params=None,
                  replacement_threshold=None,
                  replacement_model=None,
                  replacement_init_params=None,
                  replacement_constraints=None):

        # TODO: non JSON serializable objects may cause issues if trying to re-run with later
        self.config["model"] = self._method_inputs_to_config(locals(), self.set_model.__code__)

        # oi_model is a str then expect to be able to import from models
        # TODO: perhaps would like to generalise this a bit more - read models from different modules
        self.model = oi_model

        # oi_model is a str then expect to be able to import from models
        # TODO: perhaps would like to generalise this a bit more - read models from different modules
        if isinstance(self.model, str):
            self.model = getattr(models, self.model)

        # TODO: should these only be set if they are not None?
        self.model_init_params = init_params
        self.constraints = constraints
        self.model_load_params = load_params

        # Replacement model (used to substitute the main model if number of training points is < replacement_threshold)
        if replacement_threshold is not None:
            self.replacement_threshold = replacement_threshold
            self.replacement_model = self.model if replacement_model is None else getattr(models, replacement_model)
            self.replacement_init_params = init_params if replacement_init_params is None else replacement_init_params
            self.replacement_constraints = constraints if replacement_constraints is None else replacement_constraints

    def set_expert_locations(self,
                             df=None,
                             file=None,
                             loc_dims=None,
                             # masks=None,
                             # ref_data=None,
                             add_data_to_col=None,
                             col_funcs=None,
                             keep_cols=None,
                             row_select=None,
                             sort_by=None,
                             verbose=False,
                             **kwargs):

        # TODO: if verbose print what the input parameters are?
        # TODO: allow for dynamically created local expert locations
        #  - e.g. provide grid spacing, mask types (spacing, over ocean only)

        # --
        # store parameters to config
        # --
        # TODO: none JSON serializable objects may cause issues if trying to re-run with later
        self.config["locations"] = self._method_inputs_to_config(locals(), self.set_expert_locations.__code__)

        # if DataFrame provide - set directly
        if df is not None:
            assert isinstance(df, pd.DataFrame), f"df provided, expected to DataFrame, got: {type(df)}"
            if verbose:
                print(f"local_expert_locations - df:\n{df.head(4)}\nprovided")

            # apply any modifications before setting
            self.expert_locs = self._modify_dataframe(df,
                                                      add_data_to_col=add_data_to_col,
                                                      col_funcs=col_funcs,
                                                      keep_cols=keep_cols,
                                                      row_select=row_select,
                                                      verbose=verbose)

        elif file is not None:
            if verbose:
                print(f"local_expert_locations - file:\n{file}\nprovided")
            locs = self._read_local_expert_locations_from_file(loc_file=file,
                                                               add_data_to_col=add_data_to_col,
                                                               col_funcs=col_funcs,
                                                               keep_cols=keep_cols,
                                                               row_select=row_select,
                                                               verbose=verbose,
                                                               **kwargs)

            if sort_by:
                locs.sort_values(sort_by, inplace=True)

            self.expert_locs = locs
        elif loc_dims is not None:
            warnings.warn("loc_dims provided to local_expert_locations but is not handled, "
                          "'expert_locs' attribute will be unchanged")
            # # dimensions for the local expert
            # # - more (columns) can be added with col_func_dict
            #
            # # expert location masks
            # # TODO: needs work
            # if masks is None:
            #     masks = None
            # el_masks = expert_locations.get("masks", [])
            # TODO: move get_masks_for_expert_loc into LocalExpertOI
            # masks = DataLoader.get_masks_for_expert_loc(ref_data=ds, el_masks=el_masks, obs_col=obs_col)
            #
            # # get the local expert locations
            # # - this will be a DataFrame which will be used to create a multi-index
            # # - for each expert values will be stored to an hdf5 using an element (row) from above multi-index
            # TODO: this method should be moved into this class
            # xprt_locs = DataLoader.generate_local_expert_locations(loc_dims,
            #                                                        ref_data=ref_data,
            #                                                        masks=masks,
            #                                                        row_select=row_select,
            #                                                        col_func_dict=col_funcs,
            #                                                        keep_cols=keep_cols,
            #                                                        sort_by=sort_by)

        else:
            warnings.warn("inputs to local_expert_locations not handled, "
                          "'expert_locs' attribute will be unchanged")

    def _modify_dataframe(self,
                          df,
                          add_data_to_col=None,
                          row_select=None,
                          col_funcs=None,
                          sort_by=None,
                          keep_cols=None,
                          copy=False,
                          verbose=False):
        """
        Modify DataFrame: add data to columns (repeatedly), select subset of rows,
        modify / add columns with col_funcs, sort_by, keep select columns (keep_cols)

        Parameters
        ----------
        df
        add_data_to_col
        row_select
        col_funcs
        sort_by
        keep_cols
        verbose

        Returns
        -------

        """
        # make a copy?
        if copy:
            df = df.copy(True)

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

        # apply column function - to add new columns
        DataLoader.add_cols(df, col_funcs)

        # (additional) select rows
        if row_select is not None:
            df = DataLoader.data_select(df, where=row_select)

        # store rows - e.g. by date?
        if sort_by is not None:
            df.sort_values(by=sort_by, inplace=True)

        # select a subset of columns
        if keep_cols is not None:
            df = df.loc[:, keep_cols]

        return df

    def _read_local_expert_locations_from_file(self,
                                               loc_file,
                                               add_data_to_col=None,
                                               row_select=None,
                                               col_funcs=None,
                                               sort_by=None,
                                               keep_cols=None,
                                               verbose=False,
                                               **read_csv_kwargs):
        # TODO: add doc string to _read_local_expert_locations_from_file
        assert os.path.exists(loc_file), f"loc_file:\n{loc_file}\ndoes not exist"
        if verbose:
            print(f"reading in (expert) locations from:\n{loc_file}")
        locs = pd.read_csv(loc_file, **read_csv_kwargs)

        if verbose:
            print(f"number of rows in location DataFrame: {len(locs)}")

        locs = self._modify_dataframe(locs,
                                      add_data_to_col=add_data_to_col,
                                      row_select=row_select,
                                      col_funcs=col_funcs,
                                      sort_by=sort_by,
                                      keep_cols=keep_cols,
                                      copy=False,
                                      verbose=verbose)

        return locs

    @timer
    def _update_global_data(self,
                            df=None,
                            global_select=None,
                            local_select=None,
                            ref_loc=None,
                            col_funcs=None,
                            prev_where=None):

        if global_select is None:
            global_select = []

        # get current where list
        where = DataLoader.get_where_list(global_select,
                                          local_select=local_select,
                                          ref_loc=ref_loc)

        # fetch new data?
        if prev_where is None:
            fetch = True
        elif isinstance(prev_where, list):
            # fetch new data
            try:
                # if not same length
                if len(prev_where) != len(where):
                    fetch = True
                else:
                    # NOTE: this does not handle same where dicts but in different order
                    fetch = not all([w == prev_where[i]
                                     for i, w in enumerate(where)])
            except IndexError as e:
                print(e)
                fetch = True
        else:
            print("prev_where was not understood, will fetch new data")
            fetch = True

        if fetch:
            # extract 'global' data
            df = DataLoader.data_select(obj=self.data.data_source,
                                        table=self.data.table,
                                        where=where,
                                        return_df=True,
                                        reset_index=True)

            # add additional columns to data - as needed
            DataLoader.add_cols(df, col_func_dict=col_funcs)

        return df, where


    @staticmethod
    def _remove_previously_run_locations(store_path, xprt_locs, table="run_details"):
        # read existing / previous results
        try:
            with pd.HDFStore(store_path, mode='r') as store:
                # get index from previous results
                # - the multi index represent the expert location
                prev_res = store.select(table, columns=[]).reset_index()
                # left join to find which have not be found (left_only)
                tmp = xprt_locs.merge(prev_res,
                                      how='left',
                                      on=prev_res.columns.values.tolist(),
                                      indicator='found_already')
                # create bool array of those to keep
                keep_bool = tmp['found_already'] == 'left_only'
                print(f"using: {keep_bool.sum()} / {len(keep_bool)} reference locations - some were already found")
                xprt_locs = xprt_locs.loc[keep_bool.values].copy(True)

        except OSError as e:
            print(e)
        except KeyError as e:
            print(e)

        return xprt_locs

    @staticmethod
    def _append_to_store_dict_or_write_to_table(save_dict, store_path,
                                                store_dict=None,
                                                store_every=1):
        if store_dict is None:
            store_dict = {}

        assert isinstance(save_dict, dict), f"save_dict must be dict got: {type(save_dict)}"

        # use reference location to change index of tables in save_dict to a multi-index
        # TODO: determine if only want to use coord_col for multi index - to keep things cleaner(?)
        #  - i.e. use: idx_dict = ref_loc[self.coords_col]
        # save_dict = DataLoader.make_multiindex_df(idx_dict=ref_loc, **save_dict)

        # if store dict is empty - populate with list of multi-index dataframes
        if len(store_dict) == 0:
            store_dict = {k: [v] for k, v in save_dict.items()}
        # otherwise add
        else:
            for k, v in save_dict.items():
                if k in store_dict:
                    store_dict[k] += [v]
                # for non 'run_details' maybe missing
                else:
                    store_dict[k] = [v]

        num_store = max([len(v) for k, v in store_dict.items()])

        if num_store >= store_every:
            print("SAVING RESULTS")
            for k, v in store_dict.items():
                print(k)
                df_tmp = pd.concat(v, axis=0)
                try:
                    with pd.HDFStore(store_path, mode='a') as store:
                        # store.append(key=k, value=df_tmp, data_columns=True)
                        store.append(key=k, value=df_tmp)
                except ValueError as e:
                    print(e)
                except Exception as e:
                    print(e)
            store_dict = {}

        return store_dict

    @timer
    def load_params(self,
                    model,
                    previous=None,
                    previous_params=None,
                    file=None,
                    param_names=None,
                    ref_loc=None,
                    index_adjust=None,
                    **param_dict):

        # TODO: add verbose print / log lines here
        # method to load (set) parameters - either from (h5) file, or specified directly
        # via param_dict

        # if file is None - provide param_dict
        if file is not None:
            # TODO: apply adjustment to location
            if index_adjust is None:
                index_adjust = {}
            # ensure reference location is expressed as a dict
            ref_loc = pandas_to_dict(ref_loc)
            # make a copy - as can change values
            rl = ref_loc.copy()

            # TODO: is this how the (expert/reference) locations should be adjusted?
            #  - this implementation won't allow for 'args' to be specified v
            for k, v in index_adjust.items():
                rl[k] = config_func(**v, args=rl[k])

            # TODO: implement fetching of parameters - from file
            param_dict = self._read_params_from_file(file=file,
                                                     model=model,
                                                     ref_loc=rl,
                                                     param_names=param_names)
        # load previous params?
        elif previous is not None:
            param_dict = previous_params
            if param_dict is None:
                param_dict = {}

        model.set_parameters(**param_dict)

    @timer
    def _read_params_from_file(self,
                               model,
                               file,
                               ref_loc,
                               param_names=None) -> dict:
        """
        for a given reference location and (h5) file, select the entry corresponding to
        the reference location and extract values.
        returns a dict of numpy arrays to be used by model.set_parameters()
        """

        # TODO: use a verbose level (should be set as attribute when initialised?)
        assert isinstance(ref_loc, (pd.Series, pd.DataFrame, dict)), f"ref_loc expected to pd.Series or dict, got: {type(ref_loc)}"

        ref_loc = pandas_to_dict(ref_loc)

        if not os.path.exists(file):
            warnings.warn(f"in '_read_params_from_file' provide file:\n{file}\ndoes not exist, returning empty dict")
            return {}

        # from the reference location create a (list of) where statements
        rl_where = [f"{k} == {str(v)}"
                    if not isinstance(v, datetime.date) else
                    f"{k} == '{str(v)}'"
                    for k, v in ref_loc.items()
                    if k in model.coords_col]

        # which param_names to get?
        # - if not specified get all
        if param_names is None:
            param_names = model.param_names

        # check provided param_names are values
        for pn in param_names:
            assert pn in model.param_names, f"provide param name:{pn}\nis not in param_names:{self.model.param_names}"

        # results
        out = {}
        # from the file read from each table in param_names
        # - selecting values aligned to reference table
        #
        with pd.HDFStore(file, mode='r') as store:
            for k in param_names:

                try:
                    # TODO: cases where there are double entries (entered by mistake) should be handled / caught here
                    #  - there should be some sort of expected value, or dimension check /
                    #  - size of the result from store.select(k, where=rl_where) should be validated
                    # TODO: check this works for arbitrary n-dim data
                    tmp_df = store.select(k, where=rl_where)
                    if len(tmp_df) == 0:
                        continue
                    tmp = DataLoader.mindex_df_to_mindex_dataarray(df=tmp_df,
                                                                   data_name=k,
                                                                   infer_dim_cols=True)
                    out[k] = tmp.values[0]

                    # nan check - should this be done else where
                    if isinstance(out[k], np.ndarray):
                        if any(np.isnan(out[k])):
                            warnings.warn(f"\n{k}: found some nans for ref location: {ref_loc}, removing those parameters")
                            out.pop(k)
                    elif isinstance(out[k], float):
                        if np.isnan(out[k]):
                            warnings.warn(f"\n{k}: found some nans for ref location: {ref_loc}, removing those parameters")
                            out.pop(k)

                except KeyError as e:
                    print("KeyError\n", e, f"\nskipping param_name: {k}")
                except Exception as e:
                    print("when reading in parameters some exception occurred\n",
                          type(e),
                          e,
                          f"\nskipping param_name: {k}")

        return out

    @staticmethod
    def dict_of_array_to_table(x, ref_loc=None, concat=False, table=None, default_dim=1):
        """given a dictionary of numpy arrays create DataFrame(s) with ref_loc as the multi index"""

        if concat:
            assert table is not None, "concat is True but (replacement) table (name) not provided"

        # create DataFrame from ndarrays
        dfs = dict_of_array_to_dict_of_dataframe(x,
                                                 concat=concat,
                                                 reset_index=True)

        # replace the index with the reference location - if provided
        if ref_loc is not None:

            # get the components need to create multi index
            # - which will be of variable length (equal to DataFrame length)
            # - but contain the same values
            # can this be done more cleanly?
            ref_loc = pandas_to_dict(ref_loc)
            assert isinstance(ref_loc, dict), f"ref_loc expected to be dict (or Series), got: {type(ref_loc)}"

            midx_tuple = tuple([v for v in ref_loc.values()])
            midx_names = [k for k in ref_loc.keys()]

            for k in dfs.keys():
                # create a multi index of length equal to DataFrame
                df = dfs[k]
                midx = pd.MultiIndex.from_tuples([midx_tuple] * len(df),
                                                 names=midx_names)
                df.index = midx
                dfs[k] = df

        # if the data was concat-ed the keys will represent the dimension of the input data
        # replace these with table name.
        # - If there are multiple dimension the one matching default_dim will be given name table
        # - the others will have the dimensions added to the name

        if not concat:
            out = dfs
        else:
            out = {}
            for k,v in dfs.items():
                if k == default_dim:
                    out[table] = v
                else:
                    out[f"{table}_{k}"] = v

        return out

    @timer
    def run(self,
            store_path,
            store_every=10,
            check_config_compatible=True,
            skip_valid_checks_on=None,
            min_obs=3):

        # ---
        # checks on attributes and inputs
        # ---

        # expert locations
        assert isinstance(self.expert_locs, pd.DataFrame), \
            f"attr expert_locs is {type(self.expert_locs)}, expected to be DataFrame"

        # data source
        assert self.data.data_source is not None, "'data_source' is None"
        assert isinstance(self.data.data_source, (pd.DataFrame, xr.Dataset, xr.DataArray, pd.HDFStore)), \
            f"'data_source' expected to be " \
            f"(pd.DataFrame, xr.Dataset, xr.DataArray, pd.HDFStore), " \
            f"got: {type(self.data.data_source)}"

        # model
        assert self.model is not None, "'model' is None"

        # check model type
        # TODO: determine why model isinstance check is not working as expected
        # assert isinstance(self.model, BaseGPRModel), \
        #     f"'model' expected to be an (inherited) instance of" \
        #     f" BaseGPRModel, got: {type(self.model)}"

        # store path
        assert isinstance(store_path, str), f"store_path expected to be str, got: {type(str)}"

        # create directory for store_path if it does not exist
        os.makedirs(os.path.dirname(store_path), exist_ok=True)

        # check configuration is compatible with previously used, if applicable
        if check_config_compatible:
            # TODO: review checking of previous configs
            prev_oi_config, skip_valid_checks_on = get_previous_oi_config(store_path,
                                                                          oi_config=self.config,
                                                                          skip_valid_checks_on=skip_valid_checks_on)

            # check previous oi_config matches current - want / need them to be consistent (up to a point)
            check_prev_oi_config(prev_oi_config,
                                 oi_config=self.config,
                                 skip_valid_checks_on=skip_valid_checks_on)

        # ----

        # remove previously found local expert locations
        # - determined by (multi-index of) 'run_details' table
        xprt_locs = self._remove_previously_run_locations(store_path,
                                                          xprt_locs=self.expert_locs.copy(True),
                                                          table="run_details")

        # create a dictionary to store result (DataFrame / tables)
        store_dict = {}
        prev_params = {}
        count = 0
        df, prev_where = None, None
        # for idx, rl in xprt_locs.iterrows():
        for idx in range(len(xprt_locs)):

            # TODO: use log_lines
            print("-" * 30)
            count += 1
            print(f"{count} / {len(xprt_locs)}")

            # select the given expert location
            rl = xprt_locs.iloc[[idx], :]
            print(rl)

            # start timer
            t0 = time.time()

            # ----------------------------
            # (update) global data - from data_source (if need be)
            # ----------------------------

            df, prev_where = self._update_global_data(df=df,
                                                      global_select=self.data.global_select,
                                                      local_select=self.data.local_select,
                                                      ref_loc=rl,
                                                      prev_where=prev_where,
                                                      col_funcs=self.data.col_funcs)

            # ----------------------------
            # select local data - relative to expert's location - from global data
            # ----------------------------
            
            df_local = DataLoader.local_data_select(df,
                                                    reference_location=rl,
                                                    local_select=self.data.local_select,
                                                    verbose=False)
            print(f"number obs: {len(df_local)}")

            # if there are too few observations store to 'run_details' (so can skip later) and continue
            if len(df_local) < min_obs:
                # HACK: for testing only - want to set the min obs to be high - then revisit
                # continue
                run_details = {
                    "num_obs": len(df_local),
                    "run_time": np.nan,
                    "mll": np.nan,
                    "optimise_success": False
                }
                save_dict = self.dict_of_array_to_table(run_details,
                                                        ref_loc=rl[self.data.coords_col],
                                                        concat=True,
                                                        table="run_details")

                store_dict = self._append_to_store_dict_or_write_to_table(save_dict=save_dict,
                                                                          store_dict=store_dict,
                                                                          store_path=store_path,
                                                                          store_every=store_every)

                continue

            # -----
            # build model - provide with data
            # -----

            # initialise model
            # TODO: needed to review the unpacking of model_params, when won't it work?
            if hasattr(self, "replacement_threshold"):
                if len(df_local) < self.replacement_threshold:
                    # Set to replacement GPR model if number of data points is low enough
                    print("Setting model to replacement GPR...")
                    _model = self.replacement_model
                    _init_params = self.replacement_init_params
                    _constraints = self.replacement_constraints
                else:
                    _model = self.model
                    _init_params = self.model_init_params
                    _constraints = self.constraints
            else:
                _model = self.model
                _init_params = self.model_init_params
                _constraints = self.constraints

            model = _model(data=df_local,
                           obs_col=self.data.obs_col,
                           coords_col=self.data.coords_col,
                           expert_loc=rl[self.data.coords_col].to_numpy().squeeze(), # Needed for VFF / ASVGP
                           **_init_params)

            # ----
            # load parameters (optional)
            # ----

            # if there are no previous parameters - get the default ones
            if len(prev_params) == 0:
                prev_params = model.get_parameters()

            # TODO: implement this - let them either be previous values, fixed or read from file
            # TODO: review different ways parameters can be loaded: - from file, fixed values,
            #   previously found (optimise success =True)
            if self.model_load_params is not None:

                # HACK: for loading previously found optimal parameters
                # TODO: allow for only a subset of these to be set - e.g. skip variational parameters
                if self.model_load_params.get("previous", False):
                    print("will load previously found params:")
                    pprint.pprint(prev_params, width=1)
                    # print(prev_params)
                    self.model_load_params["previous_params"] = prev_params

                self.load_params(ref_loc=rl,
                                 model=model,
                                 **self.model_load_params)

            # --
            # apply constraints
            # --

            # TODO: generalise this to apply any constraints - use apply_param_transform (may require more checks)
            #  - may need information from config, i.e. obj = model.kernel, specify the bijector, other parameters

            if _constraints is not None:
                if isinstance(_constraints, dict):
                    print("applying lengthscales contraints")
                    # low = self.constraints['lengthscales'].get("low", np.zeros(len(self.data.coords_col)))
                    # high = self.constraints['lengthscales'].get("high", None)
                    # model.set_lengthscale_constraints(low=low, high=high, move_within_tol=True, tol=1e-8, scale=True)
                    if self.model_init_params['coords_scale'] is not None:
                        _constraints["lengthscales"]["scale"] = True
                    model.set_parameter_constraints(_constraints, move_within_tol=True, tol=1e-8)
                else:
                    warnings.warn(f"constraints: {_constraints} are not currently handled!")
            # --
            # optimise parameters
            # --
            # TODO: optimise should be optional
            opt_dets = model.optimise_parameters()

            # get the hyper parameters - for storing
            hypes = model.get_parameters()

            # TODO: remove this
            print(hypes)

            # --
            # prediction location(s)
            # --

            # TODO: making predictions should be optional
            # TODO: allow for pred_loc to return empty array / None (skip predictions)

            # update the expert location for the PredictionLocation attribute
            self.pred_loc.expert_loc = rl
            # generate the expert locations
            prediction_coords = self.pred_loc()

            # --
            # make prediction
            # --

            # TODO: here allow for additional arguments to be supplied to predict e.g. full_cov
            # pred = gpr_model.predict(coords=prediction_coords[self.data.coords_col].values)
            pred = model.predict(coords=prediction_coords)

            # add prediction coordinate location
            for ci, c in enumerate(self.data.coords_col):
                # TODO: review if want to force coordinates to be float
                pred[f'pred_loc_{c}'] = prediction_coords[:, ci]
            # ----
            # store results in tables (keys) in hdf file
            # ----

            t1 = time.time()
            run_time = t1 - t0

            # delete model to try to handle Out of Memory issue?
            del model
            gc.collect()

            # device_name = gpr_model.cpu_name if gpr_model.gpu_name is None else gpr_model.gpu_name

            # run details / info - for reference
            run_details = {
                "num_obs": len(df_local),
                "run_time": run_time,
                "mll": opt_dets['marginal_loglikelihood'],
                "optimise_success": opt_dets['optimise_success'],
                # "device": device_name,
            }

            # if optimisation was successful then store previous parameters
            if run_details['optimise_success']:
                # if any([np.any(np.isnan(v)) for v in hypes.values()]):
                #     print("found nan in hyper parameters - after optimise_success = True, not updating previous params")
                # else:
                for k, v in hypes.items():
                    if np.any(np.isnan(v)):
                        print(f"{k} had nans, not updating")
                    else:
                        rho = 0.95
                        prev_params[k] = rho * prev_params[k] + (1-rho) * hypes[k]

            # ---
            # convert dict of arrays to tables for saving
            # ---

            # TODO: determine if multi index should only have coord_cols - or include extras
            # TODO: could just take rl = rl[self.data.coords_col] at the top of for loop, if other coordinates aren't used
            #  - in which case probably would want to write 'other coordinates' e.g. date, lon, lat to a separate table
            pred = self.dict_of_array_to_table(pred,
                                               ref_loc=rl[self.data.coords_col],
                                               concat=True,
                                               table='preds')

            run_details = self.dict_of_array_to_table(run_details,
                                                      ref_loc=rl[self.data.coords_col],
                                                      concat=True,
                                                      table="run_details")
            hypes = self.dict_of_array_to_table(hypes,
                                                ref_loc=rl[self.data.coords_col],
                                                concat=False)

            save_dict = {
                **run_details,
                **pred,
                **hypes,
                # include a coordinates table - which can have additional coordinate information
                # "coordinates": prediction_coords.set_index(self.data.coords_col)
            }

            # ---
            # 'store' results
            # ---

            # change index to multi index (using ref_loc)
            # - add to table in store_dict or append to table in store_path if above store_every
            store_dict = self._append_to_store_dict_or_write_to_table(save_dict=save_dict,
                                                                      store_dict=store_dict,
                                                                      store_path=store_path,
                                                                      store_every=store_every)

            t2 = time.time()
            print(f"total run time : {t2 - t0:.2f} seconds")

        # ---
        # store any remaining data
        # ---

        if len(store_dict):
            print("storing any remaining tables")
            self._append_to_store_dict_or_write_to_table(save_dict={},
                                                         store_dict=store_dict,
                                                         store_path=store_path,
                                                         store_every=1)

    def plot_locations_and_obs(self,
                               image_file,
                               obs_col=None,
                               lat_col='lat',
                               lon_col='lon',
                               exprt_lon_col='lon',
                               exprt_lat_col='lat',
                               sort_by='date',
                               col_funcs=None,
                               xrpt_loc_col_funcs=None,
                               vmin=None,
                               vmax=None,
                               s=0.5,
                               s_exprt_loc=250,
                               cbar_label="Input Observations",
                               cmap='YlGnBu_r',
                               figsize=(15, 15),
                               projection=None,
                               extent=None):


        # repeating steps used in run to increment over expert locations
        # - plot observations whenever global data changes
        # - plot the local expert location, with color being the number of observations
        # - optionally plot inclusion radius

        # ---
        # checks on attributes and inputs
        # ---

        # expert locations
        assert isinstance(self.expert_locs, pd.DataFrame), \
            f"attr expert_locs is {type(self.expert_locs)}, expected to be DataFrame"

        # data source
        assert self.data.data_source is not None, "'data_source' is None"
        assert isinstance(self.data.data_source, (pd.DataFrame, xr.Dataset, xr.DataArray, pd.HDFStore)), \
            f"'data_source' expected to be " \
            f"(pd.DataFrame, xr.Dataset, xr.DataArray, pd.HDFStore), " \
            f"got: {type(self.data.data_source)}"

        if obs_col is None:
            obs_col = self.data.obs_col

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
        else:
            # TODO: here should check the projectio is of the correct instance
            pass

        # copy the expert locations
        xprt_locs = self.expert_locs.copy(True)

        # (optionally) Add columns expert location
        DataLoader.add_cols(xprt_locs, col_func_dict=xrpt_loc_col_funcs)

        # create a dictionary to store result (DataFrame / tables)
        # store_dict = {}
        # prev_params = {}
        count = 0
        df, prev_where = None, None
        # for idx, rl in xprt_locs.iterrows():

        if isinstance(sort_by, str):
            sort_by = [sort_by]
        xprt_locs.sort_values(sort_by, inplace=True)

        # HERE: start PdfPages
        with PdfPages(image_file) as pdf:
            plot_count = 0
            for idx in range(len(xprt_locs)):

                # TODO: use log_lines
                print("-" * 30)
                count += 1
                print(f"{count} / {len(xprt_locs)}")

                # select the given expert location
                rl = xprt_locs.iloc[[idx], :]
                print(rl)

                # start timer
                t0 = time.time()

                # ----------------------------
                # (update) global data - from data_source (if need be)
                # ----------------------------

                # TODO: if the prev_where changes - create a new plot, with observations
                # - then for each expert location add the location, color coded by # number of obs
                # - (optional) include the inclusion area

                org_prev_where = prev_where

                df, prev_where = self._update_global_data(df=df,
                                                          global_select=self.data.global_select,
                                                          local_select=self.data.local_select,
                                                          ref_loc=rl,
                                                          prev_where=prev_where,
                                                          col_funcs=self.data.col_funcs)

                if org_prev_where != prev_where:
                    # close any previous plots
                    # save previous plot first?
                    plot_count += 1
                    if plot_count > 1:
                        # save previous fig
                        print(f"plot_count: {plot_count}")
                        plt.tight_layout()
                        pdf.savefig(fig)
                        # plt.show()

                    plt.close()

                    # add / modify the data as need be
                    DataLoader.add_cols(df, col_func_dict=col_funcs)

                    assert lon_col in df, f"lon_col: '{lon_col}' is not in df.columns: {df.columns}"
                    assert lat_col in df, f"lat_col: '{lat_col}' is not in df.columns: {df.columns}"
                    assert obs_col in df, f"obs_col: '{obs_col}' is not in df.columns: {df.columns}"


                    fig, ax = plt.subplots(figsize=figsize,
                                           subplot_kw={'projection': projection})

                    stitle = "\n".join([f"{c}: {rl[c].values[0]}" for c in sort_by])

                    fig.suptitle(stitle)

                    plot_pcolormesh(ax,
                                    lon=df[lon_col],
                                    lat=df[lat_col],
                                    vmin=vmin,
                                    vmax=vmax,
                                    plot_data=df[obs_col],
                                    scatter=True,
                                    s=s,
                                    fig=fig,
                                    cbar_label=cbar_label,
                                    cmap=cmap,
                                    extent=extent)

                    # TODO: allow for histogram as well

                    # fig.suptitle(k)

                # ----------------------------
                # select local data - relative to expert's location - from global data
                # ----------------------------

                # df_local = DataLoader.local_data_select(df,
                #                                         reference_location=rl,
                #                                         local_select=self.data.local_select,
                #                                         verbose=False)
                # print(f"number obs: {len(df_local)}")

                # add expert location as black dot (for now)
                _ = ax.scatter(rl[exprt_lon_col],
                               rl[exprt_lat_col],
                               c="black",
                               # cmap=cmap,
                               # vmin=vmin, vmax=vmax,
                               s=s_exprt_loc,
                               transform=ccrs.PlateCarree(),
                               linewidth=0,
                               rasterized=True)



if __name__ == "__main__":

    pass

