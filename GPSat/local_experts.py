import gc
import os
import re
import sys
import importlib
import warnings
import time
import datetime
import pprint
import json
import inspect

import numpy as np
import pandas as pd
import xarray as xr
from abc import ABC, abstractmethod
from typing import List, Dict, Tuple, Union, Type
from dataclasses import dataclass

try:
    import cartopy.crs as ccrs
except ModuleNotFoundError as e:
    print(f"error importing ccrs from cartopy: {e}\ninstall with: conda install -c conda-forge cartopy=0.20.2")
    ccrs = None

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from GPSat.plot_utils import plot_pcolormesh, plot_hist

from GPSat.decorators import timer
from GPSat.dataloader import DataLoader
from GPSat.models import get_model
from GPSat.prediction_locations import PredictionLocations
from GPSat.utils import json_serializable, check_prev_oi_config, get_previous_oi_config, config_func, \
    dict_of_array_to_dict_of_dataframe, pandas_to_dict, cprint, nested_dict_literal_eval, pretty_print_class
from GPSat.config_dataclasses import (DataConfig, 
                                      ModelConfig,
                                      PredictionLocsConfig,
                                      ExpertLocsConfig,
                                      RunConfig,
                                      ExperimentConfig)

@dataclass
class LocalExpertData:
    # class attributes
    # TODO: fix the type hints below - list of what, other types things can be
    obs_col: Union[str, None] = None
    coords_col: Union[list, None] = None
    global_select: Union[list, None] = None
    local_select: Union[list, None] = None
    where: Union[list, None] = None
    row_select: Union[list, None] = None
    col_select: Union[list, None] = None
    col_funcs: Union[list, None] = None
    table: Union[str, None] = None
    data_source: Union[str, None] = None
    engine: Union[str, None] = None
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
        # NOTE: read_kwargs will be used as 'connection' kwargs for HDFStore, opendataset
        kwargs = self.read_kwargs

        if kwargs is None:
            kwargs = {}
        assert isinstance(kwargs, dict), f"expected additional read_kwargs to be dict (or None), got: {type(kwargs)}"

        # NOTE: self.engine will not get set here if it's None
        self.data_source = DataLoader._get_source_from_str(data_source, _engine=engine, **kwargs)

    def load(self, where=None, verbose=False, **kwargs):
        # wrapper for DataLoader.load, using attributes from self
        # - kwargs provided to load(...)

        # set data_source if it's a string
        if isinstance(self.data_source, str):
            self.set_data_source(verbose=verbose)

        # if self.where is not None, then any additional where's will be added
        # - additional where conditions should be list of dict
        if self.where is not None:
            use_where = self.where
            if where is not None:
                where = where if isinstance(where, list) else [where]
                use_where += where
        else:
            use_where = where

        out = DataLoader.load(source=self.data_source,
                              where=use_where,
                              table=self.table,
                              col_funcs=self.col_funcs,
                              row_select=self.row_select,
                              col_select=self.col_select,
                              engine=self.engine,
                              source_kwargs=self.read_kwargs,
                              verbose=verbose,
                              **kwargs)

        return out


# TODO: change print statements to use logging
class LocalExpertOI:
    """
    This provides the main interface for conducting an experiment in ``GPSat`` to predict
    an underlying field from satellite measurements using local Gaussian process (GP) models.

    This proceeds by iterating over the local expert locations, training the local GPs on data
    in a neighbourhood of the expert location and making predictions on specified locations.
    The results will be saved in an HDF5 file.

    Example usage:

    >>> store_path = "/path/to/store.h5"
    >>> locexp = LocalExpertOI(data_config, model_config, expert_loc_config, pred_loc_config)
    >>> locexp.run(store_path=store_path) # Run full sweep and save results in store_path

    """

    # when reading in data
    file_suffix_engine_map = {
        "csv": "read_csv",
        "tsv": "read_csv",
        "h5": "HDFStore",
        "zarr": "zarr",
        "nc": "netcdf4"
    }

    def __init__(self,
                 expert_loc_config: Union[Dict, ExpertLocsConfig, None] = None,
                 data_config: Union[Dict, DataConfig, None] = None,
                 model_config: Union[Dict, ModelConfig, None] = None,
                 pred_loc_config: Union[Dict, PredictionLocsConfig, None] = None,
                 local_expert_config: Union[ExperimentConfig, None] = None):
        """
        Parameters
        ----------
        expert_loc_config: dict or ExpertLocsConfig
            Configuration for expert locations.
        data_config: dict or DataConfig
            Configuration for data to be interpolated.
        model_config: dict or ModelConfig
            Configuration for model used to perform the local optimal interpolation.
        pred_loc_config: dict or PredictionLocsConfig
            Configuration for prediction locations.
        local_expert_config: ExperimentConfig, optional
            If the above four configurations are stored in ``ExperimentConfig``, you can pass this all at once
            by specifying a single ``ExperimentConfig``.

        Notes
        -----
        See :doc:`configuration dataclasses <config_classes>` for more details on the
        specific configuration classes.

        """

        if local_expert_config is not None:
            expert_loc_config = local_expert_config.expert_locs_config.to_dict_with_dataframe()
            data_config = local_expert_config.data_config.to_dict_with_dataframe()
            model_config = local_expert_config.model_config.to_dict()
            pred_loc_config = local_expert_config.prediction_locs_config.to_dict_with_dataframe()

        # TODO: make locations, data, model attributes with arbitrary structures
        #  maybe just dicts with their relevant attributes stored within

        self.constraints = None
        self.model_init_params = None
        self.model_load_params = None
        self.model = None
        self.data_table = None

        # data will be set as LocalExpertData instance
        self.data = None
        # expert locations will be a pandas DataFrame
        self.expert_locs = None
        # config will be used to store the parameters used to set: locations, data, model
        self.config = {}

        # ------
        # Local Expert Locations
        # ------
        expert_loc_config = expert_loc_config.to_dict_with_dataframe() if isinstance(expert_loc_config, ExpertLocsConfig) else expert_loc_config
        locations = self._none_to_dict_check(expert_loc_config)

        self.set_expert_locations(**locations)

        # ------
        # Data (source)
        # ------
        data_config = data_config.to_dict_with_dataframe() if isinstance(data_config, DataConfig) else data_config
        data_config = self._none_to_dict_check(data_config)

        self.set_data(**data_config)

        # ------
        # Model
        # ------
        model_config = model_config.to_dict() if isinstance(model_config, ModelConfig) else model_config
        model_config = self._none_to_dict_check(model_config)

        self.set_model(**model_config)

        # ------
        # Prediction Locations
        # ------
        pred_loc_config = pred_loc_config.to_dict_with_dataframe() if isinstance(pred_loc_config, PredictionLocsConfig) else pred_loc_config
        pred_loc_config = self._none_to_dict_check(pred_loc_config)

        self.set_pred_loc(**pred_loc_config)

    def _none_to_dict_check(self, x):
        if x is None:
            x = {}
        assert isinstance(x, dict)
        return x

    def _method_inputs_to_config(self, locs, code_obj, verbose=False):
        # TODO: validate this method returns expected values
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
                  optim_kwargs=None,
                  pred_kwargs=None,
                  params_to_store=None,
                  replacement_threshold=None,
                  replacement_model=None,
                  replacement_init_params=None,
                  replacement_constraints=None,
                  replacement_optim_kwargs=None,
                  replacement_pred_kwargs=None):

        # TODO: non JSON serializable objects may cause issues if trying to re-run with later
        self.config["model"] = self._method_inputs_to_config(locals(), self.set_model.__code__)

        # oi_model is a str then expect to be able to import from models
        # TODO: perhaps would like to generalise this a bit more - read models from different modules
        self.model = oi_model

        # oi_model is a str then expect to be able to import from models
        # TODO: perhaps would like to generalise this a bit more - read models from different modules
        if isinstance(self.model, str):
            # Default GPSat models can be accessed via `get_model()`
            self.model = get_model(self.model)
        elif isinstance(self.model, dict):
            # For custom models, specify a dictionary containing the path to the model and the model name itself
            model_path = self.model['path_to_model']
            model_name = self.model['model_name']
            sys.path.append(model_path)
            module = importlib.import_module(model_path)
            self.model = getattr(module, model_name)

        # TODO: should these only be set if they are not None?
        self.model_init_params = {} if init_params is None else init_params
        self.constraints = constraints
        self.model_load_params = load_params
        self.optim_kwargs = {} if optim_kwargs is None else optim_kwargs
        self.pred_kwargs = {} if pred_kwargs is None else pred_kwargs

        if params_to_store == 'all':
            self.params_to_store = None
        else:
            self.params_to_store = params_to_store

        # Replacement model (used to substitute the main model if number of training points is < replacement_threshold)
        if replacement_threshold is not None:
            self.replacement_threshold = replacement_threshold
            self.replacement_model = self.model if replacement_model is None else get_model(replacement_model) # getattr(models, replacement_model)
            self.replacement_init_params = init_params if replacement_init_params is None else replacement_init_params
            self.replacement_constraints = constraints if replacement_constraints is None else replacement_constraints
            self.replacement_optim_kwargs = {} if replacement_optim_kwargs is None else replacement_optim_kwargs
            self.replacement_pred_kwargs = {} if replacement_pred_kwargs is None else replacement_pred_kwargs


    def set_expert_locations(self,
                             df=None,
                             file=None,
                             source=None,
                             where=None,
                             add_data_to_col=None,
                             col_funcs=None,
                             keep_cols=None,
                             col_select=None,
                             row_select=None,
                             sort_by=None,
                             reset_index=False,
                             source_kwargs=None,
                             verbose=False,
                             **kwargs):

        # TODO: remove some of the inputs here, just use kwargs, which will be passed to DataLoader.load
        #  - with the exception of some legacy inputs, (df, file, keep_cols), for backwards compatibility

        # TODO: remove redundant inputs - move fully to using DataLoader.load parameters

        # TODO: if verbose print what the input parameters are?
        # TODO: allow for dynamically created local expert locations
        #  - e.g. provide grid spacing, mask types (spacing, over ocean only)

        # --
        # override legacy parameters
        # --

        if (col_select is None) & (keep_cols is not None):
            warnings.warn("\n'keep_cols' provided to set_expert_locations, use 'col_select' instead")
            col_select = keep_cols

        if (source is None) & (df is not None):
            warnings.warn("\n'df' was provided to set_expert_locations, use 'source' instead")
            source = df

        if (source is None) & (file is not None):
            warnings.warn("\n'df' was provided to set_expert_locations, use 'source' instead")
            source = file

        # if source is None, do nothing
        if source is None:
            return None

        # --
        # store parameters to config
        # --

        # TODO: none JSON serializable objects may cause issues if trying to re-run with later
        self.config["locations"] = self._method_inputs_to_config(locals(), self.set_expert_locations.__code__)

        if verbose:
            print(f"local_expert_locations: using DataLoader.load() to fetch")

        # fetch data
        locs = DataLoader.load(source=source,
                               where=where,
                               source_kwargs=source_kwargs,
                               col_funcs=col_funcs,
                               row_select=row_select,
                               col_select=col_select,
                               reset_index=reset_index,
                               add_data_to_col=add_data_to_col,
                               verbose=verbose,
                               **kwargs)

        # sort rows?
        if sort_by:
            if verbose:
                print(f"sorting values by: {sort_by}")
            locs.sort_values(sort_by, inplace=True)

        self.expert_locs = locs


    # @timer
    def _update_global_data(self,
                            df=None,
                            global_select=None,
                            local_select=None,
                            ref_loc=None,
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
            # DataLoader.load calls data_select, add_cols, plus can apply row_select
            df = DataLoader.load(source=self.data.data_source,
                                 table=self.data.table,
                                 where=where,
                                 col_funcs=self.data.col_funcs,
                                 row_select=self.data.row_select,
                                 col_select=self.data.col_select,
                                 reset_index=True,
                                 verbose=False)

        return df, where

    @staticmethod
    def _remove_previously_run_locations(store_path, xprt_locs, table="run_details", row_select=None):
        # read existing / previous results
        try:
            prev_res = DataLoader.load(source=store_path, table=table, row_select=row_select, reset_index=False)
            idx_names = prev_res.index.names
            prev_res = prev_res.reset_index()[idx_names]

            # left join to find which have not be found (left_only)
            tmp = xprt_locs.merge(prev_res,
                                  how='left',
                                  on=prev_res.columns.values.tolist(),
                                  indicator='found_already')
            # create bool array of those to keep
            keep_bool = tmp['found_already'] == 'left_only'
            print(f"for table: {table} returning {keep_bool.sum()} / {len(keep_bool)} entries")
            xprt_locs = xprt_locs.loc[keep_bool.values].copy(True)

        except OSError as e:
            print(e)
        except KeyError as e:
            print(e)

        return xprt_locs

    @staticmethod
    def _append_to_store_dict_or_write_to_table(save_dict, store_path,
                                                store_dict=None,
                                                store_every=1,
                                                table_suffix=""):
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
            cprint("SAVING RESULTS TO TABLES:", c="OKCYAN")
            for k, v in store_dict.items():
                print(k)
                df_tmp = pd.concat(v, axis=0)
                try:
                    # HARDCODED: min_itemsize for specific columns, to allow for adding of strings of longer
                    #  - length than previous ones.
                    # TODO: review the size used here, will it have a high storage cost?
                    min_itemsize = {c: 64 for c in df_tmp.columns if c in ["model", "device"]}
                    with pd.HDFStore(store_path, mode='a') as store:
                        # tmp = store.get(f"{k}{table_suffix}")
                        # TODO: here, why not using data_columns=True? - will this cause issue searching later
                        #  - if coords_col are in index should be able to search by them, is that enough?
                        # store.append(key=k, value=df_tmp, min_itemsize=min_itemsize, data_columns=True)
                        store.append(key=f"{k}{table_suffix}", value=df_tmp, min_itemsize=min_itemsize)
                except ValueError as e:
                    print(e)
                except Exception as e:
                    print(e)
            store_dict = {}

        return store_dict

    # @timer
    def load_params(self,
                    model,
                    previous=None,
                    previous_params=None,
                    file=None,
                    param_names=None,
                    ref_loc=None,
                    index_adjust=None,
                    table_suffix="",
                    **param_dict):

        # TODO: add verbose print / log lines here
        # method to load (set) parameters - either from (h5) file, or specified directly
        # via param_dict

        # if file is None - provide param_dict
        if file is not None:

            assert isinstance(file, str), f"in load_params file provided but is not str, got: {type(file)}"
            assert os.path.exists(file), f"in load_params file provided:\n{file}\nbut path does not exist"

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

            # TODO: probably worth refactoring this method
            # NOTE: it is possible only to load some parameters, and not others
            # - will return 1 if can't load any
            param_dict = self._read_params_from_file(file=file,
                                                     model=model,
                                                     ref_loc=rl,
                                                     param_names=param_names,
                                                     table_suffix=table_suffix)
            if len(param_dict) == 0:
                return 1

        # load previous params?
        elif previous is not None:
            param_dict = previous_params
            if param_dict is None:
                param_dict = {}

        model.set_parameters(**param_dict)

        return 0

    @timer
    def _read_params_from_file(self,
                               model,
                               file,
                               ref_loc,
                               param_names=None,
                               table_suffix="") -> dict:
        """
        for a given reference location and (h5) file, select the entry corresponding to
        the reference location and extract values.
        returns a dict of numpy arrays to be used by model.set_parameters()
        """

        # TODO: use a verbose level (should be set as attribute when initialised?)
        assert isinstance(ref_loc, (
        pd.Series, pd.DataFrame, dict)), f"ref_loc expected to pd.Series or dict, got: {type(ref_loc)}"

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
                    tmp_df = store.select(f"{k}{table_suffix}", where=rl_where)
                    if len(tmp_df) == 0:
                        warnings.warn(f"\n******\nno parameters found in table:\n{k}\nfor where:\n{rl_where}\n******")
                        continue
                    tmp = DataLoader.mindex_df_to_mindex_dataarray(df=tmp_df,
                                                                   data_name=k,
                                                                   infer_dim_cols=True)
                    out[k] = tmp.values[0]

                    # nan check - should this be done else where
                    if isinstance(out[k], np.ndarray):
                        if np.any(np.isnan(out[k])):
                            warnings.warn(
                                f"\n{k}: found some nans for ref location: {ref_loc}, removing those parameters")
                            out.pop(k)
                    elif isinstance(out[k], float):
                        if np.isnan(out[k]):
                            warnings.warn(
                                f"\n{k}: found some nans for ref location: {ref_loc}, removing those parameters")
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

        assert isinstance(x, dict), f"input expected to be dict, got: {type(x)}"
        # if empty dict just return
        if len(x) == 0:
            return x

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
                # create a index/multi index of length equal to DataFrame
                df = dfs[k]
                if len(ref_loc) == 1:
                    midx = pd.Index([midx_tuple[0]] * len(df), name=midx_names[0])
                else:
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
            for k, v in dfs.items():
                if k == default_dim:
                    out[table] = v
                else:
                    out[f"{table}_{k}"] = v

        return out

    def _same_param_table(self, file, table_suffix, model_load_params):
        # identify is the file and table_suffix is the same in model_load_params
        assert isinstance(model_load_params, dict), \
            f"model_load_params expected to be dict, got: {type(model_load_params)}"
        file_match = file == model_load_params.get("file", None)
        suffix_match = table_suffix == model_load_params.get("table_suffix", None)
        # require no additional keyword arguments
        additional_kwargs = [k for k in model_load_params.keys() if k not in ["file", "table_suffix"]]
        # identify if saving to same parameter table(s) if: file_match, suffix_match and there are no additional kwargs
        return file_match & suffix_match & (len(additional_kwargs) == 0)

    # @timer
    def run(self,
            store_path=None,
            store_every=10,
            check_config_compatible=True,
            skip_valid_checks_on=None,
            optimise=True,
            predict=True,
            min_obs=3,
            table_suffix=""):
        """
        Run a full sweep to perform local optimal interpolation at every expert location.
        The results will be stored in an HDF5 file containing (1) the predictions at each location,
        (2) parameters of the model at each location, (3) run details such as run times, and
        (4) the full experiment configuration.

        Parameters
        ----------
        store_path: str
            File path where results should be stored as HDF5 file.
        store_every: int, default 10
            Results will be stored to file after every ``store_every expert`` locations.
            Reduce if optimisation is slow, must be greater than 1.
        check_config_compatible: bool, default True
            Check if current ``LocalExpertOI`` configuration is compatible
            with previous, if applicable. If file exists in ``store_path``, it will check the ``oi_config`` attribute in the
            ``oi_config`` table to ensure that configurations are compatible.
        skip_valid_checks_on: list, optional
            When checking if config is compatible, skip keys specified in this list.
        optimise: bool, default True
            If ``True``, will run ``model.optimise_parameters()`` to learn the model parameters at each expert location.
        predict: bool, default True
            If ``True``, will run ``model.predict()`` to make predictions at the locations specified in
            the prediction locations configuration.
        min_obs: int, default 3
            Minimum number observations required to run optimisation or make predictions.
        table_suffix: str, optional
            Suffix to be appended to all table names when writing to file.

        Returns
        -------
        None

        Notes
        -----
            - By default, both training and inference are performed at every location.
              However one can opt to do either one with the ``optimise`` and ``predict`` options, respectively.
            - If ``check_config_compatible`` is set to ``True``, it makes sure that all results saved to ``store_path``
              use the same configurations. That is, if one re-runs an experiment with a different configuration but pointing to
              the same ``store_path``, it will return an error. Make sure that if you run an experiment with a different configuration,
              either set a different ``store_path``, or if you want to override the results, delete the generated ``store_path``.
            - The ``table_suffix`` is useful for storing multiple results in a single HDF5 file, each with a different suffix.
              See <hyperparameter smoothing> for an example use case.

        """

        # TODO: add model name to print / progress
        # store run kwargs in self.config, as to allow full reproducibility
        # NOTE: this does not work due to the @timer decorator. replaced @timer with _t0, _t1
        # TODO: could this be replaced by a decorator? - assign values to config attribute?
        self.config["run_kwargs"] = self._method_inputs_to_config(locals(), self.run.__code__)

        _t0 = time.perf_counter()

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

        # store every
        if not isinstance(store_every, int):
            store_every = int(store_every)
        assert store_every >= 1, f"store_every must be >= 1, got: {store_every}"

        # min_obs
        if not isinstance(min_obs, int):
            min_obs = int(min_obs)
        assert min_obs >= 1, f"min_obs must be >= 1, got: {min_obs}"

        # create directory for store_path if it does not exist
        os.makedirs(os.path.dirname(store_path), exist_ok=True)

        # -----
        # store / check config
        # -----

        # get previous_oi_config (if exists)
        # - check current config matches previous, if it does get previous config's idx
        # - otherwise add current config to f"oi_config{table_suffix}", will create table it does not exist
        # - getting previous config_id allows for skipping of expert locations that were already run using that config
        # TODO: review checking of previous configs
        prev_oi_config, skip_valid_checks_on, config_id = get_previous_oi_config(store_path,
                                                                                 oi_config=self.config,
                                                                                 skip_valid_checks_on=skip_valid_checks_on,
                                                                                 table_name=f"oi_config{table_suffix}")

        # check configuration is compatible with previously used, if applicable
        if check_config_compatible:
            # check previous oi_config matches current - want / need them to be consistent (up to a point)
            # TODO: should ALWAYS check 'data' and 'model' are compatible with previous run
            check_prev_oi_config(prev_oi_config,
                                 oi_config=self.config,
                                 skip_valid_checks_on=skip_valid_checks_on)

        # -------
        # store (new) locations, remove those already found
        # -------

        # store all expert locations in a table,
        #  - if table already exists only append new position
        #  - when appending if column names differ, only take previously existing, provide warning
        #  - and if not all previously existing columns exist Raise error

        # get any previously un-stored expert locations
        print(f"---------\nstoring expert locations in 'expert_locs' table")
        store_locs = self._remove_previously_run_locations(store_path,
                                                           xprt_locs=self.expert_locs.copy(True),
                                                           table=f"expert_locs{table_suffix}")
        # set index and write to table (this could be done more cleanly)
        store_locs.set_index(self.data.coords_col, inplace=True)
        with pd.HDFStore(store_path, mode="a") as store:
            store.append(f"expert_locs{table_suffix}", store_locs, data_columns=True)

        # remove previously found local expert locations
        # - determined by (multi-index of) 'run_details' table
        cprint(f"---------\ndropping expert locations that already exists in 'run_details' table", c="OKCYAN") #
        xprt_locs = self._remove_previously_run_locations(store_path,
                                                          xprt_locs=self.expert_locs.copy(True),
                                                          table=f"run_details{table_suffix}",
                                                          # row_select={"col": "config_id", "comp": "==", "val": config_id}
                                                          )

        # TODO: want to store prediction locations in a table? unique values only
        #  - chould be useful to have different types of predictions together, with a column inidicating type
        #  - e.g. pred_type: xval, pan_arctic, whatever.
        #  - currently to create separate files

        # -----
        # iterate over expert locations
        # -----


        # create a dictionary to store result (DataFrame / tables)
        store_dict = {}
        prev_params = {}
        count = 0
        df, prev_where = None, None
        # for idx, rl in xprt_locs.iterrows():
        for idx in range(len(xprt_locs)):

            # TODO: create a private method that takes in a given expert location, data, model info and runs OI
            #  - i.e. wrap the contents of this for loop into a method
            # TODO: use log_lines
            cprint("-" * 50, c="BOLD")
            count += 1
            cprint(f"{count} / {len(xprt_locs)}", c="OKCYAN")

            # select the given expert location
            rl = xprt_locs.iloc[[idx], :]
            cprint("current local expert:", c="OKCYAN")
            print(rl)

            # start timer
            t0 = time.time()

            # ----
            # get prediction location(s)
            # ----

            # TODO: making predictions should be optional, if not making predictions set pred={}
            # TODO: allow for pred_loc to return empty array / None (skip predictions) - confirm this is the case

            # prediction locations are static (once loaded)
            # - it's quick to check if expert location is close, then skip if not

            # update the expert location for the PredictionLocation attribute
            self.pred_loc.expert_loc = rl
            # generate the expert locations
            prediction_coords = self.pred_loc()

            if len(prediction_coords) == 0:
                cprint("there are no predictions locations, skipping", c="WARNING")
                # TODO: should the run_details be store here - to avoid re-running on restart
                continue

            # ----------------------------
            # (update) global data - from data_source (if need be)
            # ----------------------------

            df, prev_where = self._update_global_data(df=df,
                                                      global_select=self.data.global_select,
                                                      local_select=self.data.local_select,
                                                      ref_loc=rl,
                                                      prev_where=prev_where)

            # ----------------------------
            # select local data - relative to expert's location - from global data
            # ----------------------------

            df_local = DataLoader.local_data_select(df,
                                                    reference_location=rl,
                                                    local_select=self.data.local_select,
                                                    verbose=False)
            cprint(f"number obs: {len(df_local)}", c="OKCYAN")

            # if there are too few observations store to 'run_details' (so can skip later) and continue
            if len(df_local) < min_obs:
                # for too few run obs record their entry, meaning they will skipped over if process is restarted
                # TODO: determine if this is the desired functionality
                run_details = {
                    "num_obs": len(df_local),
                    "run_time": np.nan,
                    "objective_value": np.nan,
                    "parameters_optimised": optimise,
                    "optimise_success": False,
                    "model": pretty_print_class(self.model)[:64],  # _model.__class__.__name__,
                    "device": "",
                    "config_id": config_id,
                }
                save_dict = self.dict_of_array_to_table(run_details,
                                                        ref_loc=rl[self.data.coords_col],
                                                        concat=True,
                                                        table="run_details")

                store_dict = self._append_to_store_dict_or_write_to_table(save_dict=save_dict,
                                                                          store_dict=store_dict,
                                                                          store_path=store_path,
                                                                          store_every=store_every,
                                                                          table_suffix=table_suffix)

                continue


            # -----
            # build model - provide with data
            # -----

            # initialise model
            # TODO: needed to review the unpacking of model_params, when won't it work?
            if hasattr(self, "replacement_threshold"):
                # Use replacement GPR model if the number of data points is lower than [replacement_threshold]
                if len(df_local) < self.replacement_threshold:
                    print("Setting model to replacement GPR...")
                    _model = self.replacement_model
                    _init_params = self.replacement_init_params
                    _constraints = self.replacement_constraints
                    _optim_kwargs = self.replacement_optim_kwargs
                    _pred_kwargs = self.replacement_pred_kwargs
                else:
                    _model = self.model
                    _init_params = self.model_init_params
                    _constraints = self.constraints
                    _optim_kwargs = self.optim_kwargs
                    _pred_kwargs = self.pred_kwargs
            else:
                _model = self.model
                _init_params = self.model_init_params
                _constraints = self.constraints
                _optim_kwargs = self.optim_kwargs
                _pred_kwargs = self.pred_kwargs

            model = _model(data=df_local,
                           obs_col=self.data.obs_col,
                           coords_col=self.data.coords_col,
                           # ideally prefer not to have a specific model's key word argument explicitly given like this
                           # should be handled in _init_params.
                           expert_loc=rl[self.data.coords_col].to_numpy().squeeze(),  # Needed for VFF / ASVGP
                           **_init_params)

            # *****************
            # here should simply use: set_parameters -  refactor this section
            #
            # a models set_parameters method should have
            # - have arguments value(s), plus some **kwargs
            # - additional keyword arguments should allow to: set constraints, set trainable, etc
            # - a model config could then have a (optional) parameters key, containing valid param_names allowing to
            # - - set values, constraints, trainable, etc (will depend on the model being used)

            # ----
            # load parameters (optional)
            # ----

            # if there are no previous parameters - get the default ones
            if len(prev_params) == 0:
                prev_params = model.get_parameters()

            # TODO: implement this - let them either be previous values, fixed or read from file
            # TODO: review different ways parameters can be loaded: - from file, fixed values,
            #   previously found (optimise success =True)

            # parameters generally should be stored, unless
            # loading parameters from same file and table suffix as
            save_params = True
            if self.model_load_params is not None:

                # HACK: for loading previously found optimal parameters
                # TODO: allow for only a subset of these to be set - e.g. skip variational parameters
                if self.model_load_params.get("previous", False):
                    print("will load previously found params:")
                    pprint.pprint(prev_params, width=1)
                    # print(prev_params)
                    self.model_load_params["previous_params"] = prev_params

                # load params, getting status of load (0 is success)
                lp_status = self.load_params(ref_loc=rl,
                                             model=model,
                                             **self.model_load_params)

                # will parameters be (attempted) to be stored in the same table as being loaded from?
                same_param_table = self._same_param_table(file=store_path,
                                                          table_suffix=table_suffix,
                                                          model_load_params=self.model_load_params)
                # if so, and not optimising, don't try to save them
                # - this is just to avoid printing Error messages handled by a try / except
                # - in _append_to_store_dict_or_write_to_table
                save_params = not (same_param_table & (not optimise))

                if lp_status > 0:
                    print("there was an issue loading params, skipping this local expert")
                    continue

            # --
            # apply constraints
            # --

            # TODO: generalise this to apply any constraints - use apply_param_transform (may require more checks)
            #  - may need information from config, i.e. obj = model.kernel, specify the bijector, other parameters

            if _constraints is not None:
                if isinstance(_constraints, dict):
                    # Apply coordinate scaling to lengthscale hyperparameters if applicable
                    if self.model_init_params.get('coords_scale', None) is not None:
                        _constraints["lengthscales"]["scale"] = True
                    model.set_parameter_constraints(_constraints, move_within_tol=True, tol=1e-2)
                else:
                    warnings.warn(f"constraints: {_constraints} are not currently handled!")

            # **********************************

            # --
            # optimise parameters
            # --

            # (optionally) optimise parameters
            if optimise:
                opt_success = model.optimise_parameters(**_optim_kwargs)
            else:
                # TODO: only print this if verbose (> some level?)
                cprint("*** not optimising parameters", c="WARNING")
                # if not optimising set opt_success to False
                opt_success = False

            # get the final / current objective function value
            final_objective = model.get_objective_function_value()
            # get the hyper parameters - for storing
            # quick bug fix: params_to_store can be None, however *None does not work
            pts = [] if self.params_to_store is None else self.params_to_store
            hypes = model.get_parameters(*pts)

            # print (truncated) parameters
            cprint("parameters:", c="OKCYAN")
            for k, v in hypes.items():
                if isinstance(v, np.ndarray):
                    print(f"{k}: {repr(v[:5])} {'(truncated) ' if len(v) > 5 else ''}")
                else:
                    print(f"{k}: {v}")

            # if not saving parameters set hypes to empty dict
            if not save_params:
                hypes = {}

            # --
            # make prediction
            # --

            if predict & (len(prediction_coords) > 0):

                pred = model.predict(coords=prediction_coords,  **_pred_kwargs)

                # add prediction coordinate location
                for ci, c in enumerate(self.data.coords_col):
                    # TODO: review if want to force coordinates to be float
                    pred[f'pred_loc_{c}'] = prediction_coords[:, ci]
            else:
                if len(prediction_coords) == 0:
                    print("*** no predictions are being made because prediction_coords has len 0")
                elif predict is False:
                    print("*** no predictions made")
                pred = {}

            # ----
            # store results in tables (keys) in hdf file
            # ----

            t1 = time.time()
            run_time = t1 - t0

            # get the device name from the model
            device_name = model.cpu_name if model.gpu_name is None else model.gpu_name

            # delete model to try to handle Out of Memory issue?
            del model
            gc.collect()

            # run details / info - for reference
            run_details = {
                "num_obs": len(df_local),
                "run_time": run_time,
                "objective_value": final_objective,
                "parameters_optimised": optimise,
                "optimise_success": opt_success,
                "model": pretty_print_class(_model)[:64],  # _model.__class__.__name__,
                "device": device_name[:64],
                "config_id": config_id,
            }

            # TODO: refactor this - only needed if loading/initialising with previous parameters
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
                        try:
                            prev_params[k] = rho * prev_params[k] + (1 - rho) * hypes[k]
                        except ValueError as e:
                            # if not loading previous parameters can just ignore any isus
                            if self.model_load_params is not None:
                                if self.model_load_params.get("previous", False):
                                    # ValueError could arise if parameters shape changes, namely for inducing points
                                    cprint(f"in updating prev_params for: {k}", c="WARNING")
                                    cprint(e, c="WARNING")

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
                                                                      store_every=store_every,
                                                                      table_suffix=table_suffix)

            t2 = time.time()
            cprint(f"total run time : {t2 - t0:.2f} seconds", c="OKGREEN")

        # ---
        # store any remaining data
        # ---

        if len(store_dict):
            print("storing any remaining tables")
            self._append_to_store_dict_or_write_to_table(save_dict={},
                                                         store_dict=store_dict,
                                                         store_path=store_path,
                                                         store_every=1,
                                                         table_suffix=table_suffix)

        _t1 = time.perf_counter()

        print(f"'run': {_t1 - _t0:.3f} seconds")

        # explicitly return None
        return None
        

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

        # TODO: review this method
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
            # TODO: here should check the projection is of the correct instance
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
        os.makedirs(os.path.dirname(image_file), exist_ok=True)
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
                                                          prev_where=prev_where)

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

                    # plot the observations
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

            # save final figure (?)
            plt.tight_layout()
            pdf.savefig(fig)



def get_results_from_h5file(results_file,
                            global_col_funcs=None,
                            merge_on_expert_locations=True,
                            select_tables=None,
                            table_suffix="",
                            add_suffix_to_table=True,
                            verbose=False):
    """
    Retrieve results from an HDF5 file.

    Parameters
    ----------
    results_file: str
        The location where the results file is saved. Must point to a HDF5 file with the file extension ``.h5``.
    select_tables: list, optional
        A list of table names to select from the HDF5 file.
    global_col_funcs: dict, optional
        A dictionary of column functions to apply to selected tables.
    merge_on_expert_locations: bool, default True
        Whether to merge expert location data with results data.
    table_suffix: str, optional
        A suffix to add to selected table names.
    add_suffix_to_table: bool, default True
        Whether to add the table suffix to selected table names.
    verbose: bool, default False
        Set verbosity.

    Returns
    -------
    tuple:
        A tuple containing two elements:

        1. ``dict``: A dictionary of DataFrames where each table name is the key. \
            This contains the predictions and learned model parameters at every location.
        2. ``list``: A list of configuration dictionaries.

    Notes
    -----
        - This function reads data from an HDF5 file, applies optional column functions, and optionally merges
          expert location data with results data.
        - The ``'select_tables'`` parameter allows you to choose specific tables from the HDF5 file.
        - Column functions specified in ``'global_col_funcs'`` can be applied to selected tables.
        - Expert location data can be merged onto results data if ``'merge_on_expert_locations'`` is set to ``True``.

    """

    if select_tables is not None:
        if add_suffix_to_table:
            select_tables = [f"{table}{table_suffix}" for table in select_tables]

    # TODO: provide a single table_suffix
    # get the configuration file
    # TODO: get the list of configs
    with pd.HDFStore(results_file, mode='r') as store:

        try:
            config_df = store[f'oi_config{table_suffix}'][['config']].drop_duplicates()
            oi_config = [nested_dict_literal_eval(json.loads(c)) for c in config_df['config'].values]
        except Exception as e:
            print(f"issuing getting oi_config{table_suffix}")
            try:
                oi_config = store.get_storer(f'oi_config{table_suffix}').attrs['oi_config']
                oi_config = [nested_dict_literal_eval(oi_config)]
            except Exception as e:
                oi_config = []

    # --
    # read in results, store in dict with table as key
    # --

    print("reading in results")
    with pd.HDFStore(results_file, mode="r") as store:
        # TODO: determine if it's faster to use select_colum - does not have where condition?

        all_keys = [re.sub("^/", "", k ) for k in store.keys()]
        if select_tables is None:
            select_tables = all_keys
            print("getting all tables")
        else:
            print(f"selecting only tables: {select_tables}")

        # dfs = {re.sub("/", "", k): store.select(k, where=None).reset_index()
        #        for k in all_keys}
        dfs = {}
        for k in all_keys:

            if k not in select_tables:
                if verbose:
                    print(f"{k} not in select_tables, skipping")
                continue

            try:
                dfs[re.sub("/", "", k)] = store.select(k, where=None).reset_index()
            except Exception as e:
                print("*" * 70)
                print(f"issue with key: {k}")
                print(e)
                print("*" * 70)

        # modify / add columns using global_col_funcs
        if global_col_funcs is not None:
            print("applying global_col_funcs")
            for k in dfs.keys():
                try:
                    DataLoader.add_cols(df=dfs[k], col_func_dict=global_col_funcs)
                except Exception as e:
                    print(f"Adding/Modifying columns had Exception:{e}\non key/table: {k}")

    # ---
    # expert locations - additional info
    # ---

    expert_locations = None
    # if 'expert_locations' does not exist in result, then (try) to read from file
    # NOTE: this does not handle different table_suffix!
    if f'expert_locs{table_suffix}' not in dfs:
        try:
            expert_locations = []
            for conf in oi_config:
                leoi = LocalExpertOI(expert_loc_config=conf['locations'])
                expert_locations.append(leoi.expert_locs.copy(True))
            expert_locations = pd.concat(expert_locations)
        except Exception as e:
            print(f"in get_results_from_h5file trying read expert_locations from file got Exception:\n{e}")
    else:
        expert_locations = dfs[f'expert_locs{table_suffix}'].copy(True)

    # (optionally) merge on
    if (expert_locations is not None) & (merge_on_expert_locations):

        print("merging on expert location data")
        # get the coordinates columns
        # - try / except to handle legacy format
        try:
            coords_col = oi_config[0]['data']['coords_col']
        except KeyError:
            coords_col = oi_config[0]['input_data']['coords_col']

        for k in dfs.keys():

            if np.in1d(coords_col, dfs[k].columns).all():
                # if there a duplicates in columns suffixes will be added
                dfs[k] = dfs[k].merge(expert_locations,
                                      on=coords_col,
                                      how='left',
                                      suffixes=["", "_expert_location"])
            else:
                print(f"table: '{k}' does not have all coords_col: {coords_col} in columns, "
                      f"not merging on expert_locations")

    else:
        print("expert_locations data will not be merged on results data")

    return dfs, oi_config


if __name__ == "__main__":
    pass
