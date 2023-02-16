
import os
import re
import time
import datetime
import gpflow
import numpy as np
import pandas as pd
import xarray as xr
from abc import ABC, abstractmethod
from typing import List, Dict, Tuple, Union, Type

import warnings

from PyOptimalInterpolation.decorators import timer
from PyOptimalInterpolation.dataloader import DataLoader
import PyOptimalInterpolation.models as models
from PyOptimalInterpolation.models import BaseGPRModel
from PyOptimalInterpolation.utils import json_serializable, check_prev_oi_config, get_previous_oi_config, config_func, \
    dict_of_array_to_dict_of_dataframe

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
                 locations: Union[Dict, None]=None,
                 data: Union[Dict, None]=None,
                 model: Union[Dict, None]=None):

        # TODO: make locations, data, model attributes with arbitrary structures
        #  maybe just dicts with their relevant attributes stored within

        self.constraints = None
        self.model_init_params = None
        self.model_load_params = None
        self.model = None
        self.local_select = None
        self.global_select = None
        self.coords_col = None
        self.obs_col = None

        
        self.config = {}

        # ------
        # Location
        # ------

        # set self.expert_locs
        if locations is None:
            locations = {}
        assert isinstance(locations, dict)

        self.set_expert_locations(**locations)

        # ------
        # Data (source)
        # ------

        if data is None:
            data = {}
        assert isinstance(data, dict)

        self.set_data(**data)

        # ------
        # Model
        # ------

        if model is None:
            model = {}
        assert isinstance(model, dict)
        
        self.set_model(**model)


    def set_data(self,
                 data_source=None,
                 table=None,
                 engine=None,
                 obs_col=None,
                 coords_col=None,
                 global_select=None,
                 local_select=None,
                 col_funcs=None):

        # --
        # store parameters to config
        # --

        # TODO: non JSON serializable objects may cause issues if trying to re-run with later
        config = {}
        locs = locals()
        for k in range(self.set_data.__code__.co_argcount):
            var = self.set_data.__code__.co_varnames[k]
            if var == "self":
                continue
            else:
                config[var] = locs[var]
        self.config["data"] = json_serializable(config)
        
        # ---

        # TODO: 'file' should be changed to data_source
        if data_source is not None:
            # set data_source attribute
            self.set_data_source(data_source=data_source,
                                 engine=engine)

        # set data related attributes
        # TODO: should these be stored nested in another attribute - use dict?
        # for _ in ['obs_col', 'coords_col', 'global_select', 'local_select', 'col_funcs']:
        #     if _ in data:
        #         setattr(self, _, data[_])
        #     else:
        #         setattr(self, _, None)

        # TODO: assign these differently
        self.obs_col = obs_col
        self.coords_col = coords_col
        self.global_select = global_select
        self.local_select = local_select
        self.col_funcs = col_funcs
        self.table = table


    def set_data_source(self, data_source, engine=None, verbose=False, **kwargs):

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
        self._data_engine = engine

        self.data_source = None
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
                          f"data_source was not set")
            self._data_engine = None

    def set_model(self, oi_model, init_params=None, constraints=None, load_params=None):

        # TODO: non JSON serializable objects may cause issues if trying to re-run with later
        config = {}
        locs = locals()
        for k in range(self.set_model.__code__.co_argcount):
            var = self.set_model.__code__.co_varnames[k]
            if var == "self":
                continue
            else:
                config[var] = locs[var]
        self.config["model"] = json_serializable(config)

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

    def set_expert_locations(self,
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
        # self._store_method_inputs_to_config("set_expert_locations", "locations")
        config = {}
        locs = locals()
        for k in range(self.set_expert_locations.__code__.co_argcount):
            var = self.set_expert_locations.__code__.co_varnames[k]
            if var == "self":
                continue
            else:
                config[var] = locs[var]
        self.config["locations"] = json_serializable(config)


        if file is not None:
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
                      f" current locs size: {len(locs)} -> new locs size: {len(locs) * len(v)}")

            for vv in v:
                _ = locs.copy(True)
                _[k] = vv
                tmp += [_]
            locs = pd.concat(tmp, axis=0)

        # apply column function - to add new columns
        DataLoader.add_cols(locs, col_funcs)

        # (additional) select rows
        if row_select is not None:
            locs = DataLoader.data_select(locs, where=row_select)

        # store rows - e.g. by date?
        if sort_by is not None:
            locs.sort_values(by=sort_by, inplace=True)

        # select a subset of columns
        if keep_cols is not None:
            locs = locs.loc[:, keep_cols]

        return locs

    def load_global_data(self):
        # load global data into memory
        # - local data (for each expert) will be selected from this data
        # store as attribute
        pass

    def select_local_data(self):
        # select subset of global data for a given local expert location
        # return data frame
        pass

    @timer
    def _update_global_data(self,
                            df=None,
                            global_select=None,
                            local_select=None,
                            ref_loc=None,
                            col_funcs=None,
                            prev_where=None):

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
            df = DataLoader.data_select(obj=self.data_source,
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


    # @staticmethod
    def _append_to_store_dict_or_write_to_table(self, ref_loc, save_dict, store_path,
                                                store_dict=None,
                                                store_every=1):
        if store_dict is None:
            store_dict = {}

        assert isinstance(save_dict, dict), f"save_dict must be dict got: {type(save_dict)}"

        # use reference location to change index of tables in save_dict to a multi-index
        # TODO: determine if only want to use coord_col for multi index - to keep things cleaner(?)
        #  - i.e. use: idx_dict = ref_loc[self.coords_col]
        save_dict = DataLoader.make_multiindex_df(idx_dict=ref_loc, **save_dict)

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
                    file=None,
                    param_names=None,
                    ref_loc=None,
                    index_adjust=None,
                    **param_dict):
        # method to load (set) parameters - either from (h5) file, or specified directly
        # via param_dict

        # if file is None - provide param_dict
        if file is None:
            pass
        else:
            # TODO: apply adjustment to location
            if index_adjust is None:
                index_adjust = {}
            # make a copy as can change vlaues
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
        assert isinstance(ref_loc, (pd.Series, dict)), f"ref_loc expected to pd.Series or dict, got: {type(ref_loc)}"

        if isinstance(ref_loc, pd.Series):
            ref_loc = ref_loc.to_dict()

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
        assert self.data_source is not None, "'data_source' is None"
        assert isinstance(self.data_source, (pd.DataFrame, xr.Dataset, xr.DataArray, pd.HDFStore)), \
            f"'data_source' expected to be " \
            f"(pd.DataFrame, xr.Dataset, xr.DataArray, pd.HDFStore), " \
            f"got: {type(self.data_source)}"

        # model
        assert self.model is not None, "'model' is None"
        # TODO: determine why model isinstance check is not working as expected
        #  -
        # assert isinstance(self.model, BaseGPRModel), \
        #     f"'model' expected to be an (inherited) instance of" \
        #     f" BaseGPRModel, got: {type(self.model)}"

        # store path
        assert isinstance(store_path, str), f"store_path expected to be str, got: {type(str)}"

        #
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
        count = 0
        df, prev_where = None, None
        for idx, rl in xprt_locs.iterrows():

            # TODO: use log_lines
            print("-" * 30)
            count += 1
            print(f"{count} / {len(xprt_locs)}")

            # start timer
            t0 = time.time()

            # ----------------------------
            # (update) global data - from data_source (if need be)
            # ----------------------------

            df, prev_where = self._update_global_data(df=df,
                                                      global_select=self.global_select,
                                                      local_select=self.local_select,
                                                      ref_loc=rl,
                                                      prev_where=prev_where,
                                                      col_funcs=self.col_funcs)

            # ----------------------------
            # select local data - relative to expert's location - from global data
            # ----------------------------

            df_local = DataLoader.local_data_select(df,
                                                    reference_location=rl,
                                                    local_select=self.local_select,
                                                    verbose=False)
            print(f"number obs: {len(df_local)}")

            # if there are too few observations store to 'run_details' (so can skip later) and continue
            if len(df_local) < min_obs:
                save_dict = {
                    "run_details": pd.DataFrame({
                        "num_obs": len(df_local),
                        "run_time": np.nan,
                        "mll": np.nan,
                        "optimise_success": False
                    }, index=[0])
                }
                store_dict = self._append_to_store_dict_or_write_to_table(ref_loc=rl,
                                                                          save_dict=save_dict,
                                                                          store_dict=store_dict,
                                                                          store_path=store_path,
                                                                          store_every=store_every)

                continue

            # -----
            # build model - provide with data
            # -----

            # initialise model
            # TODO: needed to review the unpacking of model_params, when won't it work?
            # TODO: rename model instance from gpr_model - to just model (or mdl)
            gpr_model = self.model(data=df_local,
                                   obs_col=self.obs_col,
                                   coords_col=self.coords_col,
                                   **self.model_init_params)

            # ----
            # load parameters (optional)
            # ----

            # TODO: implement this - let them either be fixed or read from file
            if self.model_load_params is not None:
                self.load_params(ref_loc=rl,
                                 model=gpr_model,
                                 **self.model_load_params)

            # --
            # apply constraints
            # --

            # TODO: generalise this to apply any constraints - use apply_param_transform (may require more checks)
            #  - may need information from config, i.e. obj = model.kernel, specify the bijector, other parameters

            if self.constraints is not None:
                if isinstance(self.constraints, dict):
                    print("applying lengthscales contraints")
                    low = self.constraints['lengthscales'].get("low", np.zeros(len(self.coords_col)))
                    high = self.constraints['lengthscales'].get("high", None)
                    gpr_model.set_lengthscale_constraints(low=low, high=high, move_within_tol=True, tol=1e-8, scale=True)
                else:
                    warnings.warn(f"constraints: {self.constraints} are not currently handled!")
            # --
            # optimise parameters
            # --

            # TODO: optimise should be optional
            opt_dets = gpr_model.optimise_parameters()

            # get the hyper parameters - for storing
            hypes = gpr_model.get_parameters()

            # ------
            # make prediction
            # ------

            # TODO: making predictions should be optional
            # TODO: tidy the following up!

            # prediction location(s)
            # TODO: create a method generate prediction locations
            #  - perhaps being relative to some reference location
            prediction_coords = pd.DataFrame(rl).T
            # HACK: just for testing
            # prediction_coords = pd.concat([prediction_coords, prediction_coords], axis=0)
            prediction_coords = prediction_coords[gpr_model.coords_col]

            pred = gpr_model.predict(coords=rl,
                                     full_cov=False)

            # - remove y to avoid conflict with coordinates
            # pop no longer needed?
            # pred.pop('y')

            # remove * from names - causes issues when saving to hdf5 (?)
            # TODO: make this into a private method
            # for k, v in pred.items():
            #     if re.search("\*", k):
            #         pred[re.sub("\*", "s", k)] = pred.pop(k)

            # store data to specified tables according to key
            # - will add mutli-index based on location
            # pred_df = pd.DataFrame(pred, index=np.arange(len(prediction_coords)))
            # pred_df = pd.DataFrame(pred, index=[0])
            # pred_df.rename(columns={c: re.sub("\*", "s", c) for c in pred_df.columns}, inplace=True)

            # add the prediction locations - so will know where prediction was made
            # for c in prediction_coords.columns:
            #     pred_df[f'pred_loc_{c}'] = prediction_coords[c].values

            # add prediction coordinate location
            for c in prediction_coords.columns:
                pred[f'pred_loc_{c}'] = prediction_coords[c].values

            t1 = time.time()

            # ----
            # store results in tables (keys) in hdf file
            # ----

            run_time = t1 - t0

            # device_name = gpr_model.cpu_name if gpr_model.gpu_name is None else gpr_model.gpu_name

            # run details / info - for reference
            run_details = {
                "num_obs": len(df_local),
                "run_time": run_time,
                # "device": device_name,
                "mll": opt_dets['marginal_loglikelihood'],
                "optimise_success": opt_dets['optimise_success']
            }
            # run_details = pd.DataFrame(run_details, index=[0])


            pred = dict_of_array_to_dict_of_dataframe(pred, concat=True)

            dict_of_array_to_dict_of_dataframe(hypes, concat=True)

            save_dict = {
                # "preds": pred_df,
                "run_details": run_details,
                **hypes
            }

            # ---
            # 'store' results
            # ---

            # change index to multi index (using ref_loc)
            # - add to table in store_dict or append to table in store_path if above store_every
            store_dict = self._append_to_store_dict_or_write_to_table(ref_loc=rl,
                                                                      save_dict=save_dict,
                                                                      store_dict=store_dict,
                                                                      store_path=store_path,
                                                                      store_every=store_every)

            t2 = time.time()
            print(f"total run time : {t2 - t0:.2f} seconds")

        # ---
        # store any remaining data
        # ---

        if len(store_dict):
            print("storing final tables")
            store_dict = self._append_to_store_dict_or_write_to_table(ref_loc=rl,
                                                                      save_dict={},
                                                                      store_dict=store_dict,
                                                                      store_path=store_path,
                                                                      store_every=1)


if __name__ == "__main__":

    pass

