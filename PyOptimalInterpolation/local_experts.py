
import os
import re
import time
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
                 locations=None,
                 data=None,
                 model=None):
        # inputs expected to dicts (or None)

        # local expert locations - DataFrame, should contain coordinates to align with Data
        # self.expert_locs = None

        # # data
        # # - source from which data will be extracted for local expert
        # # - DataFrame, Dataset, or HDFStore
        # self.data_source = None
        # # - where data is read from file system
        # self._data_file = None
        # # - how data is read from file system
        # self._data_engine = None

        # ------
        # Location
        # ------

        # set self.expert_locs
        if locations is None:
            locations = {}
        assert isinstance(locations, dict)
        self.local_expert_locations(**locations)

        # ------
        # Data (source)
        # ------

        if data is None:
            data = {}
        assert isinstance(data, dict)

        # TODO: 'file' should be changed to data_source
        if 'file' in data:
            # set data_source attribute
            self.set_data_source(file=data['file'],
                                 engine=data.get("engine", None))

        # set data related attributes
        # TODO: should these be stored nested in another attribute - use dict?
        for _ in ['obs_col', 'coords_col', 'global_select', 'local_select', 'col_funcs']:
            if _ in data:
                setattr(self, _, data[_])
            else:
                setattr(self, _, None)

        # ------
        # Model
        # ------

        if model is None:
            model = {}
        assert isinstance(model, dict)

        # TODO: change 'model' to 'oi_model' to avoid confusion (?)
        self.model = model.get('model', None)

        # oi_model is a str then expect to be able to import from models
        # TODO: perhaps would like to generalise this a bit more - read models from different modules
        if isinstance(self.model, str):
            self.model = getattr(models, self.model)

        self.model_init_params = model.get("init_params", {})
        self.constraints = model.get("constraints", {})

        # store config
        # taken from answers:
        # https://stackoverflow.com/questions/218616/how-to-get-method-parameter-names
        # config = {}
        # locs = locals()
        # for k in range(self.__init__.__code__.co_argcount):
        #     var = self.__init__.__code__.co_varnames[k]
        #     if isinstance(locs[var], np.ndarray):
        #         config[var] = locs[var].tolist()
        #     elif isinstance(locs[var], (float, int, str, list, tuple, dict)):
        #         config[var] = locs[var]
        #     else:
        #         config[var] = locs[var]



    def set_data_source(self, file, engine=None, verbose=False, **kwargs):

        # TODO: change file to data_source
        # TODO: allow engine to not be case sensitive
        # TODO: allow for files to be handled by DataLoader.read_flat_files()
        #  - i.e. let file be a dict to be unpacked into read_flat_files, set engine = "read_flat_files"
        # TODO: add verbose statements

        # read in or connect to data

        # if engine is None then asdfinfer from file name
        if engine is None:
            # from the beginning (^) match any character (.) zero
            # or more times (*) until last (. - require escape with \)
            file_suffix = re.sub("^.*\.", "", file)

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

        self._data_file = file
        self._data_engine = engine

        self.data_source = None
        # read in via pandas
        if engine in pandas_read_methods:
            self.data_source = getattr(pd, engine)(file, **kwargs)
        # xarray open_dataset
        elif engine in xr_dataset_engine:
            self.data_source = xr.open_dataset(file, engine=engine, **kwargs)
        # or hdfstore
        elif engine == "HDFStore":
            self.data_source = pd.HDFStore(file, mode="r", **kwargs)
        else:
            warnings.warn(f"file: {file} was not read in as\n"
                          f"engine: {engine}\n was not understood. "
                          f"data_source was not set")
            self._data_engine = None


    def local_expert_locations(self,
                               file=None,
                               loc_dims=None,
                               # masks=None,
                               # ref_data=None,
                               add_cols=None,
                               col_funcs=None,
                               keep_cols=None,
                               row_select=None,
                               sort_by=None,
                               verbose=False,
                               **kwargs):

        # TODO: if verbose print what the input parameters are?
        # TODO: allow for dynamically created local expert locations
        #  - e.g. provide grid spacing, mask types (spacing, over ocean only)

        if file is not None:
            if verbose:
                print(f"local_expert_locations - file:\n{file}\nprovided")
            locs = self._read_local_expert_locations_from_file(loc_file=file,
                                                               add_cols=add_cols,
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
                                               add_cols=None,
                                               row_select=None,
                                               col_funcs=None,
                                               sort_by=None,
                                               keep_cols=None,
                                               verbose=False,
                                               **read_csv_kwargs):

        assert os.path.exists(loc_file), f"loc_file:\n{loc_file}\ndoes not exist"
        if verbose:
            print(f"reading in (expert) locations from:\n{loc_file}")
        locs = pd.read_csv(loc_file, **read_csv_kwargs)

        if verbose:
            print(f"number of rows in location DataFrame: {len(locs)}")

        # add columns - repeatedly (e.g. dates)
        if add_cols is None:
            add_cols = {}

        assert isinstance(add_cols, dict), f"add_cols expected to be dict, got: {type(add_cols)}"

        # for each element in add_cols will copy location data
        # TODO: is there a better way of doing this?
        # TODO: add_cols could be confused with DataLoader.add_cols - give different name
        for k, v in add_cols.items():
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


    def run(self,
            model=None,
            data_source=None,
            expert_locs=None,
            store_path=None,
            store_every=10):

        # --------
        # checks
        # --------

        # location
        if expert_locs is None:
            print("expert_locs was not provided / is None, will use 'expert_locs' attribute")
            expert_locs = self.expert_locs
        assert expert_locs is not None, "'expert_locs' is None"
        assert isinstance(expert_locs, pd.DataFrame), f"'expert_locs' expected to be DataFrame, got: {type(expert_locs)}"

        # data_source
        if data_source is None:
            print("data_source was not provided / is None, will use 'data_source' attribute")
            data_source = self.data_source

        assert data_source is not None, "'data_source' is None"
        assert isinstance(data_source, (pd.DataFrame, xr.Dataset, xr.DataArray, pd.HDFStore)), \
            f"'data_source' expected to be " \
            f"(pd.DataFrame, xr.Dataset, xr.DataArray, pd.HDFStore), " \
            f"got: {type(expert_locs)}"

        # model
        if model is None:
            print("model was not provided / is None, will use 'model' attribute")
            model = self.model

        assert model is not None, "'model' is None"
        assert isinstance(model, BaseGPRModel), \
            f"'model' expected to be an (inherited) instance of" \
            f" BaseGPRModel, got: {type(model)}"

        # store path
        assert isinstance(store_path, str), "store_path expected to be "

        print("there's nothing else (method incomplete)")

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
    def _append_to_store_dict_or_write_to_table(ref_loc, save_dict, store_path,
                                                store_dict=None,
                                                store_every=1):
        if store_dict is None:
            store_dict = {}

        assert isinstance(save_dict, dict), f"save_dict must be dict got: {type(save_dict)}"

        # use reference location to change index of tables in save_dict to a multi-index
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
                        store.append(key=k, value=df_tmp, data_columns=True)
                except ValueError as e:
                    print(e)
                except Exception as e:
                    print(e)
            store_dict = {}

        return store_dict

    def run(self, store_path, store_every=10):

        # ---
        # checks on attributes and inputs
        # ---

        assert isinstance(self.expert_locs, pd.DataFrame), \
            f"attr expert_locs is {type(self.expert_locs)}, expected to be DataFrame"

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
            if len(df_local) <= 2:
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
            # set hyper parameters (optional)
            # ----

            # TODO: implement this - let them either be fixed or read from file

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
            opt_dets = gpr_model.optimise_hyperparameters()

            # get the hyper parameters - for storing
            hypes = gpr_model.get_hyperparameters()

            # --
            # make prediction - at the local expert location
            # --

            # TODO: making predictions should be optional
            pred = gpr_model.predict(coords=rl)
            # - remove y to avoid conflict with coordinates
            # pop no longer needed?
            pred.pop('y')

            # remove * from names - causes issues when saving to hdf5 (?)
            # TODO: make this into a private method
            for k, v in pred.items():
                if re.search("\*", k):
                    pred[re.sub("\*", "s", k)] = pred.pop(k)

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

            # store data to specified tables according to key
            # - will add mutli-index based on location
            pred_df = pd.DataFrame(pred, index=[0])
            pred_df.rename(columns={c: re.sub("\*", "s", c) for c in pred_df.columns}, inplace=True)
            save_dict = {
                "preds": pred_df,
                "run_details": pd.DataFrame(run_details, index=[0]),
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


if __name__ == "__main__":


    # ---
    # configuration
    # ---

    import os
    from PyOptimalInterpolation import get_data_path



    oi_config = {
        # "results": {
        #     # "dir":  get_parent_path("results", "sats_ra_cry_processed_arco"),
        #     "dir": get_parent_path("results", "tide_gauge"),
        #     "file": f"oi_bin_{data_source}_{days_ahead}_{int(incl_rad / 1000)}_{ocean_or_lead}_{obs_col}_{grid_size}_{prior_mean}.h5"
        # },
        "input_data": {
            "file_path": get_data_path("example", "ABC.h5"),
            "table": "data",
            "obs_col": "obs",
            "coords_col": ['x', 'y', 't']
        },
        # from either ncdf, zarr or ndf
        "global_select": [
            # {"col": "lat", "comp": ">=", "val": 60}1
        ],
        # how to select data for local expert
        "local_select": [
            {"col": "t", "comp": "<=", "val": 4},
            {"col": "t", "comp": ">=", "val": -4},
            {"col": ["x", "y"], "comp": "<", "val": 300 * 1000}
        ],
        "constraints": {
            # "lengthscales": {
            #     "low": [0, 0, 0],
            #     "high": [2 * incl_rad, 2 * incl_rad, days_ahead + days_behind + 1]
            # }
        },
        "local_expert_locations": {
            # "file": get_data_path("tide_gauge", "arctic_stations_with_loc.csv"),
            "file": get_data_path("tide_gauge", "uhawaii_arctic_station_info_above64_small.csv"),
            # "add_cols": {
            #     "date": oi_dates
            # },
            "col_func_dict": {
                "date": {"func": "lambda x: x.astype('datetime64[D]')", "col_args": "date"},
                "t": {"func": "lambda x: x.astype('datetime64[D]').astype(int)", "col_args": "date"},
                "x": {
                    "source": "PyOptimalInterpolation.utils",
                    "func": "WGS84toEASE2_New",
                    # "col_kwargs": {"lon": "longitude", "lat": "latitude"},
                    "col_kwargs": {"lon": "lon", "lat": "lat"},
                    "kwargs": {"return_vals": "x"}
                },
                "y": {
                    "source": "PyOptimalInterpolation.utils",
                    "func": "WGS84toEASE2_New",
                    # "col_kwargs": {"lon": "longitude", "lat": "latitude"},
                    "col_kwargs": {"lon": "lon", "lat": "lat"},
                    "kwargs": {"return_vals": "y"}
                }
            },
            "keep_cols": ["x", "y", "date", "t"]
        },
        # DEBUGGING: shouldn't skip model params - only skip misc (?)
        # "skip_valid_checks_on": ["local_expert_locations", "misc", "results", "input_data"],
        "skip_valid_checks_on": ["local_expert_locations", "misc"],
        # parameters to provide to model (inherited from BaseGPRModel) when initialising
        "model_params": {
            "coords_scale": [50000, 50000, 1]
        },
        "misc": {
            "store_every": 10,
            # TODO: this should be used in the model_params
            "obs_mean": None
        }
    }

    # ---
    # Parameters
    # ----

    # input data
    input_data = oi_config['input_data']['file_path']

    assert os.path.exists(input_data), \
        f"input_data file:\n{input_data}\ndoes not exist. Create by running: " \
        f"examples/read_and_store_raw_data.py, change input config (configs.example_read_and_store_raw_data.json) " \
        f"as needed"


