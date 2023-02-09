
import os
import re
import gpflow
import numpy as np
import pandas as pd
import xarray as xr
from abc import ABC, abstractmethod
from typing import List, Dict, Tuple, Union, Type

import warnings

from PyOptimalInterpolation.decorators import timer
from PyOptimalInterpolation.dataloader import DataLoader
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

    def __init__(self):
        # initialise

        # local expert locations - DataFrame, should contain coordinates to align with Data
        self.expert_locs = None

        # data
        # - source from which data will be extracted for local expert
        # - DataFrame, Dataset, or HDFStore
        self.data_source = None
        # - where data is read from file system
        self._data_file = None
        # - how data is read from file system
        self._data_engine = None

    def set_data_source(self, file, engine=None, verbose=False, **kwargs):
        # read in or connect to data
        # TODO: allow engine to not be case sensitive

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
                               verbose=False,
                               **kwargs):

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


