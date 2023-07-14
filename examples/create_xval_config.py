# create configuration for cross validation (xval) using a config containing
# - reference configuration
# - xval specifications

import copy
import os
import re
import sys
import warnings
import time
import json

import inspect
from typing import List, Dict, Tuple, Union, Type

import numpy as np
import pandas as pd

from GPSat import get_parent_path, get_data_path
from GPSat.dataloader import DataLoader
from GPSat.utils import get_config_from_sysargv, cprint, json_serializable, \
    _method_inputs_to_config, nested_dict_literal_eval

pd.set_option("display.max_columns", 200)

# ---
# helper functions
# ---

def return_as_list(x):
    if isinstance(x, list):
        return x
    else:
        return [x]

def remove_bad_table_char(x):
    # move bad characters - those that can be used in table name for HDF5
    # TODO: there are probably more
    # TODO: there 00's are a hack for dates
    return re.sub("-| |:|00:00:00", "", x)


class XvalConfig():

    def __init__(self,
                 ref_config: Union[dict, None] = None,
                 xval_config: Union[dict, None] = None):

        # store the inputs as a dict
        self.config = _method_inputs_to_config(locs=locals(),
                                                code_obj=self.__init__.__code__)

        # ---
        # reference configuration
        # ---

        assert ref_config is not None, "ref_config can't be None"
        assert isinstance(ref_config, dict), f"reference config must be a dict, got: {type(ref_config)}"
        self.ref = copy.deepcopy(ref_config)

        # remove any prediction locations from the ref config
        # - they should be specified in xval
        self.ref.pop("pred_loc", None)

        self.data = copy.deepcopy(self.ref['data'])
        self.locations = copy.deepcopy(self.ref['locations'])
        self.model = copy.deepcopy(self.ref['model'])
        self.run_kwargs = copy.deepcopy(self.ref['run_kwargs'])
        self.comment = self.ref.get('comment', "")

        # require in the data config for where to be present, if just an empty list
        if 'where' not in self.data:
            self.data['where'] = []
        if self.data['where'] is None:
            self.data['where'] = []

        # similarly where row_select - these will be added to later for each xval config
        if 'row_select' not in self.data:
            self.data['row_select'] = []
        if self.data['row_select'] is None:
            self.data['row_select'] = []

        # ---
        # cross validation configuration
        # ---

        assert xval_config is not None, "xval_config can't be None"
        assert isinstance(xval_config, dict), f"cross validate (xval) config must be a dict, got: {type(xval_config)}"

        self.xval = copy.deepcopy(xval_config)

        # prep
        self.output_file = self.xval.get("output_file", None)
        # TODO: if not supplied use a sensible default - based on store_path in ref_config.run_kwargs?
        # assert self.output_file is not None, f"output_file can't be None, please supply"

        # get the columns
        assert 'data' in self.ref, "'data' expected to be reference config"

        # get the load_kwargs from 'data'
        # - these will be extended upon when determining the hold out data
        self.load_kwargs = self._get_load_kwargs_from_data(self.ref)

        # prediction locations
        # - can either be fixed locations, or
        assert "pred_loc" in self.xval, "currently xval expected to contain pred_loc"
        self.pred_loc = self.xval["pred_loc"]
        assert isinstance(self.pred_loc, dict)

        self.hold_out_data_is_pred_loc = self.xval.get("hold_out_data_is_pred_loc", True)

        # if the prediction locations are to be the hold out (typical in xval)
        # - then use the same load_kwargs from the data
        # - this will be extended / modified for specific xval data
        if self.hold_out_data_is_pred_loc:
            self.pred_loc['load_kwargs'] = self.load_kwargs

        # columns need to select
        col_select = list(self.xval.get("wheres", {}).keys()) + list(self.xval.get('row_select_values', [])) # list(self.xval['row_select_values'].keys())
        # - want unique values and to preserve the order of the columns
        _ = []
        for c in col_select:
            if c not in _:
                _.append(c)
        self.col_select = _


    @staticmethod
    def _get_load_kwargs_from_data(ref, verbose=False):

        # prep / handle kwargs for DataLoader.load
        load_kwargs = copy.deepcopy(ref['data'])
        load_kwargs['source'] = load_kwargs.pop("data_source")

        load_params = [k for k in inspect.signature(DataLoader.load).parameters]
        for k in list(load_kwargs.keys()):
            if k not in load_params:
                if verbose:
                    print(f"popping from load_kwarg - '{k}' :{load_kwargs.pop(k)}")
                else:
                    load_kwargs.pop(k)

        # add kwargs that will used for xval, if they're not there already
        # - ensure row_select and where exists, set as empty list if missing
        load_kwargs["row_select"] = load_kwargs.get("row_select", [])
        # HACK to deal with row_select being None
        if load_kwargs['row_select'] is None:
            load_kwargs['row_select'] = []
        load_kwargs["where"] = load_kwargs.get("where", [])

        # make sure row_select and where are in a list
        load_kwargs['row_select'] = return_as_list(load_kwargs['row_select'])
        load_kwargs['where'] = return_as_list(load_kwargs['where'])

        return load_kwargs

    @staticmethod
    def _get_where_lists(wheres):

        if wheres is None:
            return [None]
        elif isinstance(wheres, list):
            return wheres
        # otherwise assumed to be a dict
        else:
            midx = pd.MultiIndex.from_product([v for v in wheres.values()],
                                              names=[k for k in wheres.keys()])
            where_df = pd.DataFrame(index=midx).reset_index()

            where_list = []
            for idx, row in where_df.iterrows():
                _ = []
                for k, v in row.to_dict().items():
                    _.append({"col": k, "comp": "==", "val": v})
                where_list.append(_)

        return where_list

    def get_xrs(self):

        where_out = []
        row_select_out = []
        vals_out = []

        # get all the wheres into a list
        where_list = self._get_where_lists(self.xval.get("wheres", None))

        # increment over each where in list
        for w in where_list:

            # copy the (universal) load_kwargs
            lkw = copy.deepcopy(self.load_kwargs)

            # where expected to be a list at this point
            if w is not None:
                if isinstance(w, list):
                    lkw['where'] = lkw['where'] + w
                elif isinstance(w, dict):
                    lkw['where'].append(w)

            # load data
            df = DataLoader.load(**lkw)

            # select the columns needed, drop duplicates
            df = df[self.col_select].drop_duplicates()

            # for each row in df, make a row select function
            func = self.xval['func']

            for idx, row in df.iterrows():
                # full testing of unpacking dict from pd.Series should be checked
                rs = {
                    # "func": func.format(**row.to_dict()),
                    # using the formatted version for regression purposes - remove in future
                    "func": func.format(**self._format_row_value_dict(row.to_dict())),
                }
                # add the column args/kwargs specified in xval
                for _ in ["col_args", "col_kwargs"]:
                    if _ in self.xval:
                        rs[_] = self.xval[_]

                row_select_out.append(rs)
                where_out.append(w)
                vals_out.append(row.to_dict())

        return where_out, row_select_out, vals_out

    @staticmethod
    def _format_row_value_dict(row):
        # this function is used to just align previous implementation,
        # particularly on dealing with date (Timestamp) formats
        # - remove this after regression completed
        out = {}
        for k, v in row.items():

            if isinstance(v, pd._libs.tslibs.timestamps.Timestamp):
                out[k] = str(np.datetime64(v).astype('datetime64[D]'))
            else:
                out[k] = v
        return out

    def make_xval_oi_configs(self,
                             hold_out_data_is_pred_loc=True,
                             add_where_as_col_to_location=True,
                             add_to_table_suffix=True,
                             verbose=True):

        # get list of where, row_selects (and column/value info)
        where_out, row_select_out, vals_out = self.get_xrs()

        # store each xval oi_config in a list
        oic = []

        for i in np.arange(len(where_out)):

            w = where_out[i]
            xrs = copy.deepcopy(row_select_out[i])

            pl = copy.deepcopy(self.pred_loc)
            dc = copy.deepcopy(self.data)
            locs = copy.deepcopy(self.locations)
            m = copy.deepcopy(self.model)
            rkw = copy.deepcopy(self.run_kwargs)

            # add the negated row_select to data
            nxrs = copy.deepcopy(xrs)
            nxrs['negate'] = True
            dc['row_select'] = dc['row_select'] + [nxrs]

            # (optionally) let the prediction locations be the hold out data?
            if hold_out_data_is_pred_loc:
                # if the hold out data is going the prediction locations then the
                # - prediction location method must be from_source, as load_kwargs will be added
                assert pl['method'] == "from_source"
                # if the where condition is None no need to add
                if w is None:
                    pass
                elif isinstance(w, list):
                    pl['load_kwargs']['where'] = pl['load_kwargs']['where'] + w
                else:
                    pl['load_kwargs']['where'] = pl['load_kwargs']['where'] + [w]

                # add the xval (hold out) row select
                pl['load_kwargs']['row_select'] = pl['load_kwargs']['row_select'] + [xrs]

            # (optionally) add where values as column values to location
            # - this is a bit of a HACK
            if add_where_as_col_to_location:
                if isinstance(w, dict):
                    locs["add_data_to_col"] = {w['col']: w['val'] if isinstance(w['val'], list) else [w['val']]}
                # otherwise expect list
                else:
                    locs["add_data_to_col"] = {_['col']:  _['val'] if isinstance(_['val'], list) else [_['val']]
                                               for _ in w}

            # (optionally) append to suffix
            if add_to_table_suffix:
                # new_suffix = "_".join([str(v) for k, v in vals_out[i].items()])
                new_suffix = "_".join([str(vals_out[i][k]) for k in self.col_select])

                rkw['table_suffix'] = remove_bad_table_char(rkw['table_suffix'] + "_" + new_suffix)

                if verbose:
                    print(f"table_suffix: {rkw['table_suffix']}")

            # for regression purposes only, empty where should be ok (double check)
            if dc['where'] == []:
                dc.pop('where')

            # create the xval oi config
            tmp_config = {
                "data": dc,
                "locations": locs,
                "model": m,
                "pred_loc": pl,
                "run_kwargs": rkw,
                "comment": self.comment # modify the comment?
            }
            oic.append(tmp_config)
        return oic


def get_xval_input_config():

    config = get_config_from_sysargv()

    if config is None:

        # get the example config
        config_file = get_parent_path("configs", "example_xval_reference_config.json")
        warnings.warn(f"\nconfig is empty / not provided, will just use an example config:\n{config_file}")
        with open(config_file, "r") as f:
            config = nested_dict_literal_eval(json.load(f))

        # change "/<path>/<to>/<package>" to the path to package on this system
        path_to_package = get_parent_path()
        #
        config['xval']['output_file'] = \
            re.sub("/<path>/<to>/<package>", path_to_package, config['xval']['output_file'])
        config['run_kwargs']['store_path'] = \
            re.sub("/<path>/<to>/<package>", path_to_package, config['run_kwargs']['store_path'])
        config['data']['data_source'] = \
            re.sub("/<path>/<to>/<package>", path_to_package, config['data']['data_source'])

    assert 'xval' in config, "'xval' key expected to be in config when generating oi_config for xval"

    return config


if __name__ == "__main__":

    # ---
    # reference LocalExpertOI config - to be used as basis for config
    # ---

    # TODO: specify reference config as input parameter (file path) from config
    # NOTE: reference config will be used for data, model and location
    config = get_xval_input_config()

    # assert ref_config is not None, f"ref_config is None, issue reading it in? check argument provided to script"

    # ----
    # inline xval_config
    # ----

    # xval_config = {
    #     "output_file": None,
    #     # let the prediction locations be the hold out data?
    #     "hold_out_data_is_pred_loc": True,
    #     # if above is True, pred_loc expected to have method:
    #     "pred_loc": {
    #         "method": "from_source",
    #         "max_dist": 200 * 1000,
    #     },
    #     # selecting using where can save on memory (where is used like a query on data)
    #     "wheres": {
    #         "date": ["2019-12-03"]
    #     },
    #     # unique values of row_select_values (and from wheres) will be used to create row_select funcs
    #     # - see below
    #     "row_select_values": ['track'],
    #     # values in {} will be populated with the unique combinations found in the data
    #     # - these will be used to define the hold out data via a 'row_select'
    #     # - which will be added to the data load_kwargs with a negate=True
    #     # - thus excluding the hold out data from the observations
    #     "func": "lambda x, y: (x == np.datetime64('{date}')) & (y == {track}) ",
    #     # specify the columns to be passed into above row_select func
    #     # "col_kwargs": {"x": "date", "y": "track"},
    #     "col_args": ["date", "track"]
    # }
    # ref_config['xval'] = xval_config

    # ---
    # initialise XvalConfig
    # ---

    # remove the xval key, everything else expected to be part of the reference configuration
    xval_config = config.pop("xval")

    xc = XvalConfig(ref_config=config,
                    xval_config=xval_config)

    # ---
    # create a list of oi_configs (list of dicts)
    # ---

    oi_config_list = xc.make_xval_oi_configs(hold_out_data_is_pred_loc=xc.hold_out_data_is_pred_loc,
                                             add_where_as_col_to_location=xval_config.get("add_where_as_col_to_location", True),
                                             add_to_table_suffix=xval_config.get("add_to_table_suffix", True),
                                             verbose=True)

    # ---
    # write to file
    # ---

    output_file = xc.output_file
    if output_file is None:
        join_cols = "_and_".join([str(c) for c in xc.col_select])
        output_file = re.sub("\.json", f"_XVAL_by_{join_cols}.json", sys.argv[1], re.IGNORECASE)

    # write list of configs to file
    cprint(f"writing to file:\n{output_file}", "OKBLUE")

    with open(output_file, "w") as f:
        json.dump(oi_config_list, f, indent=4)

