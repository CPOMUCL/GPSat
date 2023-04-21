import re
import sys
import json
import os
import shutil
import datetime
import subprocess
import logging
import warnings
import copy

import tables

import pandas as pd
import numpy as np

import scipy.stats as scst

from datetime import datetime as dt
from ast import literal_eval
from functools import reduce
from pyproj import Transformer
from scipy.stats import skew, kurtosis, norm
from typing import Union

from PyOptimalInterpolation.decorators import timer


def nested_dict_literal_eval(d, verbose=False):
    # convert keys that are string as tuple to tuple - could have side affects?
    org_keys = list(d.keys())
    for k in org_keys:
        if re.search("^\(.*\)$", k):
            try:
                k_eval = literal_eval(k)
                if k_eval != k:
                    if verbose:
                        print(f"converting key: {k} (type: {type(k)})")
                        print(f"to key: {k_eval} (type: {type(k_eval)})")
                    d[k_eval] = d.pop(k)
            except ValueError as e:
                print(e)
                print(k)

    out = dict()
    for k in d.keys():
        if isinstance(d[k], dict):
            out[k] = nested_dict_literal_eval(d[k])
        else:
            out[k] = d[k]
    return out


def json_load(file_path):
    # wrapper for json.load, apply nested_dict_literal_eval
    with open(file_path, mode='r') as f:
        config = json.load(f)

    # convert any '(...,...)' keys to tuple: (...,...)
    if isinstance(config, list):
        tmp = []
        for _ in config:
            tmp.append(nested_dict_literal_eval(_))
        config = tmp
    else:
        config = nested_dict_literal_eval(config)

    return config


def get_config_from_sysargv(argv_num=1):
    """read json config from argument location in sys.argv"""
    # TODO: refactor to use argparse package
    config = None
    try:
        if bool(re.search('\.json$', sys.argv[argv_num], re.IGNORECASE)):
            print('using input json: %s' % sys.argv[argv_num])
            config = json_load(sys.argv[argv_num])
        else:
            print('sys.argv[%s]: %s\n(is not a .json file)\n' % (argv_num, sys.argv[argv_num]))

    except IndexError as e:
        print(f'index error with reading in config with sys.argv:\n{e}')

    return config


def move_to_archive(top_dir, file_names=None, suffix="", archive_sub_dir="Archive", verbose=False):
    """
    Move file(s) matching a pattern to an 'archive' directory, adding suffixes if specified

    Parameters
    ----------
    top_dir: str, specifying path to existing directory containing files to archive
    file_names: str or list of str, default None. file names to move to archive
    suffix: str, default "", to be added to file names (before file type) before move to archive
    archive_sub_dir: str, default "Archive". name of sub-directory (in top_dir) to
        use as an Archive dir. Will be created if it does not exist
    Returns
    -------
    None

    """

    assert os.path.exists(top_dir), f"top_dir:\n{top_dir}\ndoes not exist, expecting it to"

    assert file_names is not None, f"file_names not specified"

    # get the archive directory
    adir = os.path.join(top_dir, archive_sub_dir)
    if verbose & (not os.path.exists(adir)):
        print(f"archive directory:\n{adir}\ndoes not exist, will create")
    os.makedirs(adir, exist_ok=True)

    files_in_dir = os.listdir(top_dir)

    # check for files names
    for fn in file_names:
        if verbose:
            print("-" * 10)
        # see if file names exists folder - has to be an exact match
        if fn in files_in_dir:

            _ = os.path.splitext(fn)
            # make a file name for the destination (add suffix before extension)
            fna = "".join([_[0], suffix, _[1]])

            # source and destination files
            src = os.path.join(top_dir, fn)
            dst = os.path.join(adir, fna)

            if verbose:
                print(f"{fn} moving to archive")
                print(f"file name in archive: {fna}")
            # move file
            shutil.move(src, dst)

        else:
            if verbose:
                print(f"{fn} not found")


def get_col_values(df, col, return_numpy=True):
    """get column value, either by column name or index from a dataframe"""
    # get column values using either column name or column index
    try:
        out = df.loc[:, col]
    except KeyError as e:
        print(f"KeyError: {e}\non col: {col} - will try as int")
        assert isinstance(col, int), f"col: {col} not a column name, and isn't an integer"
        out = df.iloc[:, col]
    if return_numpy:
        out = out.values
    return out


def config_func(func, source=None,
                args=None, kwargs=None,
                col_args=None, col_kwargs=None,
                df=None,
                filename_as_arg=False,
                filename=None, col_numpy=True, verbose=False):
    """
    apply a function based on configuration input
    the aim to allow one to apply function, possibly using data from a DataFrame,
    using a specification that can be store in a JSON file

    if df (DataFrame) is provided then can provide input (col_args and/or col_kwargs)
    based on columns of df


    NOTES
    -----
    this function uses eval() so could allow for arbitrary code execution

    Parameters
    ----------
    func: str or function. If str will use eval(func) to convert to function.
        If str and contains "[\|&\=\+\-\*/\%<>]" will create a lambda function: lambda arg1, arg2: eval(f"arg1 {func} arg2")
        If eval(func) raises NameError and source is not None will run f"from {source} import {func}" and try again
        This is to allow import function from a source
    source: str or None, default None. Used to import func from a package.
        e.g. func = "cumprod", source = "numpy"
    args: list or None, default None. If None empty list will be used, i.e. no args will be used
        values will be unpacked and provided to function: e.g. fun(*args, **kwargs)
    kwargs: dict or None, default None. If dict will be unpacked (**kwargs) to provide key word arguments
    col_args: None or list of str, default None. If df (DataFrame) provided can use col_args to specify
        which columns of df will be passed into func as arguments
    col_kwargs: None or dict, default is None.
    df: DataFrame or None, default None
    filename_as_arg: bool, default False. Provide filename as an argument?
    filename: str or None, default None. If filename_as_arg is True then will provide filename as first arg
    col_numpy: bool, default True. If True when extracting columns from DataFrame .values used
    verbose: bool, default False. NOT USED - REMOVE


    Examples
    --------
    # TODO: put proper examples here
    see examples.config_func


    Returns
    -------
    function values, depends on func

    """
    # TODO: apply doc string for config_func - generate function output from a configuration parameters
    # TODO: allow data from column to be pd.Series, instead of np.array (from df[col].values)
    if args is None:
        args = []
    elif not isinstance(args, list):
        args = [args]

    if col_args is None:
        col_args = []
    elif not isinstance(col_args, list):
        col_args = [col_args]

    if kwargs is None:
        kwargs = {}

    if col_kwargs is None:
        col_kwargs = {}

    # assert isinstance(args, list)
    # assert isinstance(col_args, list)
    assert isinstance(kwargs, dict), "kwargs needs to be a dict"
    assert isinstance(col_kwargs, dict), "col_kwargs needs to be a dict"

    if df is None:
        assert len(col_args) == 0, f"df not provide, but col_args: {col_args} were"
        assert len(col_kwargs) == 0, f"df not provide, but col_kwargs: {col_kwargs} were"
    else:
        col_args = [get_col_values(df, col, return_numpy=col_numpy)
                    for col in col_args]
        col_kwargs = {k: get_col_values(df, col, return_numpy=col_numpy)
                      for k, col in col_kwargs.items()}

    # combine args and kwargs
    # - putting col_* first in args
    args = col_args + args

    if filename_as_arg:
        if filename is None:
            print(f"filename_as_arg is: {filename_as_arg} but filename is {filename}, won't add to 'args'")
        else:
            args = [filename] + args

    # - and letting col_kwargs
    for k in col_kwargs.keys():
        if k in kwargs:
            print(f"key: {k} is in kwargs and col_kwargs, values from col_kwargs will be used")
    kwargs = {**col_kwargs, **kwargs}

    # check function
    if isinstance(func, str):
        # operator type function?
        # - check for special characters
        if re.search("^lambda", func):
            fun = eval(func)
        elif re.search("[\|&\=\+\-\*/\%<>]", func):
            # NOTE: using eval can be insecure
            fun = lambda arg1, arg2: eval(f"arg1 {func} arg2")
        else:
            try:
                fun = eval(func)
            except NameError as e:
                # TODO: extend this be able to import from arbitrary file? is that dangerous?
                #  - ref: https://stackoverflow.com/questions/19009932/import-arbitrary-python-source-file-python-3-3#19011259
                assert source is not None, f"NameError occurred on eval({func}), cannot import"
                exec(f"from {source} import {func}")
                fun = eval(func)
    else:
        assert callable(func), f"func provided is not str nor is it callable"
        fun = func

    out = fun(*args, **kwargs)
    if isinstance(out, pd.Series):
        out = out.values
    return out

@timer
def stats_on_vals(vals, measure=None, name=None, qs=None):
    """given a vals (np.array) get a DataFrame of some descriptive stats"""
    out = {}
    out['measure'] = measure
    out['size'] = vals.size
    out['num_not_nan'] = (~np.isnan(vals)).sum()
    out['num_inf'] = np.isinf(vals).sum()
    out['min'] = np.nanmin(vals)
    out['mean'] = np.nanmean(vals)
    out['max'] = np.nanmax(vals)
    out['std'] = np.nanstd(vals)
    out['skew'] = skew(vals[~np.isnan(vals)])
    out['kurtosis'] = kurtosis(vals[~np.isnan(vals)])

    if qs is None:
        qs = [0.05] + np.arange(0.1, 1.0, 0.1).tolist() + [0.95]

    quantiles = {f"q{q:.3f}": np.nanquantile(vals, q=q) for q in qs}
    out = {**out, **quantiles}

    columns = None if name is None else [name]
    return pd.DataFrame.from_dict(out, orient='index', columns=columns)


def WGS84toEASE2_New(lon, lat, return_vals="both", lon_0=0, lat_0=90):
    valid_return_vals = ['both', 'x', 'y']
    assert return_vals in ['both', 'x', 'y'], f"return_val: {return_vals} is not in valid set: {valid_return_vals}"
    EASE2 = f"+proj=laea +lon_0={lon_0} +lat_0={lat_0} +x_0=0 +y_0=0 +ellps=WGS84 +towgs84=0,0,0,0,0,0,0 +units=m +no_defs"
    WGS84 = "+proj=longlat +ellps=WGS84 +datum=WGS84 +no_defs"
    transformer = Transformer.from_crs(WGS84, EASE2)
    x, y = transformer.transform(lon, lat)
    if return_vals == 'both':
        return x, y
    elif return_vals == "x":
        return x
    elif return_vals == "y":
        return y


def EASE2toWGS84_New(x, y, return_vals="both", lon_0=0, lat_0=90):
    valid_return_vals = ['both', 'lon', 'lat']
    assert return_vals in ['both', 'lon', 'lat'], f"return_val: {return_vals} is not in valid set: {valid_return_vals}"
    EASE2 = f"+proj=laea +lon_0={lon_0} +lat_0={lat_0} +x_0=0 +y_0=0 +ellps=WGS84 +towgs84=0,0,0,0,0,0,0 +units=m +no_defs"
    WGS84 = "+proj=longlat +ellps=WGS84 +datum=WGS84 +no_defs"
    transformer = Transformer.from_crs(EASE2, WGS84)
    lon, lat = transformer.transform(x, y)
    if return_vals == "both":
        return lon, lat
    elif return_vals == "lon":
        return lon
    elif return_vals == "lat":
        return lat


def to_array(*args, date_format="%Y-%m-%d"):
    """
    generator to convert arguments to np.ndarray
    """

    for x in args:
        if isinstance(x, datetime.date):
            yield np.array([x.strftime(date_format)], dtype="datetime64[D]")
        # if already an array yield as is
        if isinstance(x, np.ndarray):
            yield x
        elif isinstance(x, (list, tuple)):
            yield np.array(x)
        # TODO: add this back - why was it removed?
        # elif isinstance(x, (pd.Series, pd.core.indexes.base.Index, pd.core.series.Series)):
        #     yield x.values
        elif isinstance(x, (int, float, str, bool, np.bool_)):
            yield np.array([x], dtype=type(x))
        # np.int{#}
        elif isinstance(x, (np.int8, np.int16, np.int32, np.int64)):
            yield np.array([x], dtype=type(x))
        # np.float{#}
        elif isinstance(x, (np.float16, np.float32, np.float64)):
            yield np.array([x], dtype=type(x))
        # np.bool*
        elif isinstance(x, (np.bool, np.bool_, np.bool8)):
            yield np.array([x], dtype=type(x))
        # np.datetime64
        elif isinstance(x, np.datetime64):
            yield np.array([x], "datetime64[D]")
        elif x is None:
            yield np.array([])
        else:
            from warnings import warn
            warn(f"Data type {type(x)} is not configured in to_array.")
            yield np.array([x], dtype=object)


def match(x, y, exact=True, tol=1e-9):
    """match elements in x to their location in y (taking first occurrence)"""
    # require x,y to be arrays
    x, y = to_array(x, y)
    # NOTE: this can require large amounts of memory if x and y are big
    # match exactly?
    if exact:
        mask = x[:, None] == y
    # otherwise check where difference is less than tolerance
    # NOTE: only makes sense with floats (use exact=True for int, str)
    else:
        dif = np.abs(x[:, None] - y)
        mask = dif < tol

    row_mask = mask.any(axis=1)
    assert row_mask.all(), \
        f"{(~row_mask).sum()} not found, uniquely : {np.unique(np.array(x)[~row_mask])}"
    return np.argmax(mask, axis=1)


def bin_obs_by_date(df,
                    val_col,
                    date_col="date",
                    all_dates_in_range=True,
                    x_col='x',
                    y_col='y',
                    grid_res=None,
                    date_col_format="%Y%m%d",
                    x_min=-4500000.0,
                    x_max=4500000.0,
                    y_min=-4500000.0,
                    y_max=4500000.0,
                    n_x=None,
                    n_y=None,
                    bin_statistic='mean',
                    verbose=False):
    """produce a dictionary with keys - date, values - 2d array of binned values"""
    # TODO: double check getting desired results if binning is asymmetrical

    # --
    # checks
    # --

    # columns exist
    for k, v in {"date_col": date_col, "x_col": x_col, "y_col": y_col, "val_col": val_col}.items():
        assert v in df, f"{k}: {v} not in: {df.columns}"

    # number of bins determined by grid_res or (n_x, n_y)
    if grid_res is None:
        print("grid_res not provided, will used n_x, n_y")
        assert (n_x is not None) & (n_y is not None), f"n_x: {n_x} and n_y: {n_y} both need to be not None"
    else:
        print(f"grid_res: {grid_res} will be used, ")
        assert isinstance(grid_res, (int, float)), "grid_res must be int or float"

    # ---
    # number of bins / edges
    # ---

    if grid_res is not None:
        n_x = ((x_max - x_min) / (grid_res * 1000))
        n_y = ((y_max - y_min) / (grid_res * 1000))
        n_x, n_y = int(n_x), int(n_y)

    # n_x, n_y are the number of bins, adding 1 to get the number of edges
    n_x += 1
    n_y += 1

    # --
    # dates
    # --

    udates = df['date'].unique()
    udates = np.sort(udates)
    # get all the dates between first and last?
    # - allows for missing days to be provided with nan values
    if all_dates_in_range:
        if verbose:
            print(f"getting all_dates_in_range. current number of dates: {len(udates)}")
        min_date, max_date = np.min(udates), np.max(udates)
        min_date, max_date = pd.to_datetime(min_date, format=date_col_format), pd.to_datetime(max_date,
                                                                                              format=date_col_format)
        min_date, max_date = np.datetime64(min_date).astype('datetime64[D]'), np.datetime64(max_date).astype(
            'datetime64[D]')
        udates = np.arange(min_date, max_date + np.timedelta64(1, "D"))
        udates = np.array([re.sub("-", "", _) for _ in udates.astype(str)])

        if verbose:
            print(f"new number of dates: {len(udates)}")

    # ----
    # bin data

    # store results in dict
    bvals = {}

    # assert x_col in df, f"x_col: {x_col} is not in df columns: {df.columns}"
    # assert y_col in df, f"y_col: {y_col} is not in df columns: {df.columns}"
    # assert val_col in df, f"val_col: {val_col} is not in df columns: {df.columns}"

    # NOTE: x will be dim 1, y will be dim 0
    x_edge = np.linspace(x_min, x_max, int(n_x))
    y_edge = np.linspace(y_min, y_max, int(n_y))

    # increment over each 'unique' date
    for ud in udates:
        if verbose >= 3:
            print(f"binning data for date: {ud}")
        # get values for the date
        _ = df.loc[df['date'] == ud]

        # if there is some data, bin that values
        if len(_) > 0:
            # extract values
            x_in, y_in, vals = _[x_col].values, _[y_col].values, _[val_col].values

            # apply binning
            binned = scst.binned_statistic_2d(x_in, y_in, vals,
                                              statistic=bin_statistic,
                                              bins=[x_edge,
                                                    y_edge],
                                              range=[[x_min, x_max], [y_min, y_max]])
            # extract binned values
            # - transposing
            bvals[ud] = binned[0].T

        else:
            print(f"there was no data for {ud}, populating with nan")
            bvals[ud] = np.full((n_y - 1, n_x - 1), np.nan)

    # NOTE: x,y are swapped because of the transpose - which is confusing and not needed
    # return bvals, y_edge, x_edge
    return bvals, x_edge, y_edge


def not_nan(x):
    return ~np.isnan(x)


def get_git_information():
    """
    helper function to get current git info
    - will get branch, current commit, last commit message
    - and the current modified file

    Returns
    -------
    dict with keys
        branch: branch name
        commit: current commit
        details: from last commit message
        modified: files modified since last commit, only provided if there were any modified files

    """
    # get current branch
    try:
        branch = subprocess.check_output(["git", "branch", "--show-current"], shell=False)
        branch = branch.decode("utf-8").lstrip().rstrip()
    except Exception as e:
        branches = subprocess.check_output(["git", "branch"], shell=False)
        branches = branches.decode("utf-8").split("\n")
        branches = [b.lstrip().rstrip() for b in branches]
        branch = [re.sub("^\* ", "", b) for b in branches if re.search("^\*", b)][0]

    # remote
    try:
        remote = subprocess.check_output(["git", "remote", "-v"], shell=False)
        remote = remote.decode("utf-8").lstrip().rstrip().split("\n")
        remote = [re.sub("\t", " ", r) for r in remote]
    except Exception as e:
        remote = []

    # current commit hash
    cur_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], shell=False)
    cur_commit = cur_commit.decode("utf-8").lstrip().rstrip()

    # last message
    last_msg = subprocess.check_output(["git", "log", "-1"], shell=False)
    last_msg = last_msg.decode("utf-8").lstrip().rstrip()
    last_msg = last_msg.split("\n")
    last_msg = [lm.lstrip().rstrip() for lm in last_msg]
    last_msg = [lm for lm in last_msg if len(lm) > 0]

    # modified files since last commit
    mod = subprocess.check_output(["git", "status", "-uno"], shell=False)
    mod = mod.decode("utf-8").split("\n")
    mod = [m.lstrip().rstrip() for m in mod]
    # keep only those that begin with mod
    mod = [re.sub("^modified:", "", m).lstrip() for m in mod if re.search("^modified", m)]

    out = {
        "branch": branch,
        "remote": remote,
        "commit": cur_commit,
        "details": last_msg
    }

    # add modified files if there are any
    if len(mod) > 0:
        out["modified"] = mod

    return out


def assign_category_col(val, df, categories=None):
    # _ = pd.Series(val, index=df.index, dtype='category')
    # if categories is not None:
    #     _.cat.add_categories(categories)
    # [val] * len(df) could each up a fair amount of memory if df is big enough
    return pd.Categorical([val] * len(df), categories=categories)


def sparse_true_array(shape, grid_space=1, grid_space_offset=0):
    # get a np.array bool with True values regularly spaced through out (False elsewhere)
    # TODO: move sparse_true_array into an appropriate Class
    # create a ND array of False except along every grid_space points
    # - can be used to select a subset of points from a grid
    # NOTE: first dimension is treated as y dim
    # assert shape is not None, "shape is None, please provide iterable (e.g. list or tuple)"

    # function will return a bool array with dimension equal to shape
    # with False everywhere except for Trues regularly spaced every 'grid_space'
    # the fraction of True will be roughly equal to (1/n)^d where n = grid_space, d = len(shape)

    assert isinstance(grid_space, int)
    assert grid_space > 0

    # list of bool arrays
    idxs = [np.zeros(s, dtype='bool') for s in shape]
    # set regularly spaced values to True
    # TODO: allow for grid_space_offset to be specific to each dimension
    for i in range(len(idxs)):
        _ = np.arange(len(idxs[i]))
        b = (_ % grid_space) == grid_space_offset
        idxs[i][b] = True

    # add dimensions
    for i, s in enumerate(shape):
        tmp = (1,) * i + (s,) + (1,) * (len(shape) - i - 1)
        idxs[i] = idxs[i].reshape(tmp)

    # multiply (broadcast) results together
    # using reduce + lambda to avoid warning from np.prod
    return reduce(lambda x, y: x * y, idxs)


def get_previous_oi_config(store_path, oi_config, skip_valid_checks_on=None):

    # TODO: this should be refactored, configs should be store in table, with one conf per row
    #  - this will reduce the ability to check compatible: only check the most recent?

    if skip_valid_checks_on is None:
        skip_valid_checks_on = []

    # create dataframe entry for the oi_config table
    # - idx may need to be changed
    now = dt.now().strftime("%Y-%m-%d %H:%M:%S")
    idx = 1
    tmp = pd.DataFrame({
        "idx": idx,
        "datetime": now,
        "config": json.dumps(json_serializable(oi_config))}, index=[idx])

    # if the file exists - it is expected to contain a dummy table (oi_config) with oi_config as attr
    if os.path.exists(store_path):
        # TODO: put try/except here
        with pd.HDFStore(store_path, mode='a') as store:
            # prev_oi_config = store.get_storer("oi_config").attrs['oi_config']

            # get the most recent (last row) oi_config from the oi_config table
            # TODO: have a try/except here in case this fails for some reason
            oi_conf_df = store.get("oi_config")
            prev_oi_config = oi_conf_df.iloc[[-1], :]["config"].values[-1]

            # convert from str to dict
            prev_oi_config = nested_dict_literal_eval(json.loads(prev_oi_config))

            # update the idx
            tmp['idx'] = oi_conf_df['idx'].max() + 1

            store.append(key="oi_config",
                         value=tmp,
                         data_columns=["idx", "datetime"],
                         min_itemsize={"config": 50000})

    else:
        # otherwise, add first entry
        with pd.HDFStore(store_path, mode='a') as store:

            # add the current entry
            store.append(key="oi_config",
                         value=tmp,
                         data_columns=["idx", "datetime"],
                         min_itemsize={"config": 50000})

            # for legacy reasons, still write (first) oi_config as an attribute
            try:
                store.get_storer("oi_config").attrs['oi_config'] = oi_config
            except tables.exceptions.HDF5ExtError as e:
                # TODO: log
                print(e)
                oi_config['local_expert_locations']['add_cols'].pop('date')
                store.get_storer("oi_config").attrs['oi_config'] = oi_config
                skip_valid_checks_on += ['local_expert_locations']

            # store.get_storer("raw_data_config").attrs["raw_data_config"] = raw_data_config
            # store.get_storer("oi_config").attrs['input_data_config'] = input_data_config
            prev_oi_config = oi_config

    return prev_oi_config, skip_valid_checks_on


def check_prev_oi_config(prev_oi_config, oi_config, skip_valid_checks_on=None):
    if skip_valid_checks_on is None:
        skip_valid_checks_on = []

    # check configs match (where specified to)
    if prev_oi_config != oi_config:
        # TODO: if didn't match exactly - should the difference be stored / updated ?
        # TODO: change this to a warning
        print("there are differences between the configuration provided and one used previously")
        for k, v in oi_config.items():
            if k in skip_valid_checks_on:
                print(f"skipping: {k}")
            else:
                assert v == prev_oi_config[k], f"config check - key: '{k}' did not match (==), will not proceed"


def log_lines(*args, level="debug"):
    assert level in ["debug", "info", "warning", "error", "critical"]
    for idx, a in enumerate(args):
        # TODO: review the types that can be written to .log files
        if isinstance(a, (str, int, float, dict, tuple, list)):
            print(a)
            getattr(logging, level)(a)
        else:
            print(f"not logging arg [{idx}] - type: {type(a)}")


def json_serializable(d, max_len_df=100):
    # convert a dict to format that can be stored as json (via json.dumps())
    assert isinstance(d, dict), f"input is type: {type(d)}, expect dict"

    out = {}
    for k, v in d.items():
        # if key a tuple - convert to string
        if isinstance(k, tuple):
            # NOTE: to recover tuple use
            # from ast import literal_eval
            # literal_eval(k)
            k = str(k)
        # if value is dict call self
        if isinstance(v, dict):
            out[k] = json_serializable(v)
        # convert nd-array to list (these could be very long!)
        elif isinstance(v, np.ndarray):
            out[k] = v.tolist()
        elif isinstance(v, (pd.DataFrame, pd.Series)):
            if len(v) <= max_len_df:
                out[k] = v.to_dict()
            else:
                print(f"in json_serializable - key: '{k}' has value DataFrame/Series,"
                      f" but is too long: {len(v)} >  {max_len_df}\nstoring as str")
                out[k] = str(v)
        else:
            # check if data JSON serializable
            try:
                json.dumps({k: v})
                out[k] = v
            except (TypeError, OverflowError) as e:
                print(f"in json_serializable - key: '{k}' has value type: {type(v)}, "
                      f"which not JSON serializable, will cast with str")
                out[k] = str(v)

    return out


def array_to_dataframe(x, name, dim_prefix="_dim_", reset_index=False):
    """store array into a DataFrame, adding index / columns specifying location"""
    # if x is single value - store as array
    if isinstance(x, (int, float, bool, str)):
        x = np.array([x])

    assert isinstance(x, np.ndarray), f"for 'x' expected np.ndarray, got: {type(x)}"

    # get the shape of the data
    shape = x.shape
    # create multi index
    dim_names = [f"{dim_prefix}{i}" for i in range(len(shape))]
    midx = pd.MultiIndex.from_product([np.arange(i) for i in shape], names=dim_names)
    # store data in DataFrame
    out = pd.DataFrame(x.flat, index=midx, columns=[name])
    if reset_index:
        out.reset_index(inplace=True)

    return out


def dataframe_to_array(df, val_col, idx_col=None, dropna=True, fill_val=np.nan):
    """
    convert a DataFrame with multi-index to ndarray
    multi-index must be integers - corresponding to location along a dimension
    idx_col can be used to get index locations
    """
    # TODO: allow for idx_col to be True and then to use
    if dropna:
        df = df[[val_col]].dropna()

    # get the dimension - i.e. the index location
    # - expected to integers and for each dimension range from 0,..., n-1

    # TODO: review this bit - copied from DataDict
    if idx_col is None:
        idx = df.index

        # index is MultiIndex?
        if isinstance(idx, pd.core.indexes.multi.MultiIndex):
            dim_names = idx.names
            # get the dims dict from idx.values
            # - convert the index values to a 2-d array, then select values
            idx_vals = np.array(idx.values.tolist())
            dims = {dn: idx_vals[:, i] for i, dn in enumerate(dim_names)}
        # otherwise expect single index
        else:
            # TODO: consider preventing the dim_name from being None, need to allow for a default (idx0?)
            dim_name = idx.names[0]
            dims = {dim_name: idx.values}
    else:
        idx_col = idx_col if isinstance(idx_col, list) else [idx_col]
        assert np.in1d(idx_col, df.columns).all(), \
            f"not all idx_col: {idx_col} are in df.columns: {df.columns.values}"
        dims = {ic: df[ic].values for ic in idx_col}

    # check dims make sense - are integers and have no gaps
    for k, v in dims.items():
        assert v.dtype == int, f"'{k}' dimension dtype expected to int, got: {v.dtype}"
        assert v.min() == 0, f"'{k}'min value in dimension expected to be 0, got: {v.min()}"
        max_diff = np.max(np.diff(np.unique(v)))
        assert max_diff == 1, f""

    # let the shape be defined by the dims
    shape = tuple([len(np.unique(v)) for v in dims.values()])

    # write output to array
    out = np.full(shape, fill_val, dtype=df[val_col].dtype)

    # assign values according to dimension values
    idx = tuple([v for v in dims.values()])
    out[idx] = df[val_col].values

    return out



def dict_of_array_to_dict_of_dataframe(array_dict, concat=False, reset_index=False):
    """
    given a dictionary of arrays convert these to DataFrames with dimension locations in multi-index
    if convert=True arrays with the same number of dimensions will be combined

    """
    out = {}
    for k, v in array_dict.items():

        # if concating results - will do for those with the same number of dimensions (shapes can differ)
        if concat:
            if isinstance(v, (int, float, bool, str)):
                num_dims = 1
            else:
                num_dims = len(v.shape)
            tmp = array_to_dataframe(v, k)
            if num_dims in out:
                out[num_dims].append(tmp)
            else:
                out[num_dims] = [array_to_dataframe(v, k)]
        # otherwise just convert the array to a DataFrame
        else:
            out[k] = array_to_dataframe(v, k)

    if concat:
        _ = {}
        for k, v in out.items():
            _[k] = pd.concat(v, join='outer', axis=1)
        out = _

    if reset_index:
        for k in out.keys():
            out[k] = out[k].reset_index()

    return out


def pandas_to_dict(x):

    if isinstance(x, pd.Series):
        return x.to_dict()
    elif isinstance(x, pd.DataFrame):
        assert len(x) == 1, \
            f"in pandas_to_dict input provided as DataFrame, " \
            f"expected to only have 1 row, shape is: {x.shape}"
        return x.iloc[0, :].to_dict()
    elif isinstance(x, dict):
        return x
    else:
        warnings.warn(f"\npandas_to_dict received object of type: {type(x)}\npassing back as is")
        return x


def grid_2d_flatten(x_range, y_range,
                    grid_res=None,
                    step_size=None,
                    num_step=None,
                    center=True):
    # create grid points defined by x/y ranges, and step size (grid_res

    # TODO: allow this to generalise to n-dims

    x_min, x_max = x_range[0], x_range[1]
    y_min, y_max = y_range[0], y_range[1]

    if grid_res is not None:
        # number of bin (edges)
        n_x = ((x_max - x_min) / grid_res) + 1
        n_y = ((y_max - y_min) / grid_res) + 1
        n_x, n_y = int(n_x), int(n_y)

        # NOTE: x will be dim 1, y will be dim 0
        x_edge = np.linspace(x_min, x_max, int(n_x))
        y_edge = np.linspace(y_min, y_max, int(n_y))
    elif step_size is not None:

        x_edge = np.arange(x_min, x_max + step_size, step_size)
        y_edge = np.arange(y_min, y_max + step_size, step_size)

    elif num_step is not None:
        x_edge = np.linspace(x_min, x_max, num_step)
        y_edge = np.linspace(y_min, y_max, num_step)

    # move from bin edge to bin center
    if center:
        x_, y_ = x_edge[:-1] + np.diff(x_edge) / 2, y_edge[:-1] + np.diff(y_edge) / 2
    else:
        x_, y_ = x_edge, y_edge

    # create a grid of x,y coordinates
    x_grid, y_grid = np.meshgrid(x_, y_)

    # flatten and concat results
    out = np.concatenate([x_grid.flatten()[:, None], y_grid.flatten()[:, None]], axis=1)

    return out


def convert_lon_lat_str(x):
    # example inputs:
    # '74 0.1878 N' , ' 140 0.1198 W'
    # TODO: double check this conversion
    assert isinstance(x, str)
    x = x.lstrip().rstrip()
    deg, min, direction = x.split(" ")
    if direction in ['N', 'S']:
        ns = -1 if direction == 'S' else 1
        deg = float(deg)
        min = float(min) / 60
        out = ns * (deg + min)
    else:
        ns = -1 if direction == 'W' else 1
        deg = float(deg)
        min = float(min) / 60
        out = ns * (deg + min)
    return out


def expand_dict_by_vals(d, expand_keys):
    # recursion function to expand certain keys of dict
    # returns a list of dicts
    assert isinstance(d, dict), "input must be a dict"
    if isinstance(expand_keys, str):
        expand_keys = [expand_keys]

    # assert expand_key in d, f"expand_key: {expand_key} is not in dict: {d.keys()}"
    out = []

    for ek in expand_keys:
        expanded_any = False
        if ek not in d:
            continue
        if isinstance(d[ek], (list, tuple, np.ndarray)):
            expanded_any = True
            # for each value in the key to be expanded
            for v in d[ek]:
                # make a (deep?) copy of dict
                _ = copy.deepcopy(d)
                _[ek] = v
                out += expand_dict_by_vals(_, expand_keys)
            # don't continue with other expanded keys, will pick those up in recursion
            break

    # [recursion termination] if no keys were expanded just return input dict
    if not expanded_any:
        return [d]
    else:
        return out


def pretty_print_class(x):
    # invoke __str__
    out = str(x)
    # remove any leading <class ' and trailing '>
    return re.sub("^<class '|'>$", "", out)


def glue_local_predictions(preds_df: pd.DataFrame,
                           expert_locs_df: pd.DataFrame,
                           sigma: Union[int, float, list]=3
                           ) -> pd.DataFrame:
    """
    Glues overlapping predictions by taking a normalised Gaussian weighted average.
    WARNING: This method only deals with expert locations on a regular grid

    :param preds_df: dataframe of predictions generated from local expert OI
    :param expert_locs_df: dataframe consisting of local expert locations used to perform OI
    :param sigma: standard deviation of Gaussian used to generate the weights

    :return: dataframe consisting of glued predictions (mean and std)
    """
    preds = preds_df.copy(deep=True)
    hx = np.diff(np.sort(expert_locs_df['x'].unique())).min() # Spacing in x direction (assuming equal spacing)
    hy = np.diff(np.sort(expert_locs_df['y'].unique())).min() # Spacing in y direction (assuming equal spacing)
    if isinstance(sigma, (int, float)):
        sigma = [sigma for _ in range(2)]
    # Add a std column
    preds.insert(preds.columns.get_loc("f*_var")+1, "f*_std", np.sqrt(preds["f*_var"]))
    # Compute Gaussian weights
    preds['weights_x'] = norm.pdf(preds['pred_loc_x'], preds['x'], hx/sigma[0])
    preds['weights_y'] = norm.pdf(preds['pred_loc_y'], preds['y'], hy/sigma[1])
    preds['total_weights'] = preds['weights_x'] * preds['weights_y']
    # Multiply predictive mean and std by weights
    preds['f*'] = preds['f*'] * preds['total_weights']
    preds['f*_std'] = preds['f*_std'] * preds['total_weights']
    # Compute weighted sum of mean and std, in addition to the total weights at each location
    glued_preds = preds[['pred_loc_x', 'pred_loc_y',  'total_weights', 'f*', 'f*_std']].groupby(['pred_loc_x', 'pred_loc_y']).sum()
    glued_preds = glued_preds.reset_index()
    # Normalise weighted sums with total weights
    glued_preds['f*'] = glued_preds['f*'] / glued_preds['total_weights']
    glued_preds['f*_std'] = glued_preds['f*_std'] / glued_preds['total_weights']
    return glued_preds.drop("total_weights", axis=1)


def dataframe_to_2d_array(df, x_col, y_col, val_col, tol=1e-9, fill_val=np.nan, dtype=None):
    """
    Extract values from DataFrame to a 2-d array - assumes values came from 2-d array
    requires dimension columns x_col, y_col (do not have to be ordered in DataFrame)
    create a 2-d array of values (val_col)

    the spacing of grid is determined by the smallest step size in the x_col, y_col direction, respectively

    NOTE: this is meant to reverse the process of putting values from regularly spaced grid into a DataFrame
    DO NOT EXPECT THIS TO WORK ON ARBITRARY x,y coordinates

    Parameters
    ----------
    df: DataFrame
    x_col: str, column of df
    y_col: str, column of df
    val_col: str, column of df
    tol: float, default 1e-9, tol value passed to match
    fill_val: float, default np.nan. Default value to populate output array
    dtype: str or None, default None, passed to np.full

    Returns
    -------
    (val_col, x_grid, y_grid) - 2d arrays of values, x,y locations values
    """


    # TODO: this function should be tested
    # TODO: allow

    # check columns
    missing_cols = []
    for k, v in {"x_col": x_col, "y_col": y_col, "val_col": val_col}.items():
        if v not in df:
            missing_cols.append([(k,v)])
    assert len(missing_cols) == 0, f"the following columns are missing from df:\n{missing_cols}"

    # check there is only one value per coordinate
    val_count = pd.pivot_table(df, index=[x_col, y_col], values=val_col, aggfunc='count')
    assert val_count[val_col].max() == 1, "some coordinates have more than one value"

    # - assumes predictions are already on some sort of regular grid!
    unique_x = np.sort(df[x_col].unique())
    unique_y = np.sort(df[y_col].unique())

    # get the smallest step size
    delta_x = np.diff(unique_x).min()
    delta_y = np.diff(unique_y).min()

    # x coordinates / grid
    x_start = unique_x.min()
    x_end = unique_x.max()
    # x_coords = np.arange(x_start, x_end + delta_x, delta_x)
    num_x = np.round((x_end - x_start) / delta_x) + 1
    x_coords = np.linspace(x_start, x_end, int(num_x))

    # y coordinates / grid
    y_start = unique_y.min()
    y_end = unique_y.max()
    # y_coords = np.arange(y_start, y_end + delta_y, delta_y)
    num_y = np.round((y_end - y_start) / delta_y) + 1
    y_coords = np.linspace(y_start, y_end, int(num_y))

    # create a mesh grid
    x_grid, y_grid = np.meshgrid(x_coords, y_coords)

    # get the grid location (index) for each dimension
    # NOTE: this assumes that all values have landed on grid (coords) created
    #  - there might be issues here with float precision
    grid_loc_x = match(df[x_col].values, x_coords, exact=False, tol=tol)
    grid_loc_y = match(df[y_col].values, y_coords, exact=False, tol=tol)

    # check there is only one grid_loc for each point
    # df[['grid_loc_x', 'grid_loc_y']].drop_duplicates().shape[0] == pave.shape[0]

    # create a 2d array to populate
    # TODO: allow dtype, fill value to be determined from df[val_col]?
    val2d = np.full(x_grid.shape, fill_val,  dtype=dtype)
    # populate array
    val2d[grid_loc_y, grid_loc_x] = df[val_col].values

    return val2d, x_grid, y_grid


if __name__ == "__main__":

    # ---
    # put values into 2d array from dataframe

    # integer spacing
    x_coords = np.arange(-10, 5, 2.0)
    # float spacing
    y_coords = np.linspace(-10, 30, 20)
    x_grid, y_grid = np.meshgrid(x_coords, y_coords)

    vals = np.random.normal(0, 1, size=x_grid.shape)

    df = pd.DataFrame({"x": x_grid.flatten(),
                       "y": y_grid.flatten(),
                       "z": vals.flatten()})
    # sort - to show order in DataFrame does not matter
    df.sort_values("z", inplace=True)

    chk, x_chk, y_chk = dataframe_to_2d_array(df, x_col="x", y_col="y", val_col="z")

    # check all values were recovered
    assert np.all(chk == vals)
    assert np.all(x_chk == x_grid)
    assert np.all(y_chk == y_grid)



    # import matplotlib.pyplot as plt
    # create gridded coordinate array
    xy_range = [-4500000.0, 4500000.0]
    X = grid_2d_flatten(xy_range, xy_range, step_size=12.5 * 1000)

    # plt.scatter(X[:, 0], X[:, 1], s=0.1, color='blue')
    # plt.show()


    # basis for unit test
    d = {"a": 1, "b": 2, "c": 3}
    dl = expand_dict_by_vals(d, d.keys())
    assert len(dl) == 1
    assert dl[0] == d

    d = {"a": [1, 2, 3], "b": 2, "c": 3}
    dl = expand_dict_by_vals(d, d.keys())
    assert len(dl) == 3
    for _ in dl:
        assert isinstance(_, dict)
        for k, v in _.items():
            assert not isinstance(v, (list, tuple, np.ndarray))

    d = {"a": [1, 2, 3], "b": ['a', 'b'], "c": 3}
    dl = expand_dict_by_vals(d, d.keys())

    assert len(dl) == 6
    for _ in dl:
        assert isinstance(_, dict)
        for k,v in _.items():
            assert not isinstance(v, (list, tuple, np.ndarray))


