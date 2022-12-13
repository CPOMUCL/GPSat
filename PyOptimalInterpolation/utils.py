
import re
import sys
import json
import os
import shutil
import datetime
import subprocess

import pandas as pd
import numpy as np

import scipy.stats as scst
from pyproj import Transformer
from scipy.stats import skew, kurtosis

def get_config_from_sysargv(argv_num=1):
    """read json config from argument location in sys.argv"""
    config = None
    try:
        if bool(re.search('\.json$', sys.argv[argv_num], re.IGNORECASE)):
            print('using input json: %s' % sys.argv[argv_num])
            with open(sys.argv[argv_num]) as f:
                config = json.load(f)
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
            print("-"*10)
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
    except KeyError:
        assert isinstance(col, int), f"col: {col} not a column name, and isn't an integer"
        out = df.iloc[:, col]
    if return_numpy:
        out = out.values
    return out


def config_func(func, source=None, args=None, kwargs=None, col_args=None, col_kwargs=None, df=None, filename_as_arg=False,
                filename=None, col_numpy=True, verbose=False):
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

    quantiles = {f"q{q:.2f}": np.nanquantile(vals, q=q) for q in qs}
    out = {**out, **quantiles}

    columns = None if name is None else [name]
    return pd.DataFrame.from_dict(out, orient='index', columns=columns)


def WGS84toEASE2_New(lon, lat, return_vals="both"):
    valid_return_vals = ['both', 'x', 'y']
    assert return_vals in ['both', 'x', 'y'], f"return_val: {return_vals} is not in valid set: {valid_return_vals}"
    EASE2 = "+proj=laea +lon_0=0 +lat_0=90 +x_0=0 +y_0=0 +ellps=WGS84 +towgs84=0,0,0,0,0,0,0 +units=m +no_defs"
    WGS84 = "+proj=longlat +ellps=WGS84 +datum=WGS84 +no_defs"
    transformer = Transformer.from_crs(WGS84, EASE2)
    x, y = transformer.transform(lon, lat)
    if return_vals == 'both':
        return x, y
    elif return_vals == "x":
        return x
    elif return_vals == "y":
        return y

def EASE2toWGS84_New(x, y, return_vals="both"):
    valid_return_vals = ['both', 'lon', 'lat']
    assert return_vals in ['both', 'lon', 'lat'], f"return_val: {return_vals} is not in valid set: {valid_return_vals}"
    EASE2 = "+proj=laea +lon_0=0 +lat_0=90 +x_0=0 +y_0=0 +ellps=WGS84 +towgs84=0,0,0,0,0,0,0 +units=m +no_defs"
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


def match(x, y):
    """match elements in x to their location in y (taking first occurrence)"""
    # require x,y to be arrays
    x, y = to_array(x, y)
    # NOTE: this can require large amounts of memory if x and y are big
    mask = x[:, None] == y
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
        min_date, max_date = pd.to_datetime(min_date, format=date_col_format), pd.to_datetime(max_date, format=date_col_format)
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

def get_config_from_sysargv(argv_num=1):
    """read json config from argument location"""
    config = {}
    try:
        if bool(re.search('\.json$', sys.argv[argv_num], re.IGNORECASE)):
            print('using input json: %s' % sys.argv[argv_num])
            with open(sys.argv[argv_num]) as f:
                config = json.load(f)
        else:
            print('sys.argv[%s]: %s\n(is not a .json file)\n' % (argv_num, sys.argv[argv_num]))

    except IndexError as e:
        print(f'index error with reading in config with sys.argv:\n{e}')

    return config


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




if __name__ == "__main__":

    pass
