
import re
import sys
import json
import os
import shutil

import pandas as pd
import numpy as np

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


def get_col_values(df, col):
    """get column value, either by column name or index from a dataframe"""
    # get column values using either column name or column index
    try:
        out = df.loc[:, col].values
    except KeyError:
        assert isinstance(col, int), f"col: {col} not a column name, and isn't an integer"
        out = df.iloc[:, col].values
    return out


def config_func(func, args=None, kwargs=None, col_args=None, col_kwargs=None, df=None, filename_as_arg=False,
                filename=None, verbose=False):
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
        col_args = [get_col_values(df, col) for col in col_args]
        col_kwargs = {k: get_col_values(df, col) for k, col in col_kwargs.items()}

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
        if re.search("[\|&\=\+\-\*/\%<>]", func):
            # NOTE: using eval can be insecure
            fun = lambda arg1, arg2: eval(f"arg1 {func} arg2")
        else:
            fun = eval(func)
    else:
        assert callable(func), f"func provided is not str nor is it callable"
        fun = func

    return fun(*args, **kwargs)


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


def WGS84toEASE2_New(lon, lat):
    EASE2 = "+proj=laea +lon_0=0 +lat_0=90 +x_0=0 +y_0=0 +ellps=WGS84 +towgs84=0,0,0,0,0,0,0 +units=m +no_defs"
    WGS84 = "+proj=longlat +ellps=WGS84 +datum=WGS84 +no_defs"
    transformer = Transformer.from_crs(WGS84, EASE2)
    x, y = transformer.transform(lon, lat)
    return x, y


def date_from_datetime(dt):
    # convert a datetime column with format YYYY-MM-DD HH:mm:SS
    # would it be faster use apply on a Series?
    remove_dash_and_time = lambda x: re.sub(" .*$|-", "", x)
    return np.array([remove_dash_and_time(_) for _ in dt])

if __name__ == "__main__":

    pass
