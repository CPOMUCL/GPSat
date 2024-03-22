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
import numba as nb

import scipy.stats as scst
from scipy.spatial.distance import cdist

from datetime import datetime as dt
from ast import literal_eval
from functools import reduce
from pyproj import Transformer
from scipy.stats import skew, kurtosis, norm
from typing import Union
from deprecated import deprecated

from GPSat.decorators import timer

def nested_dict_literal_eval(d, verbose=False):
    """
    Converts a nested dictionary with string keys that represent tuples to a dictionary with tuple keys.

    Parameters
    ----------
    d: dict
        The nested dictionary to be converted.
    verbose: bool, default False
        If True, prints information about the keys being converted.

    Returns
    -------
    dict
        The converted dictionary with tuple keys.

    Raises
    ------
    ValueError: If a string key cannot be evaluated as a tuple.

    Examples
    --------
    >>> d = {'(1, 2)': {'(3, 4)': 5}}
    >>> nested_dict_literal_eval(d)
    {(1, 2): {(3, 4): 5}}

.. note::
    This function modifies the original dictionary in place.

    """
    # TODO: refactor docstring

    # convert keys that are string as tuple to tuple - could have side affects?
    org_keys = list(d.keys())
    for k in org_keys:
        if isinstance(k, str):
            if re.search(r"^\(.*\)$", k):
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
    """
    This function loads a JSON file from the specified file path and
    applies a nested dictionary literal evaluation (nested_dict_literal_eval)
    to convert any string keys in the format of '(...,...)' to tuple keys.

    The resulting dictionary is returned.

    Parameters
    ----------
    file_path: str
        The path to the JSON file to be loaded.

    Returns
    -------
    dict or list of dict
        The loaded JSON file as a dictionary or list of dictionaries.

    Examples
    --------
    Assuming a JSON file named 'config.json' with the following contents:
    {
        "key1": "value1",
         "('key2', 'key3')": "value2",
         "key4": {"('key5', 'key6')": "value3"}
    }

    The following code will load the file and convert the '(key2, key3)' and '(key5, key6)' keys to tuple keys:
    config = json_load('config.json')
    print(config)

    {'key1': 'value1',
     '(key2, key3)': 'value2',
     'key4': {'(key5, key6)': 'value3'}}

    """
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
    """
    This function takes an optional argument ``argv_num`` (default value of ``1``) and
    attempts to read a JSON configuration file from the corresponding index in ``sys.argv``.

    If the file extension is not ``.json``, it prints a message indicating that the file is not a JSON file.

    If an error occurs while reading the file, it prints an error message.

    This function could benefit from refactoring to use the ``argparse`` package instead of manually parsing ``sys.argv``.

    Parameters
    ----------
    argv_num :int, default 1
        The index in ``sys.argv`` to read the configuration file from.

    Returns
    -------
    dict or None
        The configuration data loaded from the JSON file,
        or ``None`` if an error occurred while reading the file.

    """
    # """read json config from argument location in sys.argv"""
    # TODO: refactor to use argparse package
    config = None
    try:
        if bool(re.search(r'\.json$', sys.argv[argv_num], re.IGNORECASE)):
            print('using input json: %s' % sys.argv[argv_num])
            config = json_load(sys.argv[argv_num])
        else:
            cprint('in get_config_from_sysargv: sys.argv[%s]: %s\n(is not a .json file)\n' % (argv_num, sys.argv[argv_num]), c="WARNING")

    except IndexError as e:
        cprint(f'index error occurred when reading in config (JSON) from sys.argv[{argv_num}]:\n{e}')

    return config


def move_to_archive(top_dir, file_names=None, suffix="", archive_sub_dir="Archive", verbose=False):
    """
    Moves specified files from a directory to an archive sub-directory within the same directory.
    Moved files will have a suffix added on before file extension.

    Parameters
    ----------
    top_dir : str
        The path to the directory containing the files to be moved.
    file_names : list of str, default None
        The names of the files to be moved. If not specified, all files in the directory will be moved.
    suffix : str, default "".
        A string to be added to the end of the file name before the extension in the archive directory.
    archive_sub_dir : str, default 'Archive'
        The name of the sub-directory within the top directory where the files will be moved.
    verbose : bool, default is False.
        If True, prints information about the files being moved.

    Raises
    ------
    AssertionError
        If top_dir does not exist or file_names is not specified.

    Returns
    -------
    None
        The function only moves files and does not return anything.

.. note::
    If the archive sub-directory does not exist, it will be created.

    If a file with the same name as the destination file already exists in the archive sub-directory, it will be overwritten.

    Examples
    --------
    Move all files in directory to archive sub-directory:
    >>> move_to_archive("path/to/directory")

    Move specific files to archive sub-directory with a suffix added to the file name:
    >>> move_to_archive("path/to/directory", file_names=["file1.txt", "file2.txt"], suffix="_backup")

    Move specific files to a custom archive sub-directory:
    >>> move_to_archive("path/to/directory", file_names=["file1.txt", "file2.txt"], archive_sub_dir="Old Files")

    """
    # TODO: remove this function if it's not being used?
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
    """
    This function takes in a pandas DataFrame, a column name or index, and a boolean flag indicating whether
    to return the column values as a numpy array or not. It returns the values of the specified column as
    either a pandas Series or a numpy array, depending on the value of the ``return_numpy`` flag.

    If the column is specified by name and it does not exist in the DataFrame, the function will attempt
    to use the column index instead. If the column is specified by index and it is not a valid integer index,
    the function will raise an ``AssertionError``.

    Examples
    --------
    >>> import pandas as pd
    >>> from GPSat.utils import get_col_values
    >>> df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
    >>> col_values = get_col_values(df, 'A')
    >>> print(col_values)
    [1 2 3]

    Parameters
    ----------
    df: pandas DataFrame
        A pandas DataFrame containing data.
    col: str or int
        The name of column to extract data from. If specified as an int n,
        it will extract data from the n-th column.
    return_numpy: bool, default True
        Whether to return as numpy array.

    Returns
    -------
    numpy array
        If ``return_numpy`` is set to ``True``.
    pandas Series
        If ``return_numpy`` is set to ``False``.

    """
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
                filename=None, col_numpy=True):
    """
    Apply a function based on configuration input.

    The aim is to allow one to apply a function, possibly on data from a DataFrame,
    using a specification that can be stored in a JSON configuration file.

.. note::
    - This function uses ``eval()`` so could allow for arbitrary code execution.
    - If DataFrame ``df`` is provided, then can provide input (``col_args`` and/or ``col_kwargs``) \
    based on columns of ``df``.

    Parameters
    ----------
    func: str or callable.

        - If ``str``, it will use ``eval(func)`` to convert it to a function.
        - If it contains one of ``"|"``, ``"&"``, ``"="``, ``"+"``, ``"-"``, ``"*"``, ``"/"``, ``"%"``, ``"<"``, and ``">"``, \
          it will create a lambda function:
        
        .. code-block:: python

            lambda arg1, arg2: eval(f"arg1 {func} arg2")

        - If ``eval(func)`` raises ``NameError`` and ``source`` is not ``None``, it will run
        
        .. code-block:: python

            f"from {source} import {func}"
        
        and try again.
        This is to allow import function from a source.
    source: str or None, default None
        Package name where ``func`` can be found, if applicable. Used to import ``func`` from a package.
        e.g.
        
        >>> GPSat.utils.config_func(func="cumprod", source="numpy", ...)

        calls the function ``cumprod`` from the package ``numpy``.

    args: list or None, default None
        If ``None``, an empty list will be used, i.e. no args will be used.
        The values will be unpacked and provided to ``func``: i.e. ``func(*args, **kwargs)``
    kwargs: dict or None, default None
        If ``dict``, it will be unpacked (``**kwargs``) to provide key word arguments to ``func``.
    col_args: None or list of str, default None
        If DataFrame ``df`` is provided, it can use ``col_args`` to specify
        which columns of ``df`` will be passed into ``func`` as arguments.
    col_kwargs: None or dict, default is None
        Keyword arguments to be passed to ``func`` specified as dict whose keys are parameters of ``func`` and values are
        column names of a DataFrame ``df``. Only applicable if ``df`` is provided.
    df: DataFrame or None, default None
        To provide if one wishes to use columns of a DataFrame as arguments to ``func``.
    filename_as_arg: bool, default False
        Set ``True`` if ``filename`` is used as an argument to ``func``.
    filename: str or None, default None
        If ``filename_as_arg`` is ``True``, then will provide ``filename`` as first arg.
    col_numpy: bool, default True
        If ``True``, when extracting columns from DataFrame, ``.values`` is used to convert to numpy array.


    Examples
    --------

    >>> import pandas as pd
    >>> from GPSat.utils import config_func
    >>> config_func(func="lambda x, y: x + y", args=[1, 1]) # Computes 1 + 1
    2
    >>> config_func(func="==", args=[1, 1]) # Computes 1 == 1
    True

    Using columns of a DataFrame as inputs:

    >>> df = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})
    >>> config_func(func="lambda x, y: x + y", df=df, col_args=["A", "B"]) # Computes df["A"] + df["B"]
    array([5, 7, 9])
    >>> config_func(func="<=", col_args=["A", "B"], df=df) # Computes df["A"] <= df["B"]
    array([ True,  True,  True])

    We can also use functions from an external package by specifying ``source``. For example,
    the below reproduces the last example in `numpy.cumprod <https://numpy.org/doc/stable/reference/generated/numpy.cumprod.html>`_:

    >>> df = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})
    >>> config_func(func="cumprod", source="numpy", df=df, kwargs={"axis": 0}, col_args=[["A", "B"]])
    array([[  1,   4],
           [  2,  20],
           [  6, 120]])

    Returns
    -------
    any
        Values returned by applying ``func`` on data. The type depends on ``func``.

    Raises
    ------
    AssertionError
        If ``kwargs`` is not a dict, ``col_kwargs`` is not a dict, or ``func`` is not a string or callable.
    AssertionError
        If ``df`` is not provided but ``col_args`` or ``col_kwargs`` are.
    AssertionError
        If ``func`` is a string and cannot be imported on it's own and ``source`` is ``None``.

    """
    # TODO: review use of eval - should it be limited? removed? documented?
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
        elif re.search(r"[\|&\=\+\-\*/\%<>]", func):
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
    """
    This function calculates various statistics on a given array of values.

    Parameters
    ----------
    vals: array-like
        The input array of values.
    measure: str or None, default is None
        The name of the measure being calculated.
    name: str or None, default is None
        The name of the column in the output dataframe. Default is None.
    qs: list or None, defualt None
        A list of quantiles to calculate. If None then will use
        [0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99].

    Returns
    -------
    pd.DataFrame
        containing the following statistics:
        - measure: The name of the measure being calculated.
        - size: The number of elements in the input array.
        - num_not_nan: The number of non-NaN elements in the input array.
        - num_inf: The number of infinite elements in the input array.
        - min: The minimum value in the input array.
        - mean: The mean value of the input array.
        - max: The maximum value in the input array.
        - std: The standard deviation of the input array.
        - skew: The skewness of the input array.
        - kurtosis: The kurtosis of the input array.
        - qX: The Xth quantile of the input array, where X is the value in the qs parameter.

.. note::
    The function also includes a timer decorator that calculates the time taken to execute the function.

    """
    # """given a vals (np.array) get a DataFrame of some descriptive stats"""
    out = {}
    out['measure'] = measure if measure is not None else name
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
        qs = [0.01, 0.05] + np.arange(0.1, 1.0, 0.1).tolist() + [0.95, 0.99]

    quantiles = {f"q{q:.3f}": np.nanquantile(vals, q=q) for q in qs}
    out = {**out, **quantiles}

    columns = None if name is None else [name]
    return pd.DataFrame.from_dict(out, orient='index', columns=columns)


@deprecated(reason="This function will be removed in future versions. Use `WGS84toEASE2` instead.")
def WGS84toEASE2_New(*args, **kwargs):
    return WGS84toEASE2(*args, **kwargs)


@deprecated(reason="This function will be removed in future versions. Use `EASE2toWGS84` instead.")
def EASE2toWGS84_New(*args, **kwargs):
    return EASE2toWGS84(*args, **kwargs)


def WGS84toEASE2(lon, lat, return_vals="both", lon_0=0, lat_0=90):
    """
    Converts WGS84 longitude and latitude coordinates to EASE2 grid coordinates.

    Parameters
    ----------
    lon : float
        Longitude coordinate in decimal degrees.
    lat : float
        Latitude coordinate in decimal degrees.
    return_vals : str, optional
        Determines what values to return. Valid options are ``"both"`` (default), ``"x"``, or ``"y"``.
    lon_0 : float, optional
        Longitude of the center of the EASE2 grid in decimal degrees. Default is ``0``.
    lat_0 : float, optional
        Latitude of the center of the EASE2 grid in decimal degrees. Default is ``90``.

    Returns
    -------
    float
        If ``return_vals`` is ``"x"``. Returns the x EASE2 grid coordinate in meters.
    float
        If ``return_vals`` is ``"y"``. Returns the y EASE2 grid coordinate in meters
    tuple of float
        If ``return_vals`` is ``"both"``. Returns a tuple of (x, y) EASE2 grid coordinates in meters.

    Raises
    ------
    AssertionError
        If ``return_vals`` is not one of the valid options.

    Examples
    --------
    >>> WGS84toEASE2(-105.01621, 39.57422)
    (-5254767.014984061, 1409604.1043472202)

    """

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


def EASE2toWGS84(x, y, return_vals="both", lon_0=0, lat_0=90):
    """
    Converts EASE2 grid coordinates to WGS84 longitude and latitude coordinates.

    Parameters
    ----------
        x: float
            EASE2 grid x-coordinate in meters.
        y: float
            EASE2 grid y-coordinate in meters.
        return_vals: str, optional
            Determines what values to return. Valid options are ``"both"`` (default), ``"lon"``, or ``"lat"``.
        lon_0: float, optional
            Longitude of the center of the EASE2 grid in degrees. Default is ``0``.
        lat_0: float, optional
            Latitude of the center of the EASE2 grid in degrees. Default is ``90``.

    Returns
    -------
    tuple or float
        Depending on the value of ``return_vals``, either a tuple of WGS84 longitude and latitude coordinates (both floats),
        or a single float representing either the longitude or latitude.

    Raises
    ------
    AssertionError
        If ``return_vals`` is not one of the valid options.

    Examples
    --------
    >>> EASE2toWGS84(1000000, 2000000)
    (153.434948822922, 69.86894542225777)

    """

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
    Converts input arguments to numpy arrays.

    Parameters
    ----------
    *args : tuple
        Input arguments to be converted to numpy arrays.
    date_format : str, optional
        Date format to be used when converting datetime.date objects to numpy arrays.

    Returns
    -------
    generator
        A generator that yields numpy arrays.

.. note::
    This function converts input arguments to numpy arrays. If the input argument is already a numpy array, it is yielded as is. If the input argument is a list or tuple, it is converted to a numpy array and yielded. If the input argument is an integer, float, string, boolean, or numpy boolean, it is converted to a numpy array and yielded. If the input argument is a numpy integer or float, it is converted to a numpy array and yielded. If the input argument is a datetime.date object, it is converted to a numpy array using the specified date format and yielded. If the input argument is a numpy datetime64 object, it is yielded as is. If the input argument is None, an empty numpy array is yielded. If the input argument is of any other data type, a warning is issued and the input argument is converted to a numpy array of type object and yielded.

    Examples
    --------
    >>> import datetime
    >>> import numpy as np
    >>> x = [1, 2, 3]

    since function returns are generator, get values out with next

    >>> print(next(to_array(x)))
    [1 2 3]

    or, for a single array like object, can assign with

    >>> c, =  to_array(x)

    >>> y = np.array([4, 5, 6])
    >>> z = datetime.date(2021, 1, 1)
    >>> for arr in to_array(x, y, z):
    ...     print(f"arr type: {type(arr)}, values: {arr}")
    arr type: <class 'numpy.ndarray'>, values: [1 2 3]
    arr type: <class 'numpy.ndarray'>, values: [4 5 6]
    arr type: <class 'numpy.ndarray'>, values: ['2021-01-01']

    """

    for x in args:
        if isinstance(x, datetime.date):
            yield np.array([x.strftime(date_format)], dtype="datetime64[D]")
        # if already an array yield as is
        if isinstance(x, np.ndarray):
            yield x
        elif isinstance(x, (list, tuple)):
            yield np.array(x)
        elif isinstance(x, (pd.Series, pd.core.indexes.base.Index, pd.core.series.Series)):
            yield x.values
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
    """
    This function takes two arrays, x and y, and returns an array of indices indicating
    where the elements of x match the elements of y. Can match exactly or within a specified tolerance.

    Parameters
    ----------
    x: array-like
        the first array to be matched. If not an array will convert via to_array.
    y: array-like
        the second array to be matched against. If not an array will convert via to_array.
    exact: bool, default=True.
        If True, the function matches exactly.
        If False, the function matches within a specified tolerance.
    tol: float, optional, default=1e-9.
        The tolerance used for matching when exact=False.

    Returns
    -------
    indices: array
        the indices of the matching elements in y for each element in x.

    Raises
    ------
    AssertionError: if any element in x is not found in y or if multiple matches are found for any element in x.

.. note::
    This function requires x and y to be arrays or can be converted by to_array
    If exact=False, the function only makes sense with floats. Use exact=True for int and str.
    If both x and y are large, with lengths n and m, this function can take up alot of memory
    as an intermediate bool array of size nxm is created.
    If there are multiple matches of x in y the index of the first match is return
    """
    # TODO: determine how well this stacks up?
    # np.array([np.argmin(np.abs(y - xi)) if not exact else np.argwhere(np.abs(y - xi) <= tol) for xi in x])

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
    """
    This function takes in a pandas DataFrame and bins the data based on the values in a specified column and the x and y coordinates in
    other specified columns. The data is binned based on a grid with a specified resolution or number of bins. The function returns a
    dictionary of binned values for each unique date in the DataFrame.

    Parameters
    ----------
    df: pandas DataFrame
        A DataFrame containing the data to be binned.
    val_col: string
        Name of the column containing the values to be binned.
    date_col: string, default "date"
        Name of the column containing the dates for which to bin the data.
    all_dates_in_range: boolean, default True
        Whether to include all dates in the range of the DataFrame.
    x_col: string, default "x"
        Name of the column containing the x coordinates.
    y_col: string, default "y"
        Name of the column containing the y coordinates.
    grid_res: float or int, default None
        Resolution of the grid in kilometers. If ``None``, then ``n_x`` and ``n_y`` must be specified.
    date_col_format: string, default "%Y%m%d"
        Format of the date column.
    x_min: float, default -4500000.0
        Minimum x value for the grid.
    x_max: float, default 4500000.0
        Maximum x value for the grid.
    y_min: float, default -4500000.0
        Minimum y value for the grid.
    y_max: float, default 4500000.0
        Maximum y value for the grid.
    n_x: int, default None
        Number of bins in the x direction.
    n_y: int, default None
        Number of bins in the y direction.
    bin_statistic: string or callable, default "mean"
        Statistic to compute in each bin.
    verbose: boolean, default False
        Whether to print additional information during execution.

    Returns
    -------
    bvals: dictionary
        The binned values for each unique date in the DataFrame.
    x_edge: numpy array
        x values for the edges of the bins.
    y_edge: numpy array
        y values for the edges of the bins.

    Notes
    -----
    The x and y coordinates are swapped in the returned binned values due to the transpose operation used in the function.

    """
    # TODO: review usage - this function looks like it can be removed
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
    This function retrieves information about the current state of a Git repository.

    Returns
    -------
    dict
        Contains the following keys:

        - ``"branch"``: the name of the current branch.
        - ``"remote"``: a list of strings representing the remote repositories and their URLs.
        - ``"commit"``: the hash of the current commit.
        - ``"details"``: a list of strings representing the details of the last commit (author, date, message).
        - ``"modified"`` (optional): a list of strings representing the files modified since the last commit.

.. note::
    - If the current branch cannot be determined, the function will attempt to retrieve it from the list of all branches.
    - If there are no remote repositories, the ``"remote"`` key will be an empty list.
    - If there are no modified files, the ``"modified"`` key will not be present in the output.
    - This function requires the Git command line tool to be installed and accessible from the command line.

    """
    # get current branch
    try:
        branch = subprocess.check_output(["git", "branch", "--show-current"], shell=False)
        branch = branch.decode("utf-8").lstrip().rstrip()
    except Exception as e:
        branches = subprocess.check_output(["git", "branch"], shell=False)
        branches = branches.decode("utf-8").split("\n")
        branches = [b.lstrip().rstrip() for b in branches]
        branch = [re.sub(r"^\* ", "", b) for b in branches if re.search("^\*", b)][0]

    # remote
    try:
        remote = subprocess.check_output(["git", "remote", "-v"], shell=False)
        remote = remote.decode("utf-8").lstrip().rstrip().split("\n")
        remote = [re.sub(r"\t", " ", r) for r in remote]
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
    """
    Generate categorical ``pd.Series`` equal in length to a reference DataFrame (``df``)

    Parameters
    ----------
    val : str
        The value to assign to the categorical Series.
    df : pandas DataFrame
        reference DataFrame, used to determine length of output
    categories : list, optional
        A list of categories to be used for the categorical column.

    Returns
    -------
    pandas Categorical Series
        A categorical column with the assigned value and specified categories (if provided).

    Notes
    -----
    This function creates a new categorical column in the DataFrame with the specified value and categories. If categories are not provided,
    they will be inferred from the data. The function returns a pandas Categorical object representing the new column.

    Examples
    --------
    >>> import pandas as pd
    >>> df = pd.DataFrame({'A': [1, 2, 3], 'B': ['a', 'b', 'c']})
    >>> x_series = assign_category_col('x', df)
    """
    # TODO: this function was originally used with read_and_store.py, to try to save space
    #  - when saving to hdf5. it may not longer be useful and perhaps could be removed
    return pd.Categorical([val] * len(df), categories=categories)


def sparse_true_array(shape, grid_space=1, grid_space_offset=0):
    """
    Create a boolean numpy array with True values regularly spaced throughout, and False elsewhere.

    Parameters
    ----------
    shape: iterable (e.g. list or tuple)
        representing the shape of the output array.
    grid_space: int, default 1
        representing the spacing between True values.
    grid_space_offset: int, default 0
        representing the offset of the first True value in each dimension.

    Returns
    -------
    np.array
        A boolean array with dimension equal to shape,
        with False everywhere except for Trues regularly spaced every 'grid_space'.
        The fraction of True will be roughly equal to (1/n)^d where n = grid_space, d = len(shape).

.. note::
    The first dimension is treated as the y dimension.
    This function will return a bool array with dimension equal to shape with False everywhere except for Trues regularly spaced every 'grid_space'.
    The fraction of True will be roughly equal to (1/n)^d where n = grid_space, d = len(shape).
    The function allows for grid_space_offset to be specific to each dimension.

    """

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


def get_previous_oi_config(store_path, oi_config, table_name="oi_config", skip_valid_checks_on=None):
    """
    This function retrieves the previous configuration from optimal interpolation (OI) results file (``store_path``)

    If the ``store_path`` exists, it is expected to contain a table called "oi_config"
    with the previous configurations stored as rows.

    If ``store_path`` does not exist, the function creates the file and
    adds the current configuration (``oi_config``) as the first row in "oi_config" table.

    Each row in the "oi_config" table contains columns 'idx' (index), 'datetime' and 'config'.
    The values in the 'config' are provided ``oi_config`` (dict) converted to str.

    If the table (``oi_config``) already exists, the function will match the provide ``oi_config``
    against the previous config values, if any match exactly the largest config id will be returned.
    Otherwise (``oi_config`` does **not** exactly match any previous config) then the largest idx value will be
    increment and returned.

    Parameters
    ----------
    store_path: str
        The file path where the configurations are stored.
    oi_config: dict
        Representing the current configuration for the OI system.
    table_name: str, default "oi_config"
        The table where the configurations will be store.
    skip_valid_checks_on: list of str or None, default None
        If list the names of the configuration keys that should be skipped during validation checks.
        **Note:** validation checks are not done in this function.

    Returns
    -------
    dict
        Previous configuration as a dictionary.
    list
        List of configuration keys to skipped during validation checks.
    int
        Configuration ID.


    """
    # TODO: update doc string
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

    table_exists = False
    if os.path.exists(store_path):
        with pd.HDFStore(store_path, mode='r') as store:
            if table_name in store:
                table_exists = True

    # if the file exists - it is expected to contain a dummy table (oi_config) with oi_config as attr
    if table_exists:
        # TODO: put try/except here
        with pd.HDFStore(store_path, mode='a') as store:
            # prev_oi_config = store.get_storer("oi_config").attrs['oi_config']

            # get the most recent (last row) oi_config from the oi_config table
            # TODO: have a try/except here in case this fails for some reason
            oi_conf_df = store.get(table_name)
            # prev_oi_config = oi_conf_df.iloc[[-1], :]["config"].values[-1]

            # get dict of ALL previous configs, with key being idx
            prev_oi_configs = {}
            for _, row in oi_conf_df.iterrows():
                prev_oi_configs[row['idx']] = nested_dict_literal_eval(json.loads(row['config']))

            # determine if current config matches any of the previous configs
            # - will use EXACT matching
            match_config = []
            for k, v in prev_oi_configs.items():
                if oi_config == v:
                    match_config.append(k)

            # if matched a previous config - will use that config_id
            if len(match_config):
                # set the config id (so can be returned)
                match_config_id = max(match_config)
                cprint(f"current config matched previous - idx: {match_config_id}", c="OKBLUE")
                tmp['idx'] = match_config_id
                prev_oi_config = prev_oi_configs[match_config_id]

            # else (current config does not EXACTLY match any previous)
            else:
                # otherwise take previous config as the last to be run
                # - is will be used to check_config_compatible, which might make not complete sense
                prev_oi_config = prev_oi_configs

                # increment the index and write to table
                tmp['idx'] = max(prev_oi_configs) + 1

                store.append(key=table_name,
                             value=tmp,
                             index=False,
                             data_columns=["idx", "datetime"],
                             min_itemsize={"config": 50000})

    else:
        # otherwise, add first entry
        with pd.HDFStore(store_path, mode='a') as store:

            # add the current entry
            store.append(key=table_name,
                         value=tmp,
                         index=False,
                         data_columns=["idx", "datetime"],
                         min_itemsize={"config": 50000})

            # for legacy reasons, still write (first) oi_config as an attribute
            try:
                store.get_storer(table_name).attrs['oi_config'] = oi_config
            except tables.exceptions.HDF5ExtError as e:
                # TODO: log
                print(e)
                oi_config['local_expert_locations']['add_cols'].pop('date')
                store.get_storer(table_name).attrs['oi_config'] = oi_config
                skip_valid_checks_on += ['local_expert_locations']

            # store.get_storer("raw_data_config").attrs["raw_data_config"] = raw_data_config
            # store.get_storer("oi_config").attrs['input_data_config'] = input_data_config
            prev_oi_config = oi_config

    #
    config_id = int(tmp['idx'].values[0])

    return prev_oi_config, skip_valid_checks_on, config_id


def check_prev_oi_config(prev_oi_config, oi_config, skip_valid_checks_on=None):
    """

    This function checks if the previous configuration matches the current one.
    It takes in two dictionaries, ``prev_oi_config`` and ``oi_config``,
    which represent the previous and current configurations respectively.

    The function also takes an optional list ``skip_valid_checks_on``, which contains keys that
    should be skipped during the comparison.

    Notes
    -----
    - If ``skip_valid_checks_on`` is not provided, it defaults to an empty list. \
      The function then compares the two configurations and \
      raises an ``AssertionError`` if any key-value pairs do not match.
    - If the configurations do not match exactly, an ``AssertionError`` is raised.
    - This function assumes that the configurations are represented as dictionaries and \
      that the keys in both dictionaries are the same.

    Parameters
    ----------
    prev_oi_config: dict
        Previous configuration to be compared against.
    oi_config: dict
        Current configuration to compare against ``prev_oi_config``.
    skip_valid_checks_on: list or None, default None
        If not ``None``, should be a list of keys to **not** check.

    Returns
    -------
    None

    """

    if skip_valid_checks_on is None:
        skip_valid_checks_on = []

    # check configs match (where specified to)
    if prev_oi_config != oi_config:
        # TODO: if didn't match exactly - should the difference be stored / updated ?
        # TODO: change this to a warning
        print("there are differences between the configuration provided and one used previously")
        bad_keys = []
        for k, v in oi_config.items():
            if k in skip_valid_checks_on:
                print(f"skipping: {k}")
            else:
                if v != prev_oi_config[k]:
                    bad_keys.append(k)
                # assert v == prev_oi_config[k], f"config check - key: '{k}' did not match (==), will not proceed"

        assert len(bad_keys), f"the following keys did not have values that matched exactly: {bad_keys}"

def log_lines(*args, level="debug"):
    """
    This function logs lines to a file with a specified logging level.

    This function takes in any number of arguments and a logging level.

    The function checks that the logging level is valid and then iterates through the arguments.

    If an argument is a string, integer, float, dictionary, tuple, or list, it is printed and
    logged with the specified logging level.

    If an argument is not one of these types, it is not logged and a message is printed indicating
    the argument's type.

    Parameters
    ----------
    *args: tuple
        arguments to be provided to logging using the method specified by level
    level: str, default "debug"
        must be one of ["debug", "info", "warning", "error", "critical"]
        each argument provided is logged with getattr(logging, level)(arg)

    Returns
    -------
    None

    """
    assert level in ["debug", "info", "warning", "error", "critical"]
    for idx, a in enumerate(args):
        # TODO: review the types that can be written to .log files
        if isinstance(a, (str, int, float, dict, tuple, list)):
            print(a)
            getattr(logging, level)(a)
        else:
            print(f"not logging arg [{idx}] - type: {type(a)}")


def json_serializable(d, max_len_df=100):
    """
    Converts a dictionary to a format that can be stored as JSON via the `json.dumps()` method.

    Parameters
    ----------
        d :dict
            The dictionary to be converted.
        max_len_df: int, default 100
            The maximum length of a Pandas DataFrame or Series that can be converted to a string representation.
            If the length of the DataFrame or Series is greater than this value, it will be stored as a string. Defaults to 100.

    Returns
    -------
        dict
            The converted dictionary.

    Raises
    ------
        AssertionError: If the input is not a dictionary.

    Notes
    -----
        - If a key in the dictionary is a tuple, it will be converted to a string.
        To recover the original tuple, use nested_dict_literal_eval.
        - If a value in the dictionary is a dictionary, the function will be called recursively to convert it.
        - If a value in the dictionary is a NumPy array, it will be converted to a list.
        - If a value in the dictionary is a Pandas DataFrame or Series,
        it will be converted to a dictionary and the function will be called recursively to convert it
        if its length is less than or equal to `max_len_df`. Otherwise, it will be stored as a string.
        - If a value in the dictionary is not JSON serializable, it will be cast as a string.


    """
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
                out[k] = json_serializable(v.to_dict(), max_len_df=max_len_df)
            else:
                cprint(f"in json_serializable - key: '{k}' has value DataFrame/Series,"
                      f" but is too long: {len(v)} >  {max_len_df}\nstoring as str", "WARNING")
                out[k] = str(v)
        else:
            # check if data JSON serializable
            try:
                json.dumps({k: v})
                out[k] = v
            except (TypeError, OverflowError) as e:
                cprint(f"in json_serializable - key: '{k}' has value type: {type(v)}, "
                      f"which not JSON serializable, will cast with str", "WARNING")
                out[k] = str(v)

    return out


def array_to_dataframe(x, name, dim_prefix="_dim_", reset_index=False):
    """
    Converts a numpy array to a pandas DataFrame with a multi-index based on the array's dimensions.

    (Also see :func:`dataframe_to_array <GPSat.utils.dataframe_to_array>`)

    Parameters
    ----------
    x : np.ndarray
        The numpy array to be converted to a DataFrame.
    name : str
        The name of the column in the resulting DataFrame.
    dim_prefix : str, optional
        The prefix to be used for the dimension names in the multi-index. Default is ``"_dim_"``.
        Integers will be appended to ``dim_prefix`` for each dimension of ``x``, i.e.
        if ``x`` is 2d, it will have dimension names ``"_dim_0"``, ``"_dim_1"``, assuming default ``dim_prefix`` is used.
    reset_index : bool, optional
        Whether to reset the index of the resulting DataFrame. Default is ``False``.

    Returns
    -------
    out : pd.DataFrame
        The resulting DataFrame with a multi-index based on the dimensions of the input array.

    Raises
    ------
    AssertionError
        If the input is not a numpy array.

    Examples
    --------
    >>> # express a 2d numpy array in DataFrame
    >>> x = np.array([[1, 2], [3, 4]])
    >>> array_to_dataframe(x, "data")
                    data
    _dim_0 _dim_1
    0      0        1
           1        2
    1      0        3
           1        4
    """

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
    Converts a pandas DataFrame to a numpy array, where the DataFrame has columns that represent dimensions
    of the array and the DataFrame rows represent values in the array.

    Parameters
    ----------
    df : pandas DataFrame
        The DataFrame containing values convert to a numpy ndarray.
    val_col : str
        The name of the column in the DataFrame that contains the values to be placed in the array.
    idx_col : str or list of str or None, default None
        The name(s) of the column(s) in the DataFrame that represent the dimensions of the array.
        If not provided, the index of the DataFrame will be used as the dimension(s).
    dropna : bool, default True
        Whether to drop rows with missing values before converting to the array.
    fill_val : scalar, default np.nan
        The value to fill in the array for missing values.

    Returns
    -------
    numpy array
        The resulting numpy array.

    Raises
    ------
    AssertionError
        If the dimension values are not integers or have gaps, or if the ``idx_col``
        parameter contains column names that are not in the DataFrame.

    Examples
    --------
    >>> import pandas as pd
    >>> import numpy as np
    >>> from GPSat.utils import dataframe_to_array
    >>> df = pd.DataFrame({
    ...     'dim1': [0, 0, 1, 1],
    ...     'dim2': [0, 1, 0, 1],
    ...     'values': [1, 2, 3, 4]
    ... })
    >>> arr = dataframe_to_array(df, 'values', ['dim1', 'dim2'])
    >>> print(arr)
    [[1 2]
     [3 4]]

    """

    # """
    # convert a DataFrame with multi-index to ndarray
    # multi-index must be integers - corresponding to location along a dimension
    # idx_col can be used to get index locations
    # """

    # get the dimension - i.e. the index location
    # - expected to integers and for each dimension range from 0,..., n-1

    # TODO: review this bit - copied from DataDict
    if idx_col is None:
        idx = df.index

        if dropna:
            df = df[[val_col]].dropna()

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
        if dropna:
            # pd.isnull can be slow for certain datatypes, would want to use np.isnan if dtype float (?)
            drop_where = pd.isnull(df[val_col])
            df = df.loc[~drop_where]

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

    # check if fill_val and dtype of value col are compatible
    if not np.issubdtype(np.dtype(type(fill_val)), df[val_col].dtype):
        # if type(fill_val) != df[val_col].dtype:
        warnings.warn(f"the fill value is type: {type(fill_val)}, " 
                      f"however val_col: '{val_col}' in df has dtype: {df[val_col].dtype} " 
                      f"this may cause an issue with filling in missing values ")

    # write output to array
    out = np.full(shape, fill_val, dtype=df[val_col].dtype)


    if out.dtype != df[val_col].dtype:
        warnings.warn(f"the output array has dtype: {out.dtype}, " \
                      f"however val_col: '{val_col}' in df has dtype: {df[val_col].dtype} " \
                      f"dtype in output array is determined by fill_val: {fill_val}")


    # assign values according to dimension values
    idx = tuple([v for v in dims.values()])
    out[idx] = df[val_col].values

    return out



def dict_of_array_to_dict_of_dataframe(array_dict, concat=False, reset_index=False):
    """
    Converts a dictionary of arrays to a dictionary of pandas DataFrames.

    Parameters
    ----------
    array_dict : dict
        A dictionary where the keys are strings and the values are numpy arrays.
    concat : bool, optional
        If ``True``, concatenates DataFrames with the same number of dimensions. Default is ``False``.
    reset_index : bool, optional
        If ``True``, resets the index of each DataFrame. Default is ``False``.

    Returns
    -------
    dict
        A dictionary where the keys are strings and the values are pandas DataFrames.

    Notes
    -----
    This function uses the :func:`array_to_dataframe <GPSat.utils.array_to_dataframe>`
    function to convert each array to a DataFrame. If ``concat`` is ``True``,
    it will concatenate DataFrames with the same number of dimensions. If ``reset_index`` is ``True``,
    it will reset the index of each DataFrame.

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> array_dict = {'a': np.array([1, 2, 3]), 'b': np.array([[1, 2], [3, 4]]), 'c': np.array([1.1, 2.2, 3.3])}
    >>> dict_of_array_to_dict_of_dataframe(array_dict)
    {'a':       a
        _dim_0   
        0       1
        1       2
        2       3,
    'b':               b
        _dim_0 _dim_1   
        0      0       1
               1       2
        1      0       3
               1       4,
    'c':        c
        _dim_0     
        0       1.1
        1       2.2
        2       3.3}

    >>> dict_of_array_to_dict_of_dataframe(array_dict, concat=True)
    {1:         a    c
        _dim_0
        0       1  1.1
        1       2  2.2
        2       3  3.3,
    2:                 b
        _dim_0 _dim_1
        0      0       1
               1       2
        1      0       3
               1       4}

    >>> dict_of_array_to_dict_of_dataframe(array_dict, reset_index=True)
    {'a':    _dim_0  a
        0       0    1
        1       1    2
        2       2    3,
     'b':    _dim_0  _dim_1  b
        0       0       0    1
        1       0       1    2
        2       1       0    3
        3       1       1    4,
     'c':    _dim_0  c
        0       0    1.1
        1       1    2.2
        2       2    3.3}

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
    """
    Converts a pandas Series or DataFrame (row) to a dictionary.

    Parameters
    ----------
    x: pd.Series, pd.DataFrame or dict
        The input object to be converted to a dictionary.

    Returns
    -------
    dict:
        A dictionary representation of the input object.

    Raises
    ------
    AssertionError: If the input object is a DataFrame with more than one row.

    Warnings
    --------
    If the input object is not a pandas Series, DataFrame, or dictionary,
    a warning is issued and the input object is returned as is.

    Examples
    --------
    >>> import pandas as pd
    >>> data = {'name': ['John', 'Jane'], 'age': [30, 25]}
    >>> df = pd.DataFrame(data)
    >>> pandas_to_dict(df)
    AssertionError: in pandas_to_dict input provided as DataFrame, expected to only have 1 row, shape is: (2, 2)

    >>> series = pd.Series(data['name'])
    >>> pandas_to_dict(series)
    {0: 'John', 1: 'Jane'}

    >>> dictionary = {'name': ['John', 'Jane'], 'age': [30, 25]}
    >>> pandas_to_dict(dictionary)
    {'name': ['John', 'Jane'], 'age': [30, 25]}

    select a single row of the dataframe

    >>> pandas_to_dict(df.iloc[[0]])
    {'name': 'John', 'age': 30}

    """

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
    """

    Create a 2D grid of points defined by x and y ranges,
    with the option to specify the grid resolution, step size, or number of steps.
    The resulting grid is flattened and concatenated into a 2D array of (x,y) coordinates.

    Parameters
    ----------
    x_range: tuple or list of floats
        Two values representing the minimum and maximum values of the x-axis range.
    y_range: tuple or list of floats
        Two values representing the minimum and maximum values of the y-axis range.
    grid_res: float or None, default None
        The grid resolution, i.e. the distance between adjacent grid points.
        If specified, this parameter takes precedence over ``step_size`` and ``num_step``.
    step_size: float or None, default None
        The step size between adjacent grid points.
        If specified, this parameter takes precedence over ``num_step``.
    num_step: int or None, default None
        The number of steps between the minimum and maximum values of the x and y ranges.
        If specified, this parameter is used only if ``grid_res`` and ``step_size`` are not specified (are ``None``).
        **Note:** the number of steps includes the starting point, so from 0 to 1 is two steps
    center: bool, default True
        - If ``True``, the resulting grid points will be the centers of the grid cells.
        - If ``False``, the resulting grid points will be the edges of the grid cells.

    Returns
    -------
    ndarray
        A 2D array of (x,y) coordinates, where each row represents a single point in the grid.

    Raises
    ------
    AssertionError
        If ``grid_res``, ``step_size`` and ``num_step`` are all unspecified. Must specify at least one.

    Examples
    --------
    
    >>> from GPSat.utils import grid_2d_flatten
    >>> grid_2d_flatten(x_range=(0, 2), y_range=(0, 2), grid_res=1)
    array([[0.5, 0.5],
           [1.5, 0.5],
           [0.5, 1.5],
           [1.5, 1.5]])

    """

    # create grid points defined by x/y ranges, and step size (grid_res

    # TODO: allow this to generalise to n-dims

    x_min, x_max = x_range[0], x_range[1]
    y_min, y_max = y_range[0], y_range[1]

    assert (grid_res is not None) | (step_size is not None) | (num_step is not None), \
        "grid_res, step_size and num_step are all None, please provide one"

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
    """
    Converts a string representation of longitude or latitude to a float value.

    Parameters
    ----------
    x: str
        A string representation of longitude or latitude in the format of ``"[degrees] [minutes] [direction]"``,
        where ``[direction]`` is one of ``"N"``, ``"S"``, ``"E"``, or ``"W"``.

    Returns
    -------
    float
        The converted value of the input string as a float.

    Raises
    ------
    AssertionError
        If the input is not a string.

    Examples
    --------
    >>> convert_lon_lat_str('74 0.1878 N')
    74.00313
    >>> convert_lon_lat_str('140 0.1198 W')
    -140.001997

    """

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
    """
    This function takes in a class object as input and returns a string representation of the class name
    without the leading "<class '" and trailing "'>".

    Alternatively will remove leading '<__main__.' and remove ' object at ', including anything that follows

    The function achieves this by invoking the __str__ method of the class object and
    then using regular expressions to remove the unwanted characters.

    Parameters
    ----------
    x: an arbitrary class instance

    Returns
    -------
    str

    Examples
    --------
    class MyClass:
        pass

    print(pretty_print_class(MyClass))

    """
    # invoke __str__
    out = str(x)
    # remove any leading <class ' and trailing '>
    if re.search("^<class", out):
        return re.sub("^<class '|'>$", "", out)
    # remove any leading <__main__ and everything after: object at
    elif re.search("^<__main", out):
        return re.sub("^<__main__\.| object at .*$", "", out)
    else:
        return out

def glue_local_predictions(preds_df: pd.DataFrame,
                           expert_locs_df: pd.DataFrame,
                           sigma: Union[int, float, list]=3
                           ) -> pd.DataFrame:
    """
    **Depracated.** Use :func:`glue_local_predictions_1d <GPSat.postprocessing.glue_local_predictions_1d>`
    and :func:`glue_local_predictions_2d <GPSat.postprocessing.glue_local_predictions_2d>` instead.

    Glues overlapping predictions by taking a normalised Gaussian weighted average.

    **Warning:** This method only deals with expert locations on a regular grid.

    Parameters
    ----------
    preds_df: pd.DataFrame
        containing predictions generated from local expert OI. It should have the following columns:

        - ``pred_loc_x`` (float): The x-coordinate of the prediction location.
        - ``pred_loc_y`` (float): The y-coordinate of the prediction location.
        - ``f*`` (float): The predictive mean at the location (pred_loc_x, pred_loc_y).
        - ``f*_var`` (float): The predictive variance at the location (pred_loc_x, pred_loc_y).

    expert_locs_df: pd.DataFrame
        containing local expert locations used to perform optimal interpolation.
        It should have the following columns:

        - ``x`` (float): The x-coordinate of the expert location.
        - ``y`` (float): The y-coordinate of the expert location.

    sigma: int, float, or list, default 3
        The standard deviation of the Gaussian weighting in the x and y directions.

        - If a single value is provided, it is used for both directions.
        - If a list is provided, the first value is used for the x direction and the second value is used for the y direction. Defaults to 3.

    Returns
    -------
    pd.DataFrame:
        Dataframe consisting of glued predictions (mean and std). It has the following columns:

        - ``pred_loc_x`` (float): The x-coordinate of the prediction location.
        - ``pred_loc_y`` (float): The y-coordinate of the prediction location.
        - ``f*`` (float): The glued predictive mean at the location (``pred_loc_x``, ``pred_loc_y``).
        - ``f*_std`` (float): The glued predictive standard deviation at the location (``pred_loc_x``, ``pred_loc_y``).

    Notes
    -----
    - The function assumes that the expert locations are equally spaced in both the x and y directions.
    - The function uses the ``scipy.stats.norm.pdf`` function to compute the Gaussian weights.
    - The function normalizes the weighted sums with the total weights at each location.


    """

    # TODO: confirm notes in docstring are accurate
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



def get_weighted_values(df, ref_col, dist_to_col, val_cols,
                        weight_function="gaussian",
                        drop_weight_cols=True,
                        **weight_kwargs):
    """
    Calculate the weighted values of specified columns in a DataFrame based on the distance between two other columns,
    using a specified weighting function. The current implementation supports a Gaussian weight based on the euclidean
    distance between the values in `ref_col` and `dist_to_col`.

    Parameters
    ----------
    df : pandas.DataFrame
        The input DataFrame containing the reference column, distance-to column, and value columns.
    ref_col : list of str or str
        The name of the column(s) to use as reference points for calculating distances.
    dist_to_col : list of str or str
        The name of the column(s) to calculate distances to, from `ref_col`. They should align / correspond to the
        column(s) set by ref_col.
    val_cols : list of str or str
        The names of the column(s) for which the weighted values are calculated. Can be a single column name or a list
        of names.
    weight_function : str, optional
        The type of weighting function to use. Currently, only "gaussian" is implemented, which applies a Gaussian
        weighting (exp(-d^2)) based on the squared euclidean distance. The default is "gaussian".
    drop_weight_cols: bool, optional, default True.
        if False the total weight and total weighted function values are included in the output
    **weight_kwargs : dict
        Additional keyword arguments for the weighting function. For the Gaussian weight, this includes:
        - lengthscale (float): The length scale to use in the Gaussian function. This parameter scales the distance
        before applying the Gaussian function and must be provided.

    Returns
    -------
    pandas.DataFrame
        A DataFrame containing the weighted values for each of the specified value columns. The output DataFrame has
        the reference column as the index and each of the specified value columns with their weighted values.

    Raises
    ------
    AssertionError
        If the shapes of the `ref_col` and `dist_to_col` do not match, or if the required `lengthscale` parameter for
        the Gaussian weighting function is not provided.

    NotImplementedError
        If a `weight_function` other than "gaussian" is specified.

    Examples
    --------
    >>> import pandas as pd
    >>>
    >>> data = {
    ...     'ref_col': [0, 1, 0, 1],
    ...     'dist_to_col': [1, 2, 3, 4],
    ...     'value1': [10, 20, 30, 40],
    ...     'value2': [100, 200, 300, 400]
    ... }
    >>> df = pd.DataFrame(data)
    >>> weighted_df = get_weighted_values(df, 'ref_col', 'dist_to_col', ['value1', 'value2'], lengthscale=1.0)
    >>> print(weighted_df)

    Notes
    -----
    - The function currently only implements Gaussian weighting. The Gaussian weight is calculated as exp(-d^2 / (2 * l^2)),
      where `d` is the squared euclidean distance between `ref_col` and `dist_to_col`, and `l` is the `lengthscale`.
    - This implementation assumes the input DataFrame does not contain NaN values in the reference or distance-to columns.
      Handling NaN values may require additional preprocessing or the use of fillna methods.
    """
    # TODO: have option to deal with nans
    # TODO: allow a custom weight function to be provided as input
    # TODO: allow for keeping of some columns, e.g. date
    #  - require for each ref_loc the count of other columns is on
    #  - the below is rather slow, so is excluded

    # make sure columns are list of str
    ref_col = [ref_col] if isinstance(ref_col, str) else ref_col
    dist_to_col = [dist_to_col] if isinstance(dist_to_col, str) else dist_to_col
    val_cols = [val_cols] if isinstance(val_cols, str) else val_cols

    # extract the reference location
    # - and to columns to get the distances to
    x0 = df[ref_col].values
    x = df[dist_to_col].values

    assert x0.shape == x.shape, \
        f"ref_col gave shape: {x0.shape}, dist_to_col gave shape: {x.shape} - they should be the same"

    # get the (un-normalised) weight
    if weight_function == "gaussian":
        # calculate the distance
        # d = cdist(x0, x, metric="euclidean")
        d = np.sum((x0 - x)**2, axis=1)

        # lengthscale - currently for aggregate distance only
        lscale = weight_kwargs.get("lengthscale", None)
        assert lscale is not None, "lscale is None, please provide"

        # NOTE: d is already in squared
        d2 = (d / lscale**2)

        # weight function: gaussian exp(-x^2 / 2)
        w = np.exp(-d2/2)

    else:
        raise NotImplementedError(f"weight_function: {weight_function} is not implemented")

    # store intermediate outputs in list
    out = []

    # apply weights to val columns
    for vc in val_cols:
        _ = df[ref_col + [vc]].copy(True)
        assert "_w" not in _
        _["_w"] = w
        # get the weighted values
        _[f"w_{vc}"] = w * _[vc].values

        # sum the weights and the weighted values
        # - not resetting so can can concat (e.g. concat/merge on index)
        _ = pd.pivot_table(_,
                           index=ref_col,
                           values=["_w", f"w_{vc}"],
                           aggfunc="sum")#.reset_index()

        # normalised
        _[vc] = _[f"w_{vc}"] / _["_w"]
        if drop_weight_cols:
            _.drop(["_w", f"w_{vc}"], axis=1, inplace=True)

        out.append(_)

    out = pd.concat(out, axis=1)
    out.reset_index(inplace=True)

    return out



def dataframe_to_2d_array(df, x_col, y_col, val_col, tol=1e-9, fill_val=np.nan, dtype=None, decimals=1):
    """
    Extract values from DataFrame to create a 2-d array of values (``val_col``)
    - assuming the values came from a 2-d array.
    Requires dimension columns ``x_col``, ``y_col`` (do not have to be ordered in DataFrame).

    Parameters
    ----------
    df: pandas.DataFrame
        The dataframe to convert to a 2D array.
    x_col: str
        The name of the column in the dataframe that contains the x coordinates.
    y_col: str
        The name of the column in the dataframe that contains the y coordinates.
    val_col: str
        The name of the column in the dataframe that contains the values to be placed in the 2D array.
    tol: float, default 1e-9
        The tolerance for matching the x and y coordinates to the grid.
    fill_val: float, default np.nan
        The value to fill the 2D array with if a coordinate is missing.
    dtype: str or numpy.dtype or None, default None
        The data type of the values in the 2D array.
    decimals: int, default 1
        The number of decimal places to round x and y values to before taking unique.
        If decimals is negative, it specifies the number of positions to the left of the decimal point.

    Returns
    -------
    tuple
        A tuple containing the 2D numpy array of values, the x coordinates of the grid, and the y coordinates of the grid.

    Raises
    ------
    AssertionError
        If any of the required columns are missing from the dataframe, or if any coordinates have more than one value.

    Notes
    -----
    - The spacing of grid is determined by the smallest step size in the ``x_col``, ``y_col`` direction, respectively.
    - This is meant to reverse the process of putting values from a regularly spaced grid into a DataFrame. \
    *Do not expect this to work on arbitrary x,y coordinates*.

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
    unique_x = np.sort(np.unique(np.round(df[x_col].unique(), decimals)))
    unique_y = np.sort(np.unique(np.round(df[y_col].unique(), decimals)))

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
    grid_loc_x = match(np.round(df[x_col].values, decimals), x_coords, exact=False, tol=tol)
    grid_loc_y = match(np.round(df[y_col].values, decimals), y_coords, exact=False, tol=tol)

    # check there is only one grid_loc for each point
    # df[['grid_loc_x', 'grid_loc_y']].drop_duplicates().shape[0] == pave.shape[0]

    # create a 2d array to populate
    # TODO: allow dtype, fill value to be determined from df[val_col]?
    val2d = np.full(x_grid.shape, fill_val,  dtype=dtype)
    # populate array
    val2d[grid_loc_y, grid_loc_x] = df[val_col].values

    return val2d, x_grid, y_grid


def softplus(x, shift=0):
    # ref: https://stackoverflow.com/questions/44230635/avoid-overflow-with-softplus-function-in-python
    # - more numerically stable for large x (e.g. x>710)
    return np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0) + shift


@nb.guvectorize([(nb.float64[:], nb.float64[:], nb.float64[:], nb.float64[:]),
                 (nb.float32[:], nb.float32[:], nb.float32[:], nb.float32[:])],
                "(),(),()->()", nopython=True, target="cpu")
def _inverse_softplus(y, shift, threshold, out):
    # TODO: define a numba function version of this? just to avoid the input_scalar bit
    # inverse of softplus - solve for y: y = log(1 + exp(x))
    # x = log(exp(y) - 1)
    # x = log(1 - exp(-y)) + y # more stable

    # # supress a divide by zero warning
    # with np.errstate(divide='ignore'):
    #     out = np.log(1 - np.exp(-y)) + y

    # this works better: np.expm1(x) = np.exp(x) - 1
    y_ = y[0]-shift[0]

    if y_ <= 0:
        out[0] = -np.inf
    else:
        # handle values which are too large or too small
        # y[0].dtype
        # threshold = -34 # np.log(np.finfo().eps) + 2.
        # too small?
        if y_ < np.exp(threshold[0]):
           out[0] = np.log(y_)
        # too big?
        elif y_ > -threshold[0]:
            out[0] = y_
        else:
            out[0] = np.log(-np.expm1(-y_)) + y_


def inverse_softplus(y, shift=0):
    # https://github.com/tensorflow/probability/blob/v0.19.0/tensorflow_probability/python/math/generic.py#L530-L581
    # TODO: define a numba function version of this? just to avoid the input_scalar bit
    # inverse of softplus - solve for y: y = log(1 + exp(x))
    # x = log(exp(y) - 1)
    # x = log(1 - exp(-y)) + y # more stable

    # # supress a divide by zero warning
    # with np.errstate(divide='ignore'):
    #     out = np.log(1 - np.exp(-y)) + y

    if isinstance(y, np.ndarray):
        threshold = np.log(np.finfo(y.dtype).eps) + 2.
    elif isinstance(y, (int, float)):
        threshold = np.log(np.finfo(np.float64).eps) + 2.

    return _inverse_softplus(y, shift, threshold)


def sigmoid(x, low=0, high=1):
    # scaled sigmoid, giving an output between b and high
    assert high > low
    return (high - low) / (1 + np.exp(-x)) + low


@nb.guvectorize([(nb.float64[:], nb.float64[:], nb.float64[:], nb.float64[:]),
                 (nb.float32[:], nb.float32[:], nb.float32[:], nb.float32[:])],
                "(),(),()->()", nopython=True, target="cpu")
def _inverse_sigmoid(y, low, high, out):

    if y[0] <= low[0]:
        out[0] = -np.inf
    elif y[0] >= high[0]:
        out[0] = np.inf
    else:
        out[0] = -np.log((high[0] - low[0]) / (y[0] - low[0]) - 1)


def inverse_sigmoid(y, low=0, high=1):
    assert high > low
    out = _inverse_sigmoid(y, low, high)
    # out = -np.log((high - low) / (y - low) - 1)
    return out

def cprint(x, c="ENDC", bcolors=None, sep=" ", end="\n"):
    """
    Add color to print statements.

    Based off of https://stackoverflow.com/questions/287871/how-do-i-print-colored-text-to-the-terminal.

    Parameters
    ----------
    x: str
        String to be printed.
    c: str, default "ENDC"
        Valid key in ``bcolors``. If ``bcolors`` is not provided, then default will be used, containing keys:
        ``'HEADER'``, ``'OKBLUE'``, ``'OKCYAN'``, ``'OKGREEN'``, ``'WARNING'``, ``'FAIL'``, ``'ENDC'``,
        ``'BOLD'``, ``'UNDERLINE'``.
    bcolors: dict or None, default None
        Dict with values being colors / how to format the font. These cane be chained together.
        See the codes in: https://en.wikipedia.org/wiki/ANSI_escape_code#3-bit_and_4-bit.
    sep: str, default " "
        ``sep`` argument passed along to ``print()``.
    end: str, default "\\\\n"
        ``end`` argument passed along to ``print()``.

    Returns
    -------
    None

    """

    # ref: https://stackoverflow.com/questions/287871/how-do-i-print-colored-text-to-the-terminal
    if bcolors is None:
        bcolors = dict(
            HEADER='\033[95m',
            OKBLUE='\033[94m',
            OKCYAN='\033[96m',
            OKGREEN='\033[92m',
            WARNING='\033[93m',
            FAIL='\033[91m',
            ENDC='\033[0m',
            BOLD='\033[1m',
            UNDERLINE='\033[4m',
        )
    # x = sep.join([str(a) for a in args])
    try:
        print(f"{bcolors[c]}{x}{bcolors['ENDC']}", sep=sep, end=end)
    # handle any exception to avoid breaking
    # TODO: provide more specific error handling
    except Exception as e:
        print(x)


def rmse(y, mu):
    return np.sqrt(np.mean((y - mu)**2))


def nll(y, mu, sig, return_tot=True):
    # negative log likelihood assuming independent normal observations (y)
    out = np.log(sig * np.sqrt(2 * np.pi)) + (y - mu)**2 / (2 * sig**2)
    if return_tot:
        return np.sum(out[~np.isnan(out)])
    else:
        return out


@nb.jit(nopython=True)
def guess_track_num(x, thresh, start_track=0):
    out = np.full(len(x), np.nan)
    track_num = start_track
    for i in range(0, len(x)):
        # if there is a jump, increment track
        if x[i] > thresh:
            track_num += 1
        out[i] = track_num
    return out


@nb.jit(nopython=True)
def track_num_for_date(x):
    out = np.full(len(x), np.nan)
    out[0] = 0
    for i in range(1, len(x)):
        if x[i] == x[i - 1]:
            out[i] = out[i - 1] + 1
        else:
            out[i] = 0

    return out


def diff_distance(x, p=2, k=1, default_val=np.nan):
    # given a 2-d array, get the p-norm distance between (k) rows
    # require x be 2d if it is 1d
    if len(x.shape) == 1:
        x = x[:, None]
    assert len(x.shape) == 2, f"x must be 2d, len(x.shape) = {len(x.shape)}"
    out = np.full(x.shape[0], default_val)

    # get the difference raised to the pth power
    dx = (x[k:, :] - x[:-k, :]) ** p
    # sum over rows
    dx = np.sum(dx, axis=1)
    # raise to the 1/p
    dx = dx ** (1 / p)
    # populate output array
    out[k:] = dx

    return out


def compare_dataframes(df1, df2, merge_on, columns_to_compare,
                       drop_other_cols=False,
                       how="outer", suffixes=["_1", "_2"]):

    # compare columns of two dataframes after performing on merge
    # columns_to_compare must be in both dataframes,
    # for these columns additional columns for abs_diff and rel_diff will be added

    # original columns
    c1, c2 = [_.columns.values for _ in (df1, df2)]
    org_col = np.concatenate([c1, c2[~np.in1d(c2, c1)]])

    # merge
    df = df1.merge(df2,
                   on=merge_on,
                   how=how,
                   suffixes=suffixes)

    columns_to_compare = [columns_to_compare] if isinstance(columns_to_compare, str) else columns_to_compare

    for col in columns_to_compare:
        # Calculating absolute differences
        df[col + '_abs_diff'] = np.abs(df[col + suffixes[0]] - df[col + suffixes[1]])
        # Calculating relative differences
        df[col + '_rel_diff'] = df[col + '_abs_diff'] / np.minimum(np.abs(df[col + suffixes[0]]),
                                                                   np.abs(df[col + suffixes[1]]))

    # set column order, '' is for if col exists on only one df
    suffixes_order = suffixes + ['_abs_diff', '_rel_diff', '']

    col_ord = [f"{c}{s}" for c in org_col for s in suffixes_order if f"{c}{s}" in df]
    col_ord = np.array(col_ord)

    # check no columns are missing
    missing_cols = df.columns.values[~np.in1d(df.columns.values, col_ord)]
    assert len(missing_cols) == 0, f"missing_cols: {missing_cols}"

    # take only merge_on and those being diffed?
    if drop_other_cols:
        diff_cols = [f"{c}{s}" for c in columns_to_compare for s in suffixes + ['_abs_diff', '_rel_diff']]
        col_ord = col_ord[np.in1d(col_ord, merge_on + diff_cols)]

    return df[col_ord]


def _method_inputs_to_config(locs, code_obj, verbose=False):
    # this function aims to take the arguments of a function/method and store them in a dictionary
    # copied from LocalExpertOI
    # TODO: validate this method returns expected values - i.e. the arguments provided to a function
    # TODO: look into making this method into a decorator

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
            try:
                for kw, v in locs[var].items():
                    config[kw] = v
            except KeyError as e:
                if verbose:
                    print(f"KeyError on var: {var}\n", e, "skipping")
        else:
            # HACK: to deal with 'config' was unexpectedly coming up - in set_model only
            try:
                config[var] = locs[var]
            except KeyError as e:
                if verbose:
                    print(f"KeyError on var: {var}\n", e, "skipping")
    return json_serializable(config)



def pip_freeze_to_dataframe():
    # chatGPT generated, then modified to handle conda packages
    # TODO: add docstring
    # Run pip freeze and capture its output
    result = subprocess.run(['pip', 'freeze'], capture_output=True, text=True)
    pip_output = result.stdout

    # Split the output by lines and parse each line
    packages = []
    for line in pip_output.strip().split('\n'):
        # expect freeze to only provide '==' or '@'
        if '==' in line:
            package, version = line.split('==')
            packages.append({'Package Name': package, 'Package Version': version, 'Source': 'pip'})

        elif "@" in line:
            package, version = [_.lstrip().rstrip() for _ in line.split('@')]
            packages.append({'Package Name': package, 'Package Version': version, 'Source': 'conda'})
        else:
            raise NotImplementedError(f"in parsing 'pip freeze' output, encountered a line\n{line}\n"
                                      f" which is missing '==' or '@', do not know how to handle")

    # Convert the list of dictionaries to a DataFrame
    df = pd.DataFrame(packages)

    return df



if __name__ == "__main__":

    # ---
    # color print
    # ---

    print("color print examples")
    for k in ['HEADER', 'OKBLUE', 'OKCYAN', 'OKGREEN', 'WARNING', 'FAIL', 'ENDC', 'BOLD', 'UNDERLINE']:
        cprint(f"k:{k}", c=k)

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

    # NO LONGER WORKS!
    # chk, x_chk, y_chk = dataframe_to_2d_array(df, x_col="x", y_col="y", val_col="z")
    #
    # # check all values were recovered
    # assert np.all(chk == vals)
    # assert np.all(x_chk == x_grid)
    # assert np.all(y_chk == y_grid)

    # --
    # convert string coordinate to decimal
    # --
    convert_lon_lat_str('74 0.1878 N')
    convert_lon_lat_str('140 0.1198 W')

    # ---
    # array_to_dataframe
    # ---

    # store an ndarray in a DataFrame
    x = np.array([[1, 2], [3, 4]])
    xdf = array_to_dataframe(x, "data")

    # ---
    # dataframe_to_array
    # ---

    # extract an ndarray from DataFrame - using index values
    x0 = dataframe_to_array(xdf, val_col='data')

    # get the index/dimension names, move index to col (with reset_index), be explicit with idx_col
    idx_col = list(xdf.index.names)
    xdf.reset_index(inplace=True)
    x1 = dataframe_to_array(df=xdf, val_col='data', idx_col=idx_col)

    # convert recover original x arrays in both cases
    assert np.all(x == x0)
    assert np.all(x == x1)

    # ----
    # dict of array
    # ---

    array_dict = {'a': np.array([1, 2, 3]), 'b': np.array([[1, 2], [3, 4]]), 'c': np.array([1.1, 2.2, 3.3])}
    dict_of_array_to_dict_of_dataframe(array_dict)

    # ---
    # apply literal_eval to keys in dict: convert str to tuples where that makes sense
    # ---

    conf = {
        'key1': 'value1',
         '("key2", "key3")': 'value2',
         'key4': {'("key5", "key6")': 'value3'}
    }
    for k, v in conf.items():
        print(f"key: {k} is type: {type(k)}")
    c = nested_dict_literal_eval(conf)
    print("after applying nested_dict_literal_eval")
    for k, v in c.items():
        print(f"key: {k} is type: {type(k)}")

    # ---
    # to_array
    # ---

    x = [1, 2, 3]
    y = np.array([4, 5, 6])
    z = datetime.date(2021, 1, 1)
    for arr in to_array(x, y, z):
        print(f"arr type: {type(arr)}, values: {arr}")

    # convert a single array like object
    _, = to_array(y)

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
