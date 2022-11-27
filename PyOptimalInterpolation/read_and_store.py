# read and store data functions
import json
import os
import re
import warnings

import pandas as pd
import numpy as np

from PyOptimalInterpolation.utils import config_func


def read_flat_files(file_dirs, file_regex,
                    sub_dirs=None,
                    read_csv_kwargs=None,
                    col_funcs=None,
                    row_select=None,
                    col_select=None,
                    new_column_names=None,
                    strict=True,
                    verbose=False):
    """
    read flat files (csv, tsv, etc) from file system

    Parameters
    ----------
    file_dirs
    file_regex
    sub_dirs
    read_csv_kwargs
    col_funcs
    row_select
    col_select
    new_column_names
    strict
    verbose

    Returns
    -------
    pd.DataFrame
    """

    # TODO: finish doc string
    # TODO: provide option to use os.walk?
    # TODO: check contents of row_select, col_funcs - make sure only valid keys are provided

    # ---
    # check inputs, handle defaults

    # additional kwargs for pd.read_csv
    if read_csv_kwargs is None:
        read_csv_kwargs = {}
    assert isinstance(read_csv_kwargs, dict), f"expect read_csv_kwargs to be a dict, is type: {type(read_csv_kwargs)}"

    # functions used to generate column values
    if col_funcs is None:
        col_funcs = {}
    assert isinstance(col_funcs, dict), f"expect col_funcs to be a dict, is type: {type(col_funcs)}"

    if row_select is None:
        row_select = [{}]
    elif isinstance(row_select, dict):
        row_select = [row_select]

    if col_select is None:
        if verbose:
            print("col_select is None, will take all")
        col_select = slice(None)

    assert isinstance(row_select, list), f"expect row_select to be a list (of dict), is type: {type(col_funcs)}"
    for i, rs in enumerate(row_select):
        assert isinstance(rs, dict), f"index element: {i} of row_select was type: {type(rs)}, rather than dict"

    if isinstance(file_dirs, str):
        file_dirs = [file_dirs]

    if sub_dirs is None:
        sub_dirs = [""]
    elif isinstance(sub_dirs, str):
        sub_dirs = [sub_dirs]

    if len(sub_dirs):
        sub_file_dirs = []
        for fd in file_dirs:
            for sd in sub_dirs:
                sub_file_dirs += [os.path.join(fd, sd)]
        file_dirs = sub_file_dirs

    # check file_dirs exists
    for file_dir in file_dirs:
        if strict:
            assert os.path.exists(file_dir), f"file_dir:\n{file_dir}\ndoes not exist"
        elif not os.path.exists(file_dir):
            warnings.warn(f"file_dir:\n{file_dir}\nwas provide but does not exist")
    file_dirs = [f for f in file_dirs if os.path.exists(f)]

    # for each file, read in
    # - store results in a list
    res = []
    for file_dir in file_dirs:
        print("-" * 100)
        print(f"reading files from:\n{file_dir}")
        # get all files in file_dir matching expression
        files = [os.path.join(file_dir, _) for _ in os.listdir(file_dir) if re.search(file_regex, _)]

        # increment over each file
        for f_count, f in enumerate(files):

            if verbose >= 2:
                print(f"reading file: {f_count+1}/{len(files)}")
            # read_csv
            df = pd.read_csv(f, **read_csv_kwargs)

            if verbose >= 3:
                print(f"read in: {f}\nhead of dataframe:\n{df.head(3)}")

            # ---
            # apply column functions
            # - used to add new columns

            for new_col, col_fun in col_funcs.items():

                # add new column
                if verbose >= 3:
                    print(f"adding new_col: {new_col}")
                df[new_col] = config_func(df=df,
                                          filename=f,
                                          **col_fun)

            # ----
            # select rows
            select = np.ones(len(df), dtype=bool)

            for sl in row_select:
                # print(sl)
                if verbose >= 3:
                    print("selecting rows")
                select &= config_func(df=df, filename=f, **sl)

            # select subset of data
            if verbose >= 3:
                print(f"selecting {select.sum()}/{len(select)} rows")
            df = df.loc[select, :]

            # ----
            # select columns

            # TODO: add more checks around this
            df = df.loc[:, col_select]

            if verbose >= 2:
                print(f"adding data with shape: {df.shape}")

            # change column names
            if new_column_names is not None:
                df.columns = new_column_names

            # -----
            # store results
            res += [df]

    # ----
    # concat all
    out = pd.concat(res)

    return out


def store_data(df, output_dir, out_file,
               storage_type=None,
               table=None,
               append=False,
               config=None,
               run_info=None):

    if storage_type is None:
        print("inferring storage_type from out_file")
        storage_type = re.sub("^.*\.", "", out_file)

    # TODO: add functionality for csv, tsv
    # TODO: make sub-functions for each storage_type

    valid_store_type = ["h5"]
    assert isinstance(storage_type, str)
    storage_type = storage_type.lower()
    assert storage_type in valid_store_type

    if storage_type == 'h5':

        assert table is not None, f"for storage_type=={storage_type} 'table' must be provide"
        assert os.path.exists(output_dir), f"output_dir:\n{output_dir}\ndoes not exist"
        full_path = os.path.join(output_dir, out_file)

        # open HDFStore
        store = pd.HDFStore(full_path, mode='a')

        # write table
        store.put(key=table,
                  value=df,
                  append=append,
                  format='table',
                  data_columns=True)
        # ---
        # add meta-data / attributes
        # ---

        # configuration - used to generate df (if applicable)
        if config is None:
            print("config is None, will not assign as attr")
        else:
            # store config
            store_attrs = store.get_storer(table).attrs

            # store config and run information
            # if already has 'config' as attribute, then extend (in list)
            if hasattr(store_attrs, 'config') & append:
                prev_config = store_attrs.config
                prev_config = prev_config if isinstance(prev_config, list) else [prev_config]
                store_attrs.config = prev_config + [config]
            # store config
            else:
                store_attrs.config = config

        # run information - information data was generated
        if run_info is None:
            print("run_info is None, will not assign as attr")
        else:

            if hasattr(store_attrs, 'run_info') & append:
                prev_run_info = store_attrs.run_info
                prev_run_info = prev_run_info if isinstance(prev_run_info, list) else [prev_run_info]
                store_attrs.run_info = prev_run_info + [run_info]
            else:
                store_attrs.run_info = run_info


if __name__ == "__main__":

    import datetime
    import sys
    from PyOptimalInterpolation.utils import get_config_from_sysargv, get_git_information

    pd.set_option("display.max_columns", 200)

    # ---
    # read in / specify config
    # ---

    # get a config file (dict from json)
    # - requires argument provided
    config = get_config_from_sysargv()

    # copy the original config
    org_config = config.copy()

    # extract output_dir/prefix
    output_dict = config.pop("output", None)

    if output_dict is None:
        warnings.warn("'output' not provided, won't write data to file")
    else:
        output_dir = output_dict['dir']
        out_file = output_dict['file']
        assert os.path.exists(output_dir), f"output_dir:\n{output_dir}\ndoes not exist, please create"

    # record run time for reference
    run_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # --
    # read in data, select rows and columns, combine into a single dataframe
    # --

    df = read_flat_files(**config)

    # ----
    # store results
    # ---

    if output_dict is None:
        print("no 'output' dict provided, not writing data to file")

    else:

        # ---
        # get run information
        # ---

        run_info = {
            "run_time": run_time,
            "python_executable": sys.executable,
        }
        try:
            run_info['script_path'] = os.path.abspath(__file__)
        except NameError as e:
            pass
        try:
            git_info = get_git_information()
        except Exception as e:
            git_info = {}

        run_info = {**run_info, **git_info}

        # ---
        # write data to filesystem
        # ---

        out_file = output_dict['file']
        table = output_dict.get('table', None)
        append = output_dict.get("append", False)

        store_data(df=df,
                   output_dir=output_dir,
                   out_file=out_file,
                   append=append,
                   table=table,
                   config=config,
                   run_info=run_info)

