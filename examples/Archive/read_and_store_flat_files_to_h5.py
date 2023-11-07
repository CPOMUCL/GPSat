# example script showing the type of config file
import numpy as np
import pandas as pd
import sys
import warnings
import os
import datetime

from GPSat.dataloader import DataLoader
from GPSat.utils import get_config_from_sysargv

pd.set_option("display.max_columns", 200)

# ---
# read in / specify config
# ---

# get configuration file from argument provided to script (when running from command line)
# config = get_config_from_sysargv(argv_num=1)

# example config
# TODO: double check if this works - remove escape characters
config = {
    # output dict - specify output directory and file
    "output": {
        # directory, file and table (key) to write to
        # - in this case append data to "data" table
        "dir": "/mnt/hd1/data/ocean_elev/GPOD",
        "file": "gpod_example.h5",
        "table": "data",
        "append": True
    },
    # directories on where to read data from (can be a list of directories)
    # - change as needed
    "file_dirs": [
        "/mnt/hd1/data/ocean_elev/GPOD/proc_files/CS2_SAR",
        "/mnt/hd1/data/ocean_elev/GPOD/proc_files/CS2_SARIN",
        "/mnt/hd1/data/ocean_elev/GPOD/proc_files/S3B",
        "/mnt/hd1/data/ocean_elev/GPOD/proc_files/S3A"
    ],
    # sub directories in (each) file_dirs. can be missing or None if not needed
    "sub_dirs": [
        "202002",
        "202003",
        "202004"
    ],
    # regular expression to identify which files to read
    "file_regex": "v3\.proc$",
    # key word arguments to provided to pd.read_csv when reading in flat files
    "read_csv_kwargs": {
        "header": None,
        "sep": "\s+"
    },
    # 'column functions' - used to add columns to data
    "col_funcs": {
        # key specifies new columns name to be added - (bit faffy)
        # - storing column as category rather than str/object as it uses less memory
        # - appending columns of categories in h5 files requires "categories" to be consistent
        # parse satellite name from file name (path)
        "sat_": {
            # define a lambda function - this need not be a string (would be if saving as json)
            "func": "lambda x: os.path.basename(os.path.abspath(os.path.join(x, '../..'))).split('_')[0]",
            # provide the file name (full path) as first argument
            "filename_as_arg": True
        },
        # convert "sat" column to category - saves space
        "sat": {
            "func": "lambda x: pd.Categorical(x, categories=['S3A', 'S3B', 'CS2'])",
            "col_args": "sat_"
        },
        "datetime": {
            "source": "GPSat.datetime_utils",
            "func": "from_file_start_end_datetime_GPOD",
            # provide the file name associated with current dataframe as agrument to function?
            # - e.g. to parse date information from file name
            "filename_as_arg": True,
            "col_args": 1
        },
        "elev_mss": {
            # use a lambda function to take difference of columns 9 and 10
            # - giving elev - mss (elevation minus mean sea surface)
            "func": "lambda x,y: x-y",
            "col_args": [
                9,
                10
            ]
        }
    },
    # which rows to be select? can be missing if want to take all
    # list of dict which can be consumed by config_func, similar col_funcs above
    "row_select": [
        {
            # if function has special characters e.g. +-*/= will be used as a comparison
            # i.e. will be used to create a function: lambda arg1, arg2: arg1 func arg2
            # e.g. here 7th column of dataframe will be compared to the value 3, returning True if they are equal
            "func": "==",
            "col_args": 7,
            "args": 3
        },
        {
            # use a function that can be imported e.g. from {source} import {func}
            # - provide to function as first argument col 9 of the data
            "source": "GPSat.utils",
            "func": "not_nan",
            "col_args": 9
        },
        {
            # can use a lambda function instead of "not_nan" to evaluate of column 10 is nan or not
            # - NOTE: can be done (?) because config_func() is aware of numpy (np), i.e. is import in script
            "func": "lambda x: ~np.isnan(x)",
            "col_args": 10
        }
    ],
    # which columns to select?
    # - data read in without headers can be selected with integers
    "col_select": [
        0,
        1,
        9,
        10,
        "datetime",
        "sat"
    ],
    # re-name columns when returning output
    # - new column names must aligned to those in col_select
    "new_column_names": [
        "lon",
        "lat",
        "elev",
        "mss",
        "datetime",
        "sat"
    ],
    # increase verbose level to see more details
    "verbose": 1
}

# copy the original config
org_config = config.copy()

# extract output_dir/prefix
output_dict = config.pop("output", None)

if output_dict is None:
    warnings.warn("'output' not provided, won't write data to file")
else:
    output_dir = output_dict['dir']
    assert os.path.exists(output_dir), f"output_dir:\n{output_dir}\ndoes not exist, please create"

# record run time for reference
run_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

# --
# read in data, select rows and columns, combine into a single dataframe
# --

read_flat_files = DataLoader.read_flat_files

df = read_flat_files(**config)

# ---
# store data on files system
# ---

# def store_data()

if output_dict is None:
    print("no 'output' dict provided, not writing data to file")

else:

    # ---
    # get run information
    # ---

    # run info - if __file__ does not exist in environment (i.e. when running interactive)
    try:
        run_info = DataLoader.get_run_info(script_path=__file__)
    except NameError as e:
        run_info = DataLoader.get_run_info()

    # ---
    # write to file
    # ---

    out_file = output_dict['file']
    table = output_dict['table']
    append = output_dict.get("append", False)

    print("writing to hdf5 file")
    with pd.HDFStore(path=os.path.join(output_dir, out_file), mode='w') as store:
        DataLoader.write_to_hdf(df,
                                table=table,
                                append=append,
                                store=store,
                                config=config,
                                run_info=run_info)







