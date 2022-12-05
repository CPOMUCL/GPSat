# read and store data functions
import json
import os
import re
import warnings

import pandas as pd
import numpy as np

from PyOptimalInterpolation.utils import config_func


if __name__ == "__main__":

    import datetime
    import sys
    from PyOptimalInterpolation.dataloader import DataLoader
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

    read_flat_files = DataLoader.read_flat_files
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

        # run info - if __file__ does not exist in environment (i.e. when running interactive)
        try:
            run_info = DataLoader.get_run_info(script_path=__file__)
        except NameError as e:
            run_info = DataLoader.get_run_info()

        # ---
        # write data to filesystem
        # ---

        out_file = output_dict['file']
        table = output_dict.get('table', None)
        append = output_dict.get("append", False)
        mode = output_dict.get("mode", "w")

        print("writing to hdf5 file")
        with pd.HDFStore(path=os.path.join(output_dir, out_file), mode=mode) as store:
            DataLoader.write_to_hdf(df,
                                    table=table,
                                    append=append,
                                    store=store,
                                    config=config,
                                    run_info=run_info)

