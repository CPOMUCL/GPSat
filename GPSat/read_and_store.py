# read and store data functions
import json
import os
import re
import warnings
import time
import logging

import pandas as pd
import numpy as np

from GPSat import get_parent_path
from GPSat.utils import log_lines, cprint

# --
# helper functions
# --

def update_attr(x, cid, vals):
    # - assigning new key values does not work in place for HDFstore.get_storer(table).attrs (?)
    # - so need to copy contents (dict), modify, then re-assign
    # - could be slow is attributes get really (really) big
    assert isinstance(x, dict)
    tmp = x.copy()
    tmp[cid] = vals
    return tmp


def get_dirs_to_search(file_dirs, sub_dirs=None, walk=False):

    # match a list of dicts
    # - containing the batch components per batch
    fdirs = file_dirs if isinstance(file_dirs, (list, tuple)) else [file_dirs]

    # if not walking - just make sure any sub dirs are list/tuple
    if not walk:
        sdirs = sub_dirs if isinstance(sub_dirs, (list, tuple)) else [sub_dirs]
    else:
        print(f"will use os.walk to determine directories to search.")
        if sub_dirs is not None:
            warnings.warn(f"\nsub_dirs provide along with walk={walk}\nsub_dirs={sub_dirs}\nwill be ignored")
        sdirs = [None]

        # assert len(fdirs) == 1, f"walk currently only works with one file_dir, currently have {len(fdirs)}"

        new_fdirs = set()
        for fdir in fdirs:
            for root, subs, files in os.walk(fdir):
                # print(root)

                # check contents of file for any
                matched_regex = False
                for i in os.listdir(root):
                    if re.search(tmp_config['file_regex'], i):
                        matched_regex = True
                        break
                # if directory contained files matching regex then want to search that one
                if matched_regex:
                    print(f"adding folder to search: {root}")
                    new_fdirs.add(root)

        fdirs = list(new_fdirs)

    return fdirs, sdirs


if __name__ == "__main__":

    # TODO: clean up the print statements in this file
    # TODO: this script has become long, sections should be moved into functions / methods (in DataLoader?)
    # TODO: provide examples with walk=True?
    # TODO: add run information

    from GPSat.dataloader import DataLoader
    from GPSat.utils import get_config_from_sysargv

    pd.set_option("display.max_columns", 200)

    # ---
    # read in / specify config
    # ---

    # get a config file (dict from json)
    # - requires argument provided
    config = get_config_from_sysargv(argv_num=1)

    # assert config is not None, f"config is empty / not provided, must specify path to config (json file) as argument"
    if config is None:
        config_file = get_parent_path("configs", "example_read_and_store_raw_data.json")
        warnings.warn(f"\nconfig is empty / not provided, will just use an example config:\n{config_file}")
        with open(config_file, "r") as f:
            config = json.load(f)

        # override the
        config['output']['dir'] = get_parent_path("data", "example")
        config['file_dirs'] = get_parent_path("data", "example")

    # copy the original config - this is not used...
    org_config = config.copy()

    if config.get("verbose", False):
        cprint("*" * 25, c="BOLD")
        cprint("\nusing config:\n", c="BOLD")
        cprint(json.dumps(config, indent=4), c="HEADER")
        cprint("*" * 25, c="BOLD")

    # extract output_dir/prefix
    output_dict = config.pop("output", None)

    assert isinstance(output_dict, dict), f"'output' in config must be a dict, got: {type(output_dict)} "

    # extract parameters from output dictionary
    output_dir = output_dict['dir']
    out_file = output_dict['file']
    table = output_dict['table']

    log_file = get_parent_path("logs", re.sub("\..*$", ".log", out_file))
    os.makedirs(os.path.dirname(log_file), exist_ok=True)

    # get overwrite term - equivalent to not append (kept for legacy reasons)
    overwrite = output_dict.get("overwrite", not output_dict.get("append", True))
    verbose = config.get("verbose", True)

    assert os.path.exists(output_dir), f"output_dir:\n{output_dir}\ndoes not exist, please create"

    full_path = os.path.join(output_dir, out_file)

    # specify logging output/level      
    logging.basicConfig(filename=log_file,
                        format="%(levelname)s:%(message)s",
                        # encoding="utf-8",
                        filemode="w",
                        level=logging.DEBUG)
    # ----
    # prepare to read in files
    # ---

    # if overwrite is True, delete the table - if it exists
    if overwrite:
        print(f"overwrite is: {overwrite}\nwill remove existing table: {table}\n in:\n{full_path}")
        if not os.path.exists(full_path):
            if verbose:
                print(f"overwrite: {overwrite}, but full_path:\n{full_path}\ndoes not exist")
        else:
            # delete table if exists
            with pd.HDFStore(full_path) as store:
                try:
                    store.remove(table)
                except KeyError as e:
                    if verbose:
                        print(f"overwrite is True, the file:\n{full_path}\nexists, but the table: '{table}' does not (?)")
                        print(e)
    else:
        pass

    # make a copy of the config - will remove file_dirs and sub_dirs
    tmp_config = config.copy()

    # batch based on certain keys: file_dirs and sub_dirs (for now)
    fdirs = tmp_config.pop('file_dirs')
    sdirs = tmp_config.pop("sub_dirs", None)
    walk = tmp_config.pop("walk", False)

    # get the directories to search over
    fdirs, sdirs = get_dirs_to_search(fdirs, sub_dirs=sdirs, walk=walk)

    batch_table = f"_{table}_batches"

    # get run information - for this batch
    # run info - if __file__ does not exist in environment (i.e. when running interactive)
    try:
        run_info = DataLoader.get_run_info(script_path=__file__)
    except NameError as e:
        run_info = DataLoader.get_run_info()

    # --
    # get config_id (location), which batches to run
    # --

    # determine the config id - by checking if this config matches previous (excluding file_dirs and sub_dirs)
    with pd.HDFStore(full_path, mode="a") as store:
        try:
            store_attrs = store.get_storer(table).attrs

            matched_config = False
            # for k, v in enumerate(store_attrs['config']):
            for k, v in store_attrs['config'].items():
                if v == tmp_config:
                    #print("matched previous config")
                    config_id = k
                    log_lines("matched previous config", f"config_id: {config_id}", level="info")
                    matched_config = True
                    break

            # if have not previously used config increment config_id
            if not matched_config:
                # config_id = max([k for k in store_attrs['config'].keys()]) + 1
                config_id = k + 1
                prev_batches = []
            else:
                prev_batches = store.get(batch_table).to_dict(orient="records")

        except KeyError as e:
            print("on first iteration? got the following error:")
            print(e)
            config_id = 0
            prev_batches = []

    # get all the batches to (potentially) be run
    all_batches = [{"file_dirs": f, "sub_dirs": s, "config_id": config_id}
                   for f in np.unique(fdirs)
                   for s in np.unique(sdirs)]

    # determine the batches to run - those not run previously
    batches = [b for b in all_batches if b not in prev_batches]

    # ---
    # increment over batches
    # ---

    if verbose:
        print(f"there are: {len(batches)} batches to increment over")

    for bidx, b in enumerate(batches):
        
        if verbose:
            print("*" * 100)
            print(f"batch: {b}")

        # 'merge' the config with current batch (dict)
        b_org = b.copy()
        b.pop("config_id")
        tmp = {**tmp_config, **b}

        # read data into memory
        try:
            # df = DataLoader.read_flat_files(**tmp)
            df = DataLoader.read_from_multiple_files(**tmp)
        except pd.errors.ParserError as e:
            log_lines("*" * 10, e, b, "skipping", level="debug")
            continue
        except AssertionError as e:
            log_lines("*" * 10, e, b, "skipping", level="debug")
            continue
            
        if len(df) == 0:
            print("no data was read in, skipping")
            log_lines("*" * 10, b, "no data was read in, skipping", level="info")
            continue

        # write (append) to table
        with pd.HDFStore(full_path, mode="a") as store:

            try:
                # write table
                store.put(key=table,
                          value=df,
                          append=True,
                          format='table',
                          data_columns=True)
            except Exception as e:
                print(f"Exception:\n{e}\nskipping")
                continue

            t0 = time.time()

            # add current batch to the run_batches attribute
            store_attrs = store.get_storer(table).attrs

            # attributes will be missing on the first attempt
            if "config" not in store_attrs:
                store_attrs['config'] = {}
                store_attrs['run_info'] = {}
                # store_attrs['run_batches'] = {}

            # if on a new config_id
            # TODO: need to confirm what the limit of the attributes will be
            #   - how many configs, run_infos can be added before something breaks?
            if config_id not in store_attrs['config']:
                # need to re-assign full dict (attr) - changing a single value in place does not work (?)
                store_attrs['config'] = update_attr(store_attrs['config'], config_id, org_config)
                store_attrs['run_info'] = update_attr(store_attrs['run_info'], config_id, [])
                # store_attrs['run_batches'] = update_attr(store_attrs['run_batches'], config_id, [])

            # if on first batch - provide run_info
            if bidx == 0:
                store_attrs['run_info'] = update_attr(store_attrs['run_info'], config_id, store_attrs['run_info'][config_id] + [run_info])

            # # add batch
            # NOTE: there is a limit to adding via a batch table
            # store_attrs['run_batches'] = update_attr(store_attrs['run_batches'], config_id,
            #                                          store_attrs['run_batches'][config_id] + [b])
            try:
                store.put(key=batch_table,
                          value=pd.DataFrame(b_org, index=[0]),
                          append=True,
                          format='table',
                          data_columns=True)
            except ValueError as e:
                log_lines("*" * 10, e, f"ValueError writing to batch_table: {batch_table}",
                          "will read in entire table and re-write",
                          level="error")

                bt_tmp = store.get(key=batch_table)
                bt_tmp = pd.concat([bt_tmp, pd.DataFrame(b_org, index=[0])])
                store.put(key=batch_table,
                          value=bt_tmp,
                          append=False,
                          format='table',
                          data_columns=True)

            t1 = time.time()
            print(f"time to update attributes (and batch table): {t1-t0:.3f}")


    print(f"read_and_store.py finished, output file is:\n{full_path}")
