# run regression to compare previously generated results with current
import json
import os
import re
import numpy as np
import pandas as pd

# change tensorflow warning levels(?)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

import tensorflow as tf

from GPSat import get_parent_path, get_data_path
from GPSat.local_experts import LocalExpertOI
from GPSat.utils import cprint
from GPSat.local_experts import get_results_from_h5file


gpu_list = tf.config.list_physical_devices('GPU')
print("GPUs:", gpu_list)

pd.set_option("display.max_columns", 200)

# ---
# helper function
# ---

def replace_str_in_dict(d, pattern, replace):
    # to replace certain patterns of str values in a (nested) dictionary)
    out = {}
    for k, v in d.items():
        if isinstance(v, dict):
            out[k] = replace_str_in_dict(v, pattern, replace)
        elif isinstance(v, str):
            out[k] = re.sub(pattern, replace, v)
        else:
            out[k] = v
    return out


# -----
# Get the previously generated results
# -----

gpu_or_cpu = "CPU" if len(gpu_list) == 0 else "GPU"
previous_results = get_parent_path("results", "example", f"ABC_binned_example_FOR_INTEGRATION_TEST_{gpu_or_cpu}.h5")

assert os.path.exists(previous_results), f"integration data file:\n{previous_results}\ndoes not exist. " \
                                         f"files for integration testings can be found:\n" \
                                         f"https://drive.google.com/drive/folders/1CFVkiKTJnnmSIwQf9m52BPkLJeMmmYRl?usp=sharing\n" \
                                         f"place the files in : GPSat/results/example/"


# ------
# load previously generate results and previously used configs
# ------

# read in previously generated results
dfs, oi_configs = get_results_from_h5file(previous_results)

if isinstance(oi_configs, dict):
    print("oi_configs from previous result came back as a dict, now expected a list (of dict), converting...")
    oi_configs = [oi_configs]

# replace certain paths to match current file system
# - override paths in configs - e.g. the config stored in "ABC_binned_example_FOR_INTEGRATION_TEST.h5"
pattern = "/home/buddy/workspace/pyOI/PyOptimalInterpolation"
replace = get_parent_path()
oi_configs = [replace_str_in_dict(oi_config, pattern, replace) for oi_config in oi_configs]


# where to store results
store_path = get_parent_path("results", "integration_tests", os.path.basename(previous_results))
if os.path.exists(store_path):
    cprint(f"store_path:\n{store_path}\nexists, removing now", "OKCYAN")
    os.remove(store_path)

# increment over the list of configs
for oi_config in oi_configs:

    # # misc
    misc = oi_config.get("run_kwargs", oi_config.get("misc", {}))
    # # store results after "store_every" expert locations have been optimised
    store_every = misc.get("store_every", 10)

    # --------
    # initialise LocalExpertOI object
    # --------

    locexp = LocalExpertOI(expert_loc_config=oi_config['locations'],
                           data_config=oi_config["data"],
                           model_config=oi_config["model"],
                           pred_loc_config=oi_config.get("pred_loc", None))

    # ----------------
    # Increment over the expert locations
    # ----------------

    locexp.run(store_path=store_path,
               store_every=store_every,
               check_config_compatible=False)


# --------------------
# Compare results
# --------------------

# default comparison levels
abs_diff = 1e-6
merge_col = ['x', 'y', 't']

dfs2, oi_config2 = get_results_from_h5file(store_path)


# TODO: specify a dict with table name and columns to compare
# TODO: for now make absolute diff comparisons, allow for overriding values


# chk columns - numeric comparison
# keys: table
chk_cols = {
    "kernel_variance": {
        "check_cols": ["kernel_variance"],
        "merge_cols": ["x", "y", "t"]
    },
    "likelihood_variance": {
        "check_cols": ["likelihood_variance"],
    },
    "lengthscales": {
        "check_cols": ["lengthscales"],
        "abs_diff": 0.001
    },
    "run_details": {
        "check_cols": ["num_obs", "run_time"]
    }
}

res = {}

bad_keys = {}
for k, v in chk_cols.items():

    res[k] = {}

    mc = v.get("merge_cols", merge_col)
    chk_col = v['check_cols']
    adiff = v.get("abs_diff", abs_diff)

    chk_col = chk_col if isinstance(chk_col, list) else [chk_col]

    vv = dfs[k]
    dim_cols = [c for c in vv.columns if re.search("^_dim_", c)]

    # TODO: here check columns, pop bad ones, record which

    for c in chk_col:

        if c not in dfs[k]:
            if k not in bad_keys:
                bad_keys[k] = {}
            if 'org' not in bad_keys[k]:
                bad_keys[k]['org'] = []
            bad_keys[k]['org'].append(c)

        if c not in dfs2[k]:
            if k not in bad_keys:
                bad_keys[k] = {}
            if 'new' not in bad_keys[k]:
                bad_keys[k]['new'] = []
            bad_keys[k]['new'].append(c)
    # TODO: remove any bad keys

    try:
        df0 = dfs[k][merge_col + dim_cols + chk_col]
    except KeyError as e:
        print(f"org data- bad cols in table: {k}")
        print(e)
        bad_keys[k] = {
            "org": [c for c in merge_col + dim_cols + chk_col if c not in dfs[k]],
            "new": [c for c in merge_col + dim_cols + chk_col if c not in dfs2[k]]
        }
        continue

    try:
        df1 = dfs2[k][merge_col + dim_cols + chk_col]
    except KeyError as e:
        print(f"new data - bad cols in table: {k}")
        bad_keys[k] = {
            "org": [c for c in merge_col + dim_cols + chk_col if c not in dfs[k]],
            "new": [c for c in merge_col + dim_cols + chk_col if c not in dfs2[k]]
        }
        continue


    chk = df0.merge(df1,
                    on=merge_col + dim_cols,
                    suffixes=["", "_new"],
                    how="outer")

    for c in chk_col:
        chk[c+"_diff"] = np.abs(chk[c] - chk[c+"_new"])

        tmp = chk.loc[chk[c+"_diff"] > adiff].copy(True)
        tmp = tmp[merge_col + dim_cols + [c, c+"_new", c+"_diff"]]
        tmp.sort_values(c+"_diff", inplace=True, ascending=False)

        if len(tmp) > 0:
            res[k][c] = tmp


# print(json.dumps(chk_cols, indent=4))

for k, v in res.items():
    print("-" * 50)
    print(f"differences for table: {k}\n(won't show anything if no differences)")
    for kk, vv in v.items():
        print("-" * 25)
        print(f"column: {kk}")
        print("-" * 10)
        cprint(f"differences found: {len(vv)} entries", c="FAIL")
        print(vv)

