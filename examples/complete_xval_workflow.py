#%%
import os
import re
import warnings
import time
import json

import numpy as np
import pandas as pd
import tensorflow as tf

from GPSat import get_parent_path, get_data_path
from GPSat.local_experts import LocalExpertOI
from GPSat.utils import get_config_from_sysargv, nested_dict_literal_eval, cprint
from GPSat.models import get_model

from GPSat.postprocessing import smooth_hyperparameters, SmoothingConfig
from GPSat.config_dataclasses import (DataConfig, 
                                                       ModelConfig,
                                                       PredictionLocsConfig,
                                                       ExpertLocsConfig,
                                                       RunConfig,
                                                       ExperimentConfig)
from GPSat.postprocessing import glue_local_predictions
from examples.create_xval_config import *


#%%
# Set up configs
# config = get_xval_input_config()
f = open('../configs/xval_reference_config.json')
config = json.load(f)
f.close()

data_config = DataConfig.from_dict(config['data'])
model_config = ModelConfig.from_dict(config['model'])
expert_locs_config = ExpertLocsConfig.from_dict(config['locations'])

data_config.data_source = '/home/so/Documents/Projects/GPSat/data/example/ABC_50km_binned_by_track.h5'
config['data'] = data_config.to_dict()

xval_config = config.pop("xval")
xc = XvalConfig(ref_config=config,
                xval_config=xval_config)

oi_config_list = xc.make_xval_oi_configs(hold_out_data_is_pred_loc=xc.hold_out_data_is_pred_loc,
                                         add_where_as_col_to_location=True,
                                         add_to_table_suffix=True,
                                         verbose=True)

#%%
# Train model (no prediction)
store_path = get_parent_path("results", "example", "complete_xval_50km_test.h5")
predict = False

# increment over the list of configs
t1 = time.time()
for config_count, config in enumerate(oi_config_list):
    locexp = LocalExpertOI(expert_loc_config=config['locations'],
                           data_config=config["data"],
                           model_config=config["model"],
                           pred_loc_config=config['pred_loc'])
    table_suffix = config['run_kwargs']['table_suffix']
    locexp.run(store_path=store_path,
               predict=predict,
               table_suffix=table_suffix)

t2 = time.time()
print(f"Total training time: {t2 - t1:.2f} seconds")


#%%
# Smooth hyperparameters
store_path = get_parent_path("results", "example", "complete_xval_50km_test.h5")
params_to_smooth = ['lengthscales', 'kernel_variance', 'likelihood_variance']
with pd.HDFStore(store_path, "r") as f:
    run_details_w_suffix = [s for s in f.keys() if re.search('run_details', s)]

for config in oi_config_list:
    table_suffix = config['run_kwargs']['table_suffix']
    if '/run_details' + table_suffix in run_details_w_suffix:
        smooth_hyperparameters(result_file=store_path,
                               params_to_smooth=params_to_smooth,
                               l_x = 200_000,
                               l_y = 200_000,
                               reference_table_suffix=table_suffix,
                               save_config_file=False)


# %%
# Predict on held-out tracks
store_path = get_parent_path("results", "example", "complete_xval_50km_test.h5")
results_dir = get_parent_path("results", "example", "complete_xval_50km_test_preds.h5")
raw_data_source = get_data_path("example", "ABC.h5")
with pd.HDFStore(store_path, "r") as f:
    run_details_w_suffix = [s for s in f.keys() if re.search('run_details', s)]

for config in oi_config_list:
    # Set prediction locations to raw data
    config['pred_loc']['load_kwargs']['table'] = '/data_w_tracks'
    config['pred_loc']['load_kwargs']['source'] = raw_data_source

    # Load smoothed parameters
    ref_table_suffix = config['run_kwargs']['table_suffix']
    table_suffix = ref_table_suffix + "_SMOOTHED"
    config['model']['load_params'] = {"file": store_path, "table_suffix": table_suffix}

    if '/run_details' + ref_table_suffix in run_details_w_suffix:
        # Make predictions on held-out locations
        locexp = LocalExpertOI(expert_loc_config=config['locations'],
                               data_config=config['data'],
                               model_config=config['model'],
                               pred_loc_config=config['pred_loc'])
        
        locexp.run(store_path=results_dir,
                   predict=True,
                   optimise=False,
                   table_suffix=table_suffix)


# %%
