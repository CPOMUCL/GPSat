# read example raw data from package and store to hdf5 file

import datetime
import os
import json

import pandas as pd

from GPSat import get_data_path, get_config_path
from GPSat.dataloader import DataLoader

pd.set_option("display.max_columns", 200)

# --
# (example) data path
# --

data_dir = get_data_path("example")

# --
# configuration file to read data
# --

config_file = get_config_path("example_read_and_store_raw_data.json")

with open(config_file, "r") as f:
    config = json.load(f)

# change some of the directory locations to the package
config['output']['dir'] = data_dir
config['file_dirs'] = data_dir

print("reading raw data and storing to hdf file using config:")
print(json.dumps(config, indent=4))

# extract (pop out) the output information
output_dict = config.pop("output", None)

# --
# read in data, select rows and columns, combine into a single dataframe
# --

df = DataLoader.read_flat_files(**config)

print("read in raw data, looks like:")
df.head(5)

# --
# store as hdf5
# --

# get run information (including some details from git)
# - for auditing / future reference
# run_info = DataLoader.get_run_info()
run_info = {
    "run_time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
}

# ---
# write to file
# ---

output_dir = output_dict['dir']
out_file = output_dict['file']
table = output_dict['table']
append = output_dict.get("append", False)

print("writing to hdf5 file")
with pd.HDFStore(path=os.path.join(output_dir, out_file), mode='a' if append else 'w') as store:
    DataLoader.write_to_hdf(df,
                            table=table,
                            append=append,
                            store=store,
                            config=config,
                            run_info=run_info)

