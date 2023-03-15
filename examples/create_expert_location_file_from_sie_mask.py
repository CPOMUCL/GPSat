# create a file containing lon, lat, date derived from Sea Ice Extent (SIE)
# - will read data from SIE_masking_*_season.zarr file
# - - see 'read_data_from_legacy_pkl.py' for file creation


import os
import re

import xarray as xr
from PyOptimalInterpolation import get_data_path


# ---
# parameters
# ---

season = "2019-2020"
grid_res = "50km"
input_file = get_data_path("aux", "SIE", f"SIE_masking_{grid_res}_{season}_season.zarr")

assert os.path.exists(input_file), f"input_file:\n{input_file}\ndoes not exist, dir contains\n" \
                                   f"{os.listdir(os.path.dirname(input_file))}, " \
                                   f"try creating file via read_data_from_legacy_pkl.py"


# --
# connect to dataset
# --

ds = xr.open_dataset(input_file, engine="zarr")

# --
# convert to DataFrame
# --

df = ds.to_dataframe().dropna().reset_index()

print("data looks like", df.head(3))

# --
# write to file
# --

out_file = re.sub("\.zarr$", ".csv", input_file)
assert out_file != input_file, f"out_file:\n{out_file}\nis same as input_file:\n{input_file}"

print(f"saving as csv to:\n{out_file}")
df.to_csv(out_file, index=False)



