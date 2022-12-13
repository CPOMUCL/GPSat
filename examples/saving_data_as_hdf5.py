
import os

import numpy as np
import pandas as pd
import xarray as xr

from PyOptimalInterpolation import get_parent_path
from PyOptimalInterpolation.dataloader import DataLoader

# ---
# parameters
# ---

# file location
out_dir = get_parent_path("results", "toy_example")
os.makedirs(out_dir, exist_ok=True)

out_path = os.path.join(out_dir, "toy.h5")

# for this example remove file if it exists
if os.path.exists(out_path):
    print(f"removing: {out_path}")
    os.remove(out_path)

# ---
# toy data
# ---

# index (location) values
index_names = ['x', 'y', 't']
indices = [(-12, 10, 4.0), (30, -2, 3.0)]

# name for data
data_names = ["c", "d"]

# specify the shape of the data for a given location - arbitrary
dummy_shapes = {"c": (2, 3, 4), "d": (5, 5)}

# TODO: tidy this up better
# get existing (multi-index) of the data - so can skip
existing_indices = {}
for data_name in data_names:

    try:
        with pd.HDFStore(out_path, mode="r") as store:
            store_levels = store.get_storer(data_name).levels
            store_midx = store.select(data_name, columns=[])
            store_midx = store_midx.index.unique() #.to_numpy()
            existing_indices[data_name] = store_midx
    except OSError as e:
        print(e)

# ---
# for index (location) - store data
# ---

for i, idx in enumerate(indices):

    # TODO: here should check if data has been stored at location
    # - there shouldn't be an issue storing duplicate entries, however may (will?) cause issue with DataArray

    # check if point already exists
    # - in practice this should be done at
    if data_names[0] in existing_indices:
        tmp_midx = existing_indices[data_names[0]]
        if tmp_midx.isin([idx]).any():
            print(f"skipping index: {idx} because already exists")
            continue

    # ---
    # generate data - store in DataArray
    # ---

    param_dict = {}
    for data_name in data_names:
        shape = dummy_shapes[data_name]
        param_dict[data_name] = np.random.random(shape)

    # ---
    # store DataArray to hdf5 via DataFrame
    # ---

    idx_dict = {index_names[_]: i for _, i in enumerate(idx)}
    DataLoader.store_to_hdf_table_w_multiindex(idx_dict=idx_dict,
                                               out_path=out_path,
                                               **param_dict)

# ---
# Recover Data
# ---

# extract data from hdf5, store as DataArrays with on dimension as multi-index e.g. (x,y,t)

# where condition
where = None

data_vars = {}
for data_name in data_names:

    with pd.HDFStore(out_path, mode="r") as store:
        # TODO: determine if it's faster to use select_colum - does not have where condition?
        df = store.select(data_name, where=where)

    data_vars[data_name] = DataLoader.mindex_df_to_mindex_dataarray(df, data_name=data_name)

# store in a Dataset
ds = xr.Dataset(data_vars)

# can select on indices - or components of indices
ds[data_names[0]].sel(index=indices[1])
ds[data_names[0]].sel(t=indices[1][-1])

