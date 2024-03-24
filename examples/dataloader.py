# %% [markdown]
##  DataLoader Examples
# %%
import os
import re
import numpy as np
import pandas as pd
import xarray as xr

from GPSat.dataloader import DataLoader
from GPSat import get_data_path
from GPSat.utils import cprint

pd.set_option("display.max_columns", 200)

# %% [markdown]
### load method

# read from csv
# %%

cprint("-"*20, "BOLD")
cprint("loading individual csv files", "BOLD")

# load data, engine to use is determined by file name
a = DataLoader.load(source=get_data_path("example", "A_RAW.csv"))
print(a.head(2))

# specify engine, in this case a panda's read_* method
b = DataLoader.load(source=get_data_path("example", "B_RAW.csv"),
                    engine="read_csv")

# provide additional arguments to read_csv
c = DataLoader.load(source=get_data_path("example", "C_RAW.csv"),
                    engine="read_csv",
                    source_kwargs={"sep": ','})

# %% [markdown]
### load method con't

# save tab seperated file
# %%

cprint("-"*20, "BOLD")
cprint("loading from tab separated file", "BOLD")

# (store and) read from tab seperated
tsv_file = get_data_path("example", "tmp.tsv")
c.to_csv(tsv_file, sep="\t", index=False)

# load tsv file, providing additional (source keyword) arguments
# - which will be passed into pd.read_csv
_ = DataLoader.load(source=tsv_file,
                    engine="read_csv",
                    source_kwargs={"sep": "\t", "keep_default_na": True})

pd.testing.assert_frame_equal(c, _)


# %% [markdown]
### read in multiple files in a folder

# identify files using a regular expression
# apply col_funcs after loaded into memory - see below for details on col_funcs
# %%

col_funcs = {
    "source": {
        "func": lambda x: re.sub('_RAW.*$', '', os.path.basename(x)),
        "filename_as_arg": True
    },
    "datetime": {
        "func": lambda x: x.astype('datetime64[ns]'),
        "col_args": "datetime"
    }
}

# can also supply a list of sub-directories (sub_dirs) to look instead of file_dirs
_ = DataLoader.read_flat_files(file_dirs=get_data_path("example"),
                               file_regex="_RAW\.csv$",
                               col_funcs=col_funcs)

cprint("head:", "BOLD")
print(_.head(2))

cprint("dtypes:", "BOLD")
print(_.dtypes)

# %% [markdown]
### save and read from parquet file
# %%
cprint("-"*20, "BOLD")
cprint("loading from parquet file", "BOLD")
# store as h5 file - to demonstrate reading in
parq_tmp = get_data_path("example", "tmp.parquet")

# NOTE: fastparquet allows for appending, alternative is pyarrow (or 'auto')
_.to_parquet(parq_tmp, engine="fastparquet")

# read data from parquet file - using the pyarrow engine
df = DataLoader.load(source=parq_tmp)

pd.testing.assert_frame_equal(df, _)


# %% [markdown]
### save and read from hdf5 file
# %%
cprint("-"*20, "BOLD")
cprint("loading from hdf5 file", "BOLD")
# store as h5 file - to demonstrate reading in
hdf5_tmp = get_data_path("example", "tmp.h5")
hdf5_table = "data"
with pd.HDFStore(hdf5_tmp, mode="w") as store:
    # setting data_columns = True so will be searchable
    store.append(key=hdf5_table, value=_, data_columns=True)

# read data from table in hdf5
df = DataLoader.load(source=hdf5_tmp,
                     table=hdf5_table)

pd.testing.assert_frame_equal(df, _)

# %% [markdown]
### save and read from netcdf file
# %%

cprint("-"*20, "BOLD")
cprint("loading from netcdf file", "BOLD")

# first - save to file, using multi index for dimensions
tmp_ = _.set_index(['datetime', 'source'])
ds = xr.Dataset.from_dataframe(tmp_)

netcdf_tmp = get_data_path("example", "tmp.nc")
ds.to_netcdf(path=netcdf_tmp)

# read data from netcdf file
# NOTE: this will comeback with a multi index - can reset with reset_index=True
nc = DataLoader.load(source=netcdf_tmp)

# netcdf will have nans for missing values
# - netcdf effectively stores values in n-d array with the
# - dimensions determined by an index when converting from DataFrame
# nc.dropna(inplace=True)

# sort indices (and data) in the same way
tmp_.sort_index(inplace=True)
nc.sort_index(inplace=True)

# ensure columns are in same order - might not be needed
nc = nc[tmp_.columns]

pd.testing.assert_frame_equal(tmp_, nc)

# %% [markdown]
### use 'where' to select subset without having to read entirely into memory

# this is accomplished by using a 'where dict', containing the following keys
# - 'col' : the column of the data used for selection
# - 'comp': the comparison to used, e.g. ">", ">=", "==", "!=", "<=", "<"
# - 'val' : value being compared to column values

# where using hdf5
# %%

a = _.loc[_['source'] == "A"].copy(True)

cprint("-"*20, "BOLD")
cprint("applying 'where' dictionaries - selecting data at source", "BOLD")

df = DataLoader.load(source=hdf5_tmp,
                     table=hdf5_table,
                     where={"col": "source", "comp": "==", "val": "A"})

pd.testing.assert_frame_equal(a, df)

# hdf5 allows for a list of values to be provided
df = DataLoader.load(source=hdf5_tmp,
                     table=hdf5_table,
                     where={"col": "source", "comp": "==", "val": ["A", "B"]}
                     )

# unique doesn't sort?
np.testing.assert_array_equal(np.sort(df['source'].unique()),
                              np.array(['A', 'B'], dtype=object))


# multiple 'where dicts' can be combined in a list
# - they will be combined with an AND operation
df = DataLoader.load(source=hdf5_tmp,
                     table=hdf5_table,
                     where=[
                        {"col": "source", "comp": "==", "val": "A"},
                        {"col": "lat", "comp": ">=", "val": 65.0}
                     ]
                     )

assert df['lat'].min() >= 65.0

pd.testing.assert_frame_equal(a.loc[a['lat'] >= 65.0], df)

# %% [markdown]
# where using netcdf files
# %%

# apply where_dict to netcdf file - must be for the index, in thise case "source" or "datetime"
nc = DataLoader.load(source=netcdf_tmp,
                     where=[
                         {"col": "source", "comp": "==", "val": "A"}
                     ],
                     reset_index=True
                     )

np.testing.assert_array_equal(nc['source'].unique(), np.array(['A'], dtype=object))


# %% [markdown]
# where using parquet
# %%

# using parquet / read_parquet the where's get convert to 'filters', see pd.read_parquet for more details

df = DataLoader.load(source=parq_tmp,
                     where=[
                        {"col": "source", "comp": "==", "val": "A"},
                        {"col": "lat", "comp": ">=", "val": 65.0}
                     ]
                     )
assert df['lat'].min() >= 65.0

pd.testing.assert_frame_equal(a.loc[a['lat'] >= 65.0], df)


# NOTE: using querying on datetime / timestamp objects need to convert to datetime via -
# pd.Timestamp

max_time = pd.Timestamp("2020-03-05")
df = DataLoader.load(source=parq_tmp,
                     where=[
                        {"col": "datetime", "comp": "<=", "val": max_time}
                     ]
                     )

pd.testing.assert_frame_equal(_.loc[_['datetime'] <= max_time], df)


# %% [markdown]
### use 'row_select' to select subset after data is loaded into memory
# %%


cprint("-"*20, "BOLD")
cprint("row_select examples", "BOLD")



# 'where dict' can be used for row_select
df0 = DataLoader.load(source=hdf5_tmp,
                      table=hdf5_table,
                      row_select={"col": "source", "comp": "==", "val": "A"})

# NOTE: using where is faster
df1 = DataLoader.load(source=hdf5_tmp,
                      table=hdf5_table,
                      where={"col": "source", "comp": "==", "val": "A"})

pd.testing.assert_frame_equal(df0, df1)

# row select allows for using lambda functions that returns a bool array
# - col_args specify the columns of the data to pass in as arguments to "func"
df0 = DataLoader.load(source=hdf5_tmp,
                      table=hdf5_table,
                      row_select={
                          "func": lambda x: x >= 65.0,
                          "col_args": "lat"
                      })

assert df0['lat'].min() >= 65.0

# the lambda functions can be supplied as strings - useful when passing parameters from a json configuration
# NOTE: if func is a string it will be converted with eval(...)
df1 = DataLoader.load(source=hdf5_tmp,
                      table=hdf5_table,
                      row_select={
                          "func": "lambda x: x >= 65.0",
                          "col_args": "lat"
                      })

pd.testing.assert_frame_equal(df0, df1)

# %% [markdown]
### Advanced: more row_select and where
# %%

cprint("-"*20, "BOLD")
cprint("more row_select and where", "BOLD")


# multiple columns can be supplied
df2 = DataLoader.load(source=hdf5_tmp,
                      table=hdf5_table,
                      row_select={
                          "func": "lambda x, y: (x >= 65.0) & (y == 'A')",
                          "col_args": ["lat", "source"]
                      })

assert df2['lat'].min() >= 65.0
np.testing.assert_array_equal(df2['source'].unique(), np.array(['A'], dtype=object))


# column values can be supplied via col_kwargs
# - this can be useful if a more involved function is supplied
df2 = DataLoader.load(source=hdf5_tmp,
                      table=hdf5_table,
                      row_select={
                          "func": "lambda x, y: (x >= 65.0) & (y >= 0)",
                          "col_kwargs": {
                              "x": "lat", "y": "lon"
                          }
                      })

assert df2['lat'].min() >= 65.0
assert df2['lon'].min() >= 0.0

# row_select can be negated (or inverted) - flipping Trues to False and vice versa
# - this can be useful when defining hold out data
# - e.g. create a row_select that selects the desired data and then use negate to make sure it's excluded
df3 = DataLoader.load(source=hdf5_tmp,
                      table=hdf5_table,
                      row_select={
                          "func": "lambda x: (x >= 65.0)",
                          "col_args": "lat",
                          "negate": True
                      })

assert df3['lat'].max() < 65.0


df3 = DataLoader.load(source=hdf5_tmp,
                      table=hdf5_table,
                      row_select={
                          "func": "lambda x, y: (x >= 65.0) & (y >= 0)",
                          "col_kwargs": {
                              "x": "lat", "y": "lon"
                          },
                          "negate": True
                      })

# there should be no rows with lat>=65.0  AND lon>=0
assert len(df3.loc[(df3['lat'] >= 65.0) & (df3['lon'] >= 0.0)]) == 0

# multiple row_selects can be combined via a list
# - similar to where, they are combined via an AND boolean operation


df4 = DataLoader.load(source=hdf5_tmp,
                      table=hdf5_table,
                      row_select=[
                          {"col": "source", "comp": "==", "val": "A"},
                          {
                              "func": "lambda x: (x >= 65.0)",
                              "col_args": "lat"
                          },
                          {
                              "func": "lambda y: y >= 0.0",
                              "col_kwargs": {"y": "lon"}

                          },
                          {
                              "func": "lambda y: y >= 0.0",
                              "col_args": "z"

                          }
                      ])

assert df4['lat'].min() >= 65.0
assert df4['lon'].min() >= 0.0
assert df4['z'].min() >= 0.0
np.testing.assert_array_equal(df4['source'].unique(), np.array(['A'], dtype=object))


# where and row_selects can be used together
# - where's are used first, when reading data in from file
# - row_selects are applied to the data in memory
df5 = DataLoader.load(source=hdf5_tmp,
                      table=hdf5_table,
                      where={"col": "source", "comp": "==", "val": "A"},
                      row_select={"col": "source", "comp": "==", "val": "B"})
assert len(df5) == 0

# TODO: show where and row_select using netcdf data

# %% [markdown]
### Advanced: col_funcs - apply functions to create or modify columns

# columns functions take in a dict, with the key being the new (or existing) column and the value
# - a dict specifying how the column shall be created
# - NOTE: by default columns are extracted from dataframe columns as numpy arrays (so no need to take values)

# %%

cprint("-"*20, "BOLD")
cprint("applying col(umn)_func(tion)s to data after reading into memory", "BOLD")


# add a column
df1 = DataLoader.load(source=hdf5_tmp,
                      table=hdf5_table,
                      col_funcs={
                          "z_pos": {
                              "func": "lambda x: np.where(x>0, x, np.nan)",
                              "col_args": "z"
                          }
                      })
assert np.all(df1.loc[np.isnan(df1['z_pos']), "z"] <= 0)


# modify / create a 'date' column
df1 = DataLoader.load(source=hdf5_tmp,
                      table=hdf5_table,
                      col_funcs={
                          "date": {
                              "func": "lambda x: x.astype('datetime64[D]')",
                              "col_args": "datetime"
                          }
                      })

# reference dataframe: column modified after load
df0 = DataLoader.load(source=hdf5_tmp,
                      table=hdf5_table)

# convert datetime from
df0['date'] = df0['datetime'].values.astype('datetime64[D]')

pd.testing.assert_frame_equal(df0, df1)


# %% [markdown]
# add_data_to_col, col_select, reset_index
# %%
# TODO: add simple examples of above

# %% [markdown]
# remove tmp files
# %%

cprint("-"*20, "BOLD")
cprint("cleaning up: delete tmp files", "BOLD")

# delete tmp files
for i in [netcdf_tmp, hdf5_tmp, parq_tmp, tsv_file]:
    print(f"removing tmp file: {i}")
    os.remove(i)
    assert not os.path.exists(i)

