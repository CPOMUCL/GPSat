# read binned data
# provide to GP Model
# optimise hyper parameters
# make predictions
# extract hyper parameters

import os
import time

import numpy as np
import xarray as xr
import pandas as pd

from scipy.spatial import KDTree

t0 = time.time()
from PyOptimalInterpolation import get_parent_path
from PyOptimalInterpolation.utils import WGS84toEASE2_New
from PyOptimalInterpolation.models import GPflowGPRModel
t1 = time.time()

print(f"{t1-t0:.2f}s to read in custom packages")

# ---
# parameters
# ---

days_ahead = 4
days_behind = 4
incl_rad = 300 * 1000

verbose = False

# ---
# read data
# ---

# TODO: specify global select - for loading into memory

data_dir = get_parent_path("data", "binned")
nc_file = os.path.join(data_dir, "gpod_202003.nc")

# connect to Dataset
ds = xr.open_dataset(nc_file)

# get the coordinates
ds_coords = ds.coords

# NOTE: binned data expected to contain (y,x,date) values

# select dates (as 'day' and convert them to integers - days since 1970-01-01)
dates = ds_coords['date'].data.astype('datetime64[D]')
dates_int = dates.astype(int)

# ----
# load to memory
# ----

# TODO: allow for options to read from disk only - reading all could be memory intensive
# TODO: could use xr.concat or xr.merge for this?
tmp = []
for sat in ds:
    _ = ds[sat].to_dataframe().dropna()
    _.rename(columns={sat: "obs"}, inplace=True)
    _ = _.reset_index()
    tmp += [_]

df = pd.concat(tmp)

# create coord columns that will be used
# - could use map if this turns out to be slow
df['t'] = df['date'].values.astype('datetime64[D]').astype(int)

# ----
# select data
# ----


# --
# local selection
# --

# for location selection

# location where GP will be centered at
reference_location = {
    "lon": df['lon'].median(),
    "lat": df['lat'].median(),
    "date": df['date'].median()
}

# convert general coordinates
# TODO: allow for flexible transforming of reference location - using a config?
# ref_x_fun = lambda lon, lat: WGS84toEASE2_New(lon, lat)[0]
# ref_y_fun = lambda lon, lat: WGS84toEASE2_New(lon, lat)[1]
ref_date_fun = lambda date: np.datetime64(date, 'D').astype(int)

ref_x, ref_y = WGS84toEASE2_New(reference_location["lon"], reference_location["lat"])
reference_location['x'] = ref_x
reference_location['y'] = ref_y
reference_location['t'] = ref_date_fun(reference_location['date'])


# location selection will be made always relative to a reference location
# - the correspoding reference location will be add to "val" (for 1-d)
# - for 2-d coordindates (a combination), KDTree.ball_query will be used
local_select = [
    {"col": "t", "comp": "<=", "val": days_ahead},
    {"col": "t", "comp": ">=", "val": -days_behind},
    {"col": ["x", "y"], "comp": "<", "val": incl_rad}
]

# use a bool to select values
select = np.ones(len(df), dtype='bool')

# increment over each of the selection criteria
for ls in local_select:
    col = ls['col']
    comp = ls['comp']
    if verbose:
        print(ls)

    # single (str) entry for column
    if isinstance(col, str):
        assert col in df, f"col: {col} is not in data - {df.columns}"
        assert col in reference_location, f"col: {col} is not in reference_location - {reference_location.keys()}"
        assert comp in [">=", ">", "==", "<", "<="], f"comp: {comp} is not valid"

        tmp_fun = lambda x, y: eval(f"x {comp} y")
        _ = tmp_fun(df.loc[:, col], reference_location[col] + ls['val'])
        select &= _
    else:
        assert comp in ["<", "<="], f"for multi dimensional values only less than comparison handled"
        for c in col:
            assert c in df
            assert c in reference_location
        kdt = KDTree(df.loc[:, col].values)
        in_ids = kdt.query_ball_point(x=[reference_location[c] for c in col],
                                      r=ls['val'])
        # create a bool array of False, then populate locations with True
        _ = np.zeros(len(df), dtype=bool)
        _[in_ids] = True
        select &= _

# data to be used by a local model
df_sel = df.loc[select, :]

print("local data selection")

# quick check on distance
# dx = df_sel.loc[:, "x"].values - reference_location['x']
# dy = df_sel.loc[:, "y"].values - reference_location['y']
# np.sqrt(dx**2 + dy**2).max() / 1e3

# -----
# provide to model
# -----

gpr_model = GPflowGPRModel(data=df_sel,
                           obs_col='obs',
                           coords_col=['x', 'y', 't'],
                           coords_scale=[50000, 50000, 1])

# initial hyper parameters
hyp0 = gpr_model.get_hyperparameters()

# initial marginal log likelihood
mll0 = gpr_model.get_marginal_log_likelihood()

# make a prediction with default parameters
pred0 = gpr_model.predict(coords=[reference_location[_] for _ in gpr_model.coords_col])

# optimise hyper parameters
hyp1 = gpr_model.optimise_hyperparameters()

# optimised marginal log likelihood
mll1 = gpr_model.get_marginal_log_likelihood()

# new prediction
pred1 = gpr_model.predict(coords=[reference_location[_] for _ in gpr_model.coords_col],
                          full_cov=False)

# ---
# apply constraints - to length scales
# ---

# get current lengthscale
ls = gpr_model.model.kernel.lengthscales.numpy()

# set upper and lower bound for length scales
low = np.zeros(len(ls))
high = ls.mean() * np.ones(len(ls))

# any length scales outside of [low,high] range will be moved to be within range
# - by some amount of tol. not doing so will raise an error
gpr_model.set_lengthscale_constraints(low=low, high=high, move_within_tol=True, tol=1e-8, scale=False)

# get new values
mll2 = gpr_model.get_marginal_log_likelihood()
hyp2 = gpr_model.get_hyperparameters()
pred2 = gpr_model.predict(coords=[reference_location[_] for _ in gpr_model.coords_col],
                          full_cov=False)

# optimise
gpr_model.optimise_hyperparameters()
gpr_model.get_marginal_log_likelihood()

# pred3 = gpr_model.predict(coords=[reference_location[_] for _ in gpr_model.coords_col],
#                           full_cov=False)

# ---
# assign hyper parameters
# ---

gpr_model.model.kernel.lengthscales.assign(np.min(ls) * np.ones(len(ls)))



# re-run optimisation with constraints

# TODO: add method for making parameters / variables trainable






