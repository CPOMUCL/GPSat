# read binned data
# select local data - relative to some reference location
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


from PyOptimalInterpolation import get_parent_path
from PyOptimalInterpolation.utils import WGS84toEASE2_New
from PyOptimalInterpolation.models import GPflowGPRModel
from PyOptimalInterpolation.dataloader import DataLoader


# ---
# parameters
# ---

# netCDF containing binned observations
# nc_file = get_parent_path("data", "binned", "gpod_202003.nc")
nc_file = get_parent_path("data", "binned", "sats_ra_cry_processed_arco.nc")

# columns containing observations and coordinates
obs_col = "elev_mss"
coords_col = ['x', 'y', 't']

# parameters for location selection
days_ahead = 4
days_behind = 4
incl_rad = 300 * 1000

verbose = False

# ---
# read data
# ---

# connect to Dataset
ds = xr.open_dataset(nc_file)

# ---
# prep data
# ---

# convert to a DataFrame - dropping missing
df = ds.to_dataframe().dropna().reset_index()

# add columns that will be used as coordinates
# - this could be done with DataLoader.add_col
df['t'] = df['date'].values.astype('datetime64[D]').astype(int)

# ---
# data select for local expert
# ---

# require reference locations (i.e. center for local expert)
ref_locs = pd.DataFrame({"date": df['date'].median(),
                         "lon": df['lon'].median(),
                         "lat": df['lat'].median()},  index=[0])

# convert column(s) to be in appropriate coordinate space
# - again could use add_col here instead
ref_locs['t'] = ref_locs['date'].values.astype('datetime64[D]').astype(int)
ref_locs['x'], ref_locs['y'] = WGS84toEASE2_New(ref_locs['lon'], ref_locs['lat'])

rl = ref_locs.iloc[0, :]

# local selection criteria
# - points within the vals of the reference location will be selected
local_select = [
    {"col": "t", "comp": "<=", "val": days_ahead},
    {"col": "t", "comp": ">=", "val": -days_behind},
    {"col": ["x", "y"], "comp": "<", "val": incl_rad}
]

# select local data
df_local = DataLoader.local_data_select(df,
                                        reference_location=rl,
                                        local_select=local_select)

# -----
# provide to model
# -----

# initialise model
gpr_model = GPflowGPRModel(data=df_local,
                           obs_col=obs_col,
                           coords_col=coords_col,
                           coords_scale=[50000, 50000, 1])

# ---
# optimise / predict / get hyper params
# ---

# initial hyper parameters
hyp0 = gpr_model.get_hyperparameters()

# initial marginal log likelihood
mll0 = gpr_model.get_marginal_log_likelihood()

# make a prediction with default parameters
# - can pass reference location (pd.Series) - and the coords_col values will be selected
pred0 = gpr_model.predict(coords=rl)

# optimise hyper parameters
hyp1 = gpr_model.optimise_hyperparameters()

# optimised marginal log likelihood
mll1 = gpr_model.get_marginal_log_likelihood()

# new prediction
pred1 = gpr_model.predict(coords=rl,
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
pred2 = gpr_model.predict(coords=rl,
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

