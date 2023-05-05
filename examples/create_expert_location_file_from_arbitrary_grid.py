# create a location file from regularly spaced grid locations
# -

# TODO: wrap this into a method in LocalExpertOI - allow user to dynamically create expert locations on grid
#

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import cartopy.crs as ccrs

try:
    from global_land_mask import globe
except ModuleNotFoundError:
    print("could not import global_land_mask, won't reduce grid points to just those over ocean")
    globe = None

from PyOptimalInterpolation.local_experts import LocalExpertOI
from PyOptimalInterpolation.dataloader import DataLoader
from PyOptimalInterpolation import get_data_path
from PyOptimalInterpolation.utils import EASE2toWGS84_New, grid_2d_flatten
from PyOptimalInterpolation.plot_utils import plot_pcolormesh

# ---
# parameters
# ---

# define the x,y range (on EASE2.0 projection for North Pole: (0,0) is pole
# - in meters
x_range = [-4500000.0, 4500000.0]
y_range = [-4500000.0, 4500000.0]

# grid resolution - i.e. grid cell width, expressed in meters
grid_res = 200 * 1000

lon_0, lat_0 = 0, 90


# ---
# build grid - bits of this were taken from DataLoader.bin_data
# ---

xy_grid = grid_2d_flatten(x_range, y_range, step_size=grid_res)

# store in dataframe
df = pd.DataFrame(xy_grid, columns=['x', 'y'])

# add lon/lat
df['lon'], df['lat'] = EASE2toWGS84_New(df['x'], df['y'], lon_0=lon_0, lat_0=lat_0)

# select only locations over water (not land)
if globe is not None:
    is_in_ocean = globe.is_ocean(df['lat'], df['lon'])
    df = df.loc[is_in_ocean]


# this just store the grid
# df.to_csv(get_data_path("locations", f"expert_locations_{int(grid_res//1000)}km_nx{n_x}_ny{n_y}.csv"), index=False)
df.to_csv(get_data_path("locations", f"expert_locations_{int(grid_res//1000)}km.csv"), index=False)

# -----
# (optional) reduce expert locations to where data
# -----

# - this is done due to the (current) absence of an 'ocean' mask, to remove expert locations over land
# - here we use some input data (data_config) and the above locations
# - for each location data is select (within local_select) and the number of observations is recorded
# - only locations with min_obs or more are stored

# min number of observations per expert location to keep it
min_obs = 1000

# parameters for local selection
days_ahead = 4
days_behind = 4
incl_rad = 300 * 1000

# add a date dimension
df['date'] = np.datetime64('2020-03-05')

# def date_shift_to_datetime(x,y):
#     np.datetime64(pd.to_datetime(x + (y + 1 if y > 0 else y) ), unit='D')


# Data Config - to be provided as input to LocalExpertOI
data_config = {
    "data_source": get_data_path("example", f"ABC.h5"),
    "table": "data",
    "col_funcs": {
        "x": {
            "source": "PyOptimalInterpolation.utils",
            "func": "WGS84toEASE2_New",
            "col_kwargs": {"lon": "lon", "lat": "lat"},
            "kwargs": {"return_vals": "x"}
        },
        "y": {
            "source": "PyOptimalInterpolation.utils",
            "func": "WGS84toEASE2_New",
            "col_kwargs": {"lon": "lon", "lat": "lat"},
            "kwargs": {"return_vals": "y"}
        },
        "t": {"func": "lambda x: x.astype('datetime64[s]').astype(float) / (24 * 60 * 60)", "col_args": "datetime"},
    },
    "obs_col": "obs",
    "coords_col": ['x', 'y', 't'],
    "local_select": [
        {"col": "t", "comp": "<=", "val": days_ahead},
        {"col": "t", "comp": ">=", "val": -days_behind},
        {"col": ["x", "y"], "comp": "<", "val": incl_rad}
    ],
    # (optional) - read in a subset of data from data_source (rather than reading all into memory)
    # "global_select": [
    #     # {"col": "lat", "comp": ">=", "val": 60},
    #     {"loc_col": "t",
    #      "src_col": "datetime",
    #      # - the if else used
    #      "func": "lambda x,y: np.datetime64(pd.to_datetime(x+(y+1 if y > 0 else y), unit='D'))"}
    # ]
}

location_config = {
    "df": df,
    "col_funcs": {
        # "t": {"func": "lambda x: x.astype('datetime64[s]').astype(float) / (24 * 60 * 60)", "col_args": "date"},
        "t": {"func": "lambda x: x.astype('datetime64[D]').astype(float)", "col_args": "date"},
    }
}


locexp = LocalExpertOI(data_config=data_config,
                       expert_loc_config=location_config)


# NOTE: the following was copied from LocalExpertOI.run()
# - for each expert location getting

xprt_locs = locexp.expert_locs

# create a dictionary to store result (DataFrame / tables)
store_dict = {}
prev_params = {}
count = 0
df, prev_where = None, None

obs_details = []
for idx in range(len(xprt_locs)):

    # TODO: use log_lines
    print("-" * 30)
    count += 1
    print(f"{count} / {len(xprt_locs)}")

    # select the given expert location
    rl = xprt_locs.iloc[[idx], :].copy(True)
    print(rl)

    # ----------------------------
    # (update) global data - from data_source (if need be)
    # ----------------------------

    df, prev_where = locexp._update_global_data(df=df,
                                                global_select=locexp.data.global_select,
                                                local_select=locexp.data.local_select,
                                                ref_loc=rl,
                                                prev_where=prev_where)

    # ----------------------------
    # select local data - relative to expert's location - from global data
    # ----------------------------

    # this can be a little bit slow due to kdtree being created each time ... have a think
    df_local = DataLoader.local_data_select(df,
                                            reference_location=rl,
                                            local_select=locexp.data.local_select,
                                            verbose=False)
    print(f"number obs: {len(df_local)}")

    rl['num_obs'] = len(df_local)
    obs_details += [rl]

# combine results
od = pd.concat(obs_details)


# ----
# plot to get an idea of where expert locations will be (i.e. where there will be enough data > min obs)
# ---


figsize = (15, 15)

fig, ax = plt.subplots(figsize=figsize,
                       subplot_kw={'projection': ccrs.NorthPolarStereo()})

# tmp = od.copy(True)
od = od.loc[od['num_obs'] >= min_obs]

print(f"there are now: {len(od)} expert locations")

# data needed for plotting
plot_data, lon, lat = od['num_obs'].values, od['lon'].values, od['lat'].values

plot_pcolormesh(ax,
                lon=lon,
                lat=lat,
                # vmin=vmin,
                # vmax=vmax,
                plot_data=plot_data,
                scatter=True,
                s=200,
                fig=fig,
                cbar_label="Num Obs within Local Select of Expert Location",
                cmap='YlGnBu_r')

local_select_str = ["".join([str(v) for v in ls.values()])
                    for ls in data_config['local_select']]

stitle = ax.set_title(f"Expert Locations\nwith: num obs > {min_obs}\n Local Select: {local_select_str}")

plt.show()

# --
# write results (again)
# ---

keep_col = [c for c in od.columns if c != 'num_obs']

od.to_csv(get_data_path("locations", f"expert_locations_{int(grid_res//1000)}km_subset.csv"),
          index=False)
