# create a location file from regularly spaced grid locations
# -

# TODO: wrap this into a method in LocalExpertOI - allow user to dynamically create expert locations on grid
#

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import cartopy.crs as ccrs

from PyOptimalInterpolation.local_experts import LocalExpertOI
from PyOptimalInterpolation.dataloader import DataLoader
from PyOptimalInterpolation import get_data_path
from PyOptimalInterpolation.utils import EASE2toWGS84_New
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

# ---
# build grid - bits of this were taken from DataLoader.bin_data
# ---

# x,y - min/max
x_min, x_max = x_range[0], x_range[1]
y_min, y_max = y_range[0], y_range[1]

# number of bin (edges)
n_x = ((x_max - x_min) / grid_res) + 1
n_y = ((y_max - y_min) / grid_res) + 1
n_x, n_y = int(n_x), int(n_y)

# # NOTE: x will be dim 1, y will be dim 0
x_edge = np.linspace(x_min, x_max, int(n_x))
y_edge = np.linspace(y_min, y_max, int(n_y))

# move from bin edge to bin center
x_cntr, y_cntr = x_edge[:-1] + np.diff(x_edge) / 2, y_edge[:-1] + np.diff(y_edge) / 2

# create a grid of x,y coordinates
x_grid, y_grid = np.meshgrid(x_cntr, y_cntr)

# store in DataFrame
df = pd.DataFrame({'x': x_grid.flatten(), 'y': y_grid.flatten()})

# add lon/lat
df['lon'], df['lat'] = EASE2toWGS84_New(df['x'], df['y'])


# write to file

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


locexp = LocalExpertOI(data_config=data_config)


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
