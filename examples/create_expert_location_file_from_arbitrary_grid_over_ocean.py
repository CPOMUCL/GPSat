# create a location file from regularly spaced grid locations
# - takes only locations where is_ocean is True

# TODO: wrap this into a method in LocalExpertOI - allow user to dynamically create expert locations on grid
#

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import cartopy.crs as ccrs

from GPSat.local_experts import LocalExpertOI
from GPSat.dataloader import DataLoader
from GPSat import get_data_path
from GPSat.utils import EASE2toWGS84, grid_2d_flatten
from GPSat.plot_utils import plot_pcolormesh, get_projection

# pip install global-land-mask
# - basically a look up table
from global_land_mask import globe

# ---
# parameters
# ---

# define the x,y range (on EASE2.0 projection for North Pole: (0,0) is pole
# - in meters
x_range = [-4500000.0, 4500000.0]
y_range = [-4500000.0, 4500000.0]

# grid resolution - i.e. grid cell width, expressed in meters
grid_res = 200 * 1000

# determine lat_0, min/max lat
pole = "south"

# where to write results?
out_file = get_data_path("locations", f"expert_locations_{int(grid_res//1000)}km_on_ocean_anto.csv")


# parameters for EASE2toWGS84_New
if pole == "north":
    lon_0, lat_0 = 0, 90
    min_lat = 60
    max_lat = None
elif pole == "south":
    lon_0, lat_0 = 0, -90
    min_lat = None
    max_lat = -57.5

# ---
# build grid - bits of this were taken from DataLoader.bin_data
# ---

xy_grid = grid_2d_flatten(x_range, y_range, step_size=grid_res)

# store in dataframe
df = pd.DataFrame(xy_grid, columns=['x', 'y'])

# add lon/lat
df['lon'], df['lat'] = EASE2toWGS84(df['x'], df['y'], lon_0=lon_0, lat_0=lat_0)

# check if point is over ocean
df['is_ocean'] = globe.is_ocean(lat=df['lat'], lon=df['lon'])

# keep only points over ocean
df = df.loc[df['is_ocean']]

# keep only points above some min_lat
if min_lat:
    df = df.loc[df['lat'] >= min_lat]
if max_lat:
    df = df.loc[df['lat'] <= max_lat]

# this just store the grid
# df.to_csv(get_data_path("locations", f"expert_locations_{int(grid_res//1000)}km_nx{n_x}_ny{n_y}.csv"), index=False)
df.to_csv(out_file, index=False)

# ---
# plot locations
# ---

figsize = (15, 15)

projection = get_projection(pole)
extent = [-180, 180, 60, 90] if pole == "north" else [-180, 180, -60, -90]


fig, ax = plt.subplots(figsize=figsize,
                       subplot_kw={'projection': projection})

print(f"there are now: {len(df)} expert locations")

# data needed for plotting
plot_data, lon, lat = np.ones(len(df)), df['lon'].values, df['lat'].values

stitle = ax.set_title(f"Expert Locations from Grid Res: {grid_res/1000:.1f}km\nWhere is_ocean is True")

plot_pcolormesh(ax,
                lon=lon,
                lat=lat,
                title=stitle,
                # vmin=vmin,
                # vmax=vmax,
                plot_data=plot_data,
                scatter=True,
                s=grid_res / 1000,
                extent=extent,
                # fig=fig,
                cmap='YlGnBu_r')


plt.show()
