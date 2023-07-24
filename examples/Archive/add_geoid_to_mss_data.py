# add a geoid height column to MSS data

import re
import pandas as pd

from GPSat import get_data_path
from GPSat.decorators import timer

import pygeodesy

pd.set_option("display.max_columns", 200)

# --
# helper functions
# --

@timer
def get_multi_height(lat, lon):
    return ginterpolator.height(lat, lon)

# ---
# parameters
# ---

mss_file = get_data_path("MSS", "CryosatMSS-arco-2yr-140821.txt")
out_file = re.sub("\.txt$", "_with_geoid_h.csv", mss_file)

pgm_file = get_data_path("egm", "geoids/egm2008-5.pgm")

ginterpolator = pygeodesy.GeoidKarney(pgm_file)

# --
# read in mss data
# --
df = pd.read_csv(mss_file, header=None, sep="\s+", names=['lon', 'lat', 'mss'])

# for some reason lon, lat are not being read in as floats?
df['lon'] = df['lon'].astype(float)
df['lat'] = df['lat'].astype(float)

# ---
# get the geoid height
# ---

lat, lon = df['lat'].values, df['lon'].values
# for 15M locations this can take ~ 30min
h = get_multi_height(lat, lon)

df['h'] = h

df.to_csv(out_file, index=False)

