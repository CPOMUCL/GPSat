# assign (arbitrary) track number for raw observation data
# - this script is a work in progress
import os.path

import numpy as np
import pandas as pd
import numba as nb

import matplotlib.pyplot as plt

from PyOptimalInterpolation import get_data_path
from PyOptimalInterpolation.plot_utils import plot_pcolormesh, get_projection
from PyOptimalInterpolation.utils import WGS84toEASE2_New, cprint, diff_distance, \
    guess_track_num, get_config_from_sysargv, track_num_for_date
from PyOptimalInterpolation.dataloader import DataLoader

# ---
# helper function
# ---

def assign_tracks(_, delta_measure="dt", cut_off=2e6, plot_vals=True, plot_first=1000000, title="",
                  start_track=0):
    assert delta_measure in ["dt", "dr"]

    # _ = df  # .copy(True)

    # create a distance (space or time) to next observation
    x = _[['x', 'y']].values

    # distance in space
    _['dr'] = diff_distance(x, p=2, k=1)
    # distance in time
    _['dt'] = diff_distance(_['t'].values, p=2, k=1)

    # ------
    # visualise the deltas - in order to determine the delta cut off size for a new track
    # ------

    if plot_vals:
        print("visualise")
        # NOTE: time differences seem to be unreliable for
        # gpod_all_rows.h5
        # - would expect there to be a large jump - review how they a parsed
        dtime = _['datetime'][:plot_first]
        xmin, xmax = dtime.min(), dtime.max()

        plt.plot(dtime, _['dt'][:plot_first])
        plt.hlines(y=cut_off, xmin=xmin, xmax=xmax)
        plt.title(f"dt\n{title}")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

        plt.plot(dtime, _['dr'][:plot_first])
        plt.hlines(y=cut_off, xmin=xmin, xmax=xmax)
        plt.title(f"dr\n{title}")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

    # ---
    # id tracks
    # ---

    # based on time delta
    _['track'] = guess_track_num(_[delta_measure].values, cut_off, start_track).astype(int)

    # ---
    # add track by start_date: 'deals' with tracks spanning multiple dates
    # ---

    if add_by_track_date:
        # starting datetime of each track
        tsd = pd.pivot_table(_,
                             index=['track'],
                             values=['datetime'],
                             aggfunc='min').reset_index()
        tsd.rename(columns={"datetime": "start_date"}, inplace=True)
        tsd['start_date'] = tsd['start_date'].astype("datetime64[D]")

        # for each start_date get the track number
        tsd[['start_date', 'track']].drop_duplicates()

        sd = tsd['start_date'].values.astype('int')

        # track number for given start date
        tsd['track_start'] = track_num_for_date(sd).astype(int)

        _ = _.merge(tsd,
                    on=['track'],
                    how='left')

    return _


if __name__ == "__main__":

    # NOTE: currently this scripts expects data to have a 'datetime' column, along with some longitude and latitude

    # ----
    # read config
    # -----

    config = get_config_from_sysargv()

    # if config not supplied, use default
    if config is None:
        config = {
            "data_file": get_data_path("example", "ABC.h5"), # raw data file
            "table": "data", # table to read from
            "sort_by": ["datetime"],
            "add_by_track_date": False, # start counting tracks from 0 on each date
            "get_track_by": "sat",#"source", #
            "delta_measure": "dr", # valid values 'dt' (time), 'dr' (change in distance)
            "cut_off": 600_000, # a new track number when delta_measure is greater than cutoff
        }

    # ----
    # parameters
    # ----

    # TODO: be more memory efficient - copy less (add columns and then fill them)
    # TODO: change this to read in from config
    # TODO: add ability to further subset data, e.g. by each 'sat' - just wrap in for loop
    # TODO: add progress indication

    # --
    # data files
    # --

    # data_file = get_data_path("RAW", "sats_ra_cry_processed_arco.h5")
    # data_file = get_data_path("RAW", "gpod_all_rows.h5")
    # data_file = get_data_path("RAW", "gpod_lead.h5")
    # data_file = get_data_path("RAW", "sats_ra_cry_processed_arco_all_elev.h5")
    # data_file = "/mnt/m2_red_1tb/Data/CSAO/SAR_A.h5"
    # data_file = "/mnt/m2_red_1tb/Data/CSAO/SIN.h5"
    data_file = "/mnt/m2_red_1tb/Data/GPOD/gpod_processed_no_category.h5"
    # data_file = config['data_file']

    # table = "_data_batches"
    table = config.get("table", "data")

    load_kwargs = config.get("load_kwargs", {"reset_index": False})

    # to get 'daily' track numbers
    # add_by_track_date = True
    add_by_track_date = config.get("add_by_track_date", False)

    # get track by: if None will assumed to be from a single source
    get_track_by = config.get("get_track_by", None)

    # change in (either dt: time, or dr: space) used to infer when a new track starts
    delta_measure = config.get("delta_measure", "dt")
    assert delta_measure in ['dt', 'dr'], f"provided delta_measure: {delta_measure} is not valid"

    # a jump larger than cut_off for delta_measure will signify a new track has started
    cut_off = config.get("cut_off", 1e6)

    # output table
    out_table = config.get("out_table", f"{table}_w_tracks")
    assert out_table != table

    # TODO: allow for a group or split by, e.g. 'sat' - also allow it to be missing (implied for a single sat)
    # sort_by = ['sat', 'datetime']
    sort_by = config.get("sort_by", ['datetime'])

    # what are these used for?
    lon_col = config.get("lon_col", "lon")
    lat_col = config.get("lat_col", "lat")

    lat_0, lon_0 = config.get("lat_0", 90), config.get("lon_0", 0)

    # for plotting
    pole = config.get("pole", "north")

    # ------
    # read in data
    # ------

    pd.set_option("display.max_columns", 200)

    assert os.path.exists(data_file)

    # NOTE: reading in all data can require a lot of memory
    cprint("reading in data", c="OKBLUE")
    df = DataLoader.load(source=data_file,
                         table=table,
                         **load_kwargs)

    print("read in data with head:")
    print(df.head(2))

    # -----
    # prep data
    # -----

    # NOTE: this bit is fairly hardcoded

    print(f"data has: {len(df)} rows")

    # dt = df['datetime'].values
    cprint(f"sorting values by: {sort_by}", c="OKGREEN")
    df.sort_values(sort_by, inplace=True)

    # TODO: handle this when reading in?
    # get time in ms
    df['t'] = df['datetime'].values.astype("datetime64[ms]").astype(float)

    # add x,y coordinates - will use for measuring distance
    df['x'], df['y'] = WGS84toEASE2_New(df[lon_col], df[lat_col], lat_0=lat_0, lon_0=lon_0)

    # ----
    # add tracks
    # ----

    print("adding tracks")
    if get_track_by is None:
        _ = assign_tracks(df, delta_measure=delta_measure, cut_off=cut_off, plot_vals=True, plot_first=1000000, title="")
    else:
        tmp = []
        uvals = df[get_track_by].unique()
        print(f"get tracks by: '{get_track_by}' - {uvals}")
        start_track = 0
        for uv in uvals:
            print(f"{get_track_by}: {uv}")
            _ = df.loc[df[get_track_by] == uv].copy(True)
            _ = assign_tracks(_, delta_measure=delta_measure, cut_off=cut_off, plot_vals=True,
                              plot_first=1000000, title=f"{get_track_by}: {uv}",
                              start_track=start_track)
            start_track = _['track'].max() + 1

            tmp.append(_)
        _ = pd.concat(tmp)

    # --
    # plot tracks / investigate (make optional / remove)
    # --

    # obs per track
    # - this can be slow and take a lot of memory
    opt = pd.pivot_table(_, index=['track', get_track_by], values='datetime', aggfunc='count')
    opt.sort_values("datetime", inplace=True, ascending=False)
    opt.reset_index(inplace=True)

    plt.plot(opt['datetime'].values)
    plt.title("observations per track (sorted)")
    plt.show()

    view_tracks = opt.loc[opt['sat'] == "S3B", "track"].values[:20]
    # view_tracks = np.arange(100)
    # tmp = _.iloc[:1_000_000].copy(True)
    tmp = _.loc[_['track'].isin(view_tracks)]
    # tmp = _.loc[_['track'] == (opt.iloc[-50, :]["track"]-1)]

    utracks = tmp['track'].unique()
    print(f"unique track count: {len(utracks)}")

    # select_tracks = np.random.choice(utracks, 30, replace=False)
    # select_tracks = utracks[:26]
    # select_bool = tmp['track'].isin(select_tracks).values
    # select_bool = tmp['date'].isin(["2019-01-01"])
    select_bool = np.ones(len(tmp), dtype='bool')

    lat, lon = tmp.loc[select_bool, lat_col].values, tmp.loc[select_bool, lon_col].values

    # z = tmp['elev'].values
    # z[np.isnan(z)] = np.nanmean(z)

    # z = np.arange(len(tmp))
    z = tmp.loc[select_bool, 'track'].values
    print(len(np.unique(z)))
    # TOOD: plot a random subset of tracts

    # first plot: heat map of observations


    # parameters for projections
    projection = get_projection(pole)
    extent = [-180, 180, 60, 90] if pole == "north" else [-180, 180, -60, -90]

    figsize = (20, 20)
    fig = plt.figure(figsize=figsize)

    ax = fig.add_subplot(1, 1, 1,
                         projection=projection)

    cmap = 'YlGnBu_r'
    cmap = 'nipy_spectral'
    # TODO: allo for plotting the south pole
    plot_pcolormesh(ax,
                    lon=lon,
                    lat=lat,
                    plot_data=z,
                    scatter=True,
                    s=5,
                    fig=fig,
                    cbar_label="track number",
                    cmap=cmap,
                    extent=extent)

    # - would it be faster to write to file
    plt.show()

    # ----
    # write to table
    # ----

    # HARDCODED: columns just add to drop before writing to file
    # TODO: it's possible these columns could have been in original data, check
    drop_columns = ['x', 'y', 't', 'dt', 'dr']
    _.drop(drop_columns, axis=1, inplace=True)

    cprint("head of data:", c='OKBLUE')
    print(_.head(2))
    cprint(f"writing to file:\n{data_file}\ntable: '{out_table}'", c='OKBLUE')
    print(f"table has: {len(_)} rows")
    with pd.HDFStore(data_file, mode="a") as store:

        track_config = {"comment": f"added tracks using data from table: '{table}'"}
        run_info = DataLoader.get_run_info()

        # format table to get 'pytables' functionality
        store.put(out_table, _, data_columns=True, format='table')

        storer = store.get_storer(out_table)
        storer.attrs["config"] = track_config
        storer.attrs["run_info"] = run_info
