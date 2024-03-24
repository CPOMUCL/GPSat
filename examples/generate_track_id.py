# assign (arbitrary) track number for raw observation data
# - this script is a work in progress
import json
import os.path
import inspect

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

# from dataclasses import dataclass
from typing import List, Dict, Tuple, Union, Type

from GPSat import get_data_path
from GPSat.plot_utils import plot_pcolormesh, get_projection
from GPSat.utils import WGS84toEASE2, cprint, diff_distance, \
    guess_track_num, get_config_from_sysargv, track_num_for_date, _method_inputs_to_config
from GPSat.dataloader import DataLoader


# ---
# helper function
# ---

class TrackId:
    """
    This class provide methods to attempt to identify tracks in satellite based observations
    by considering the changes in time, or distance, from one observation to another
    """

    def __init__(self,
                 data_file: Union[str, None] = None,
                 output_file:  Union[str, None] = None,
                 output_table:  Union[str, None] = None,
                 table: Union[str, None] = None,
                 df: pd.DataFrame = None,
                 sort_by: Union[list, str, None] = None,
                 add_by_track_date: bool = False,
                 get_track_by: Union[str, list, tuple] = None,
                 delta_measure: str = "dr",
                 cut_off: Union[int, float] = 600_000,
                 load_kwargs: Union[dict, None] = None,
                 lon_col: str = "lon",
                 lat_col: str = "lat",
                 lat_0: int = 90,
                 lon_0: int = 0,
                 pole: str = "north",
                 ):
        # NOTE: not using dataclass so can assign
        self.config = _method_inputs_to_config(locs=locals(), code_obj=self.__init__.__code__)

        # assign inputs to as attributes
        # ref: https://stackoverflow.com/questions/56979052/is-it-possible-to-make-all-the-inputs-to-class-init-as-attributes-in-one-l
        for item in inspect.signature(TrackId).parameters:
            setattr(self, item, eval(item))

        # TODO: double check what the __file__ value will be if call from another script
        try:
            self.run_info = DataLoader.get_run_info(script_path=__file__)
        except NameError as e:
            self.run_info = DataLoader.get_run_info()

        # ---
        # checks on inputs
        # ---

        # TODO: finish checks on inputs
        assert (self.data_file is not None) or (self.df is not None), "both data_file and df can't be None"

        if self.data_file is not None:
            assert os.path.exists(self.data_file), f"data_file provide:\n{self.data_file}but does not exist"
            assert self.table is not None, "data_file provide, but table is None"

        assert self.sort_by is not None
        assert isinstance(self.add_by_track_date, bool)
        assert isinstance(self.cut_off, (int, float))
        assert self.delta_measure in ['dt', 'dr'], f"delta_measure: '{self.delta_measure}', must in ['dt', 'dr']"

        assert isinstance(self.sort_by, (list, tuple, str)), \
            f"sort_by should be str, list or tuple, got: {type(self.sort_by)}"

        if self.load_kwargs is None:
            self.load_kwargs = {"reset_index": False}

        assert isinstance(self.load_kwargs, dict), f"load_kwargs should be dict, got: {type(self.load_kwargs)}"

        if self.output_file is None:
            print("output_file not supplied, setting output_file to data_file")
            self.output_file = self.data_file

        assert isinstance(self.output_file, str)

        if self.output_table is None:
            self.output_table = f"{self.table}_w_tracks"
            print(f"output_table not supplied, setting output_table to '{self.output_table}'")


    def read_data(self):

        if self.data_file is not None:

            # NOTE: reading in all data can require a lot of memory
            cprint(f"reading in data from source:\n{self.data_file}\ntable:\n{self.table}", c="OKBLUE")
            cprint(f"load_kwargs: {json.dumps(self.load_kwargs, indent=4)}", c="OKGREEN")
            df = DataLoader.load(source=self.data_file,
                                 table=self.table,
                                 **self.load_kwargs)

            print("read in data with head:")
            print(df.head(2))
            print("assigning data to df attribute")
            self.df = df

        else:
            cprint("data_file is None, not reading anything in", c="WARNING")

    def add_xyt_columns(self):

        print(f"data has: {len(self.df)} rows")

        # TODO: allow this to be more generalised, perhaps performed via a DataLoader.load - with col_funcs
        # col_funcs should be provided on init
        # col_funcs = {
        #     "t": {
        #         "func": "lambda x: x.astype('datetime64[ms]').astype('float')",
        #         "col_args": "datetime"
        #     },
        #     "('x', 'y')": {
        #         "source": "GPSat.utils",
        #         "func": "WGS84toEASE2_New",
        #         "col_kwargs": {
        #             "lon": "lon",
        #             "lat": "lat"
        #         },
        #         "kwargs": {
        #             "lat_0": 90,
        #             "lon_0": 0
        #         }
        #     },
        # }
        # self.df = DataLoader.load(source=self.df,
        #                           col_funcs=col_funcs)

        # HARDCODED: check certain columns exist
        assert 'datetime' in self.df, f"'datetime' is not in data"
        for _ in [self.lon_col, self.lat_col]:
            assert _ in self.df, f"column: '{_}' is not in data"

        # get time in ms
        self.df['t'] = self.df['datetime'].values.astype("datetime64[ms]").astype(float)

        # add x,y coordinates - will use for measuring distance
        cprint(f"getting (x,y) coordinates (via WGS84toEASE2_New) using (lon, lat) = ('{self.lon_col}', '{self.lat_col}')", c="OKGREEN")
        self.df['x'], self.df['y'] = WGS84toEASE2(lon=self.df[self.lon_col],
                                                  lat=self.df[self.lat_col],
                                                  lat_0=self.lat_0,
                                                  lon_0=self.lon_0)

        # sort values - should be some time measure
        cprint(f"sorting values by: {self.sort_by}", c="OKGREEN")
        self.df.sort_values(self.sort_by, inplace=True)

    @staticmethod
    def assign_tracks(df,
                      delta_measure,
                      cut_off,
                      start_track=0,
                      add_by_track_date=False,
                      plot_vals=True,
                      plot_first=1000000,
                      plt_title=""):

        _ = df  # .copy(True)
        # _ = self.df
        # delta_measure = self.delta_measure
        # cut_off = self.cut_off

        assert delta_measure in ["dt", "dr"]

        # create a distance (space or time) to next observation
        x = _[['x', 'y']].values

        # difference in distance in space
        _['dr'] = diff_distance(x, p=2, k=1)
        # difference in distance in time
        _['dt'] = diff_distance(_['t'].values, p=2, k=1)

        # ------
        # visualise the deltas - in order to determine the delta cut off size for a new track
        # ------

        if plot_vals:
            print("visualise (plotting) differences in time and space")
            dtime = _['datetime'][:plot_first]
            xmin, xmax = dtime.min(), dtime.max()

            plt.plot(dtime, _['dt'][:plot_first])
            plt.hlines(y=cut_off, xmin=xmin, xmax=xmax, color="black", label='cutoff')
            plt.title(f"dt\n{plt_title}")
            plt.xticks(rotation=45)
            plt.legend()
            plt.tight_layout()
            plt.show()

            plt.plot(dtime, _['dr'][:plot_first])
            plt.hlines(y=cut_off, xmin=xmin, xmax=xmax, color="black", label='cutoff')
            plt.title(f"dr\n{plt_title}")
            plt.xticks(rotation=45)
            plt.legend()
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

    def add_tracks(self, start_track=0, plot_vals=True, plot_first=1000000, plt_title=""):

        get_track_by = self.get_track_by
        df = self.df

        cprint("adding tracks", c="OKBLUE")
        if get_track_by is None:
            _ = self.assign_tracks(df,
                                   delta_measure=self.delta_measure,
                                   cut_off=self.cut_off,
                                   start_track=start_track,
                                   add_by_track_date=self.add_by_track_date,
                                   plot_vals=plot_vals,
                                   plot_first=plot_first,
                                   plt_title=plot_vals)
        else:
            tmp = []
            uvals = df[get_track_by].unique()
            cprint(f"get tracks by: '{get_track_by}' - {uvals}", c="OKGREEN")
            for uv in uvals:
                print(f"{get_track_by}: {uv}")
                _ = df.loc[df[get_track_by] == uv].copy(True)
                _ = self.assign_tracks(_,
                                       delta_measure=self.delta_measure,
                                       cut_off=self.cut_off,
                                       start_track=start_track,
                                       add_by_track_date=self.add_by_track_date,
                                       plot_vals=plot_vals,
                                       plot_first=plot_first,
                                       plt_title=f"{plt_title}\n{get_track_by}: {uv}")
                start_track = _['track'].max() + 1

                tmp.append(_)
            _ = pd.concat(tmp)

        return _

    def write_dataframe_to_table(self, df, output_file=None, table=None, drop_columns=True):

        # TODO: more care should be taken for columns names added
        #  - perhaps use _ prefix to (help) avoid overwriting existing columns! otherwise check column names explicitly
        if drop_columns:
            dcol = ['x', 'y', 't', 'dt', 'dr']
            cprint(f"dropping columns:\n{dcol}")
            df.drop(dcol, axis=1, inplace=True)

        if output_file is None:
            assert self.data_file is not None, "output_file is None, so is data_file - can't default to data_file"
            cprint(f"output_file not provided, will use data_file:\n{self.data_file}", c="HEADER")
            output_file = self.data_file

        if table is None:
            assert self.table is not None, "table not provided and table attribute is None"
            table = f"{self.table}_w_tracks"
            cprint(f"in writing to file table not provide, will use: '{table}'", c="HEADER")

        cprint(f"writing results to hdf5 file:\n{output_file}\ntable: {table}", c="OKGREEN")
        
        # import pdb; pdb.set_trace()
        with pd.HDFStore(output_file, mode="a") as store_out:

            print(f"writing to table: {table}")
            store_out.put(key=table,
                          value=df,
                          append=False,
                          format='table',
                          data_columns=True)

            store_attrs = store_out.get_storer(table).attrs

            store_attrs["config"] = self.config
            store_attrs["run_info"] = self.run_info


def get_track_id_config():
    config = get_config_from_sysargv()

    # if config not supplied, use default
    if config is None:
        config = {
            "data_file": get_data_path("example", "ABC.h5"),  # raw data file
            "table": "data",  # table to read from
            "sort_by": ["datetime"],
            "add_by_track_date": False,  # start counting tracks from 0 on each date
            "get_track_by": "source",  # select data by unique values of 'get_track_by' when getting track id
            "delta_measure": "dr",  # valid values 'dt' (time), 'dr' (change in distance)
            "cut_off": 600_000,  # a new track number when delta_measure is greater than cutoff
        }

    return config


if __name__ == "__main__":

    # TODO: be more memory efficient - copy less (add columns and then fill them)
    # TODO: add progress indication

    pd.set_option("display.max_columns", 200)
    # NOTE: currently this scripts expects data to have a 'datetime' column, along with some longitude and latitude

    # ----
    # read / get default config
    # -----

    config = get_track_id_config()

    # ---
    # initialise TrackId object
    # ---

    trackid = TrackId(**config)

    # ---
    # read in data
    # ---

    # if data_file and table provide, read in and assign to df attribute
    # NOTE: reading in all data can require a lot of memory!
    trackid.read_data()

    # ---
    # add columns used to calculate
    # ---

    # x,y,t columns are used to identify tracks
    # - currently this method is rigid, expects 'datetime' to exist in data
    trackid.add_xyt_columns()

    # ---
    # add track_id
    # ---

    # set plot_vals to True to see (some) of the difference in space/time
    df_with_tracks = trackid.add_tracks(plot_vals=False)

    # ---
    # (optional) plot a sample of tracks, get statistics
    # ---

    # TODO: add - see commented out section below

    # ---
    # (optional) write to file
    # ---

    trackid.write_dataframe_to_table(df_with_tracks,
                                     output_file=trackid.output_file,
                                     table=trackid.output_table)


    # # ----
    # # parameters
    # # ----
    #

    # # --
    # # plot tracks / investigate (make optional / remove)
    # # --
    #
    # # obs per track
    # # - this can be slow and take a lot of memory
    # opt = pd.pivot_table(_, index=['track', get_track_by], values='datetime', aggfunc='count')
    # opt.sort_values("datetime", inplace=True, ascending=False)
    # opt.reset_index(inplace=True)
    #
    # plt.plot(opt['datetime'].values)
    # plt.title("observations per track (sorted)")
    # plt.show()
    #
    # view_tracks = opt.loc[opt['sat'] == "S3B", "track"].values[:20]
    # # view_tracks = np.arange(100)
    # # tmp = _.iloc[:1_000_000].copy(True)
    # tmp = _.loc[_['track'].isin(view_tracks)]
    # # tmp = _.loc[_['track'] == (opt.iloc[-50, :]["track"]-1)]
    #
    # utracks = tmp['track'].unique()
    # print(f"unique track count: {len(utracks)}")
    #
    # # select_tracks = np.random.choice(utracks, 30, replace=False)
    # # select_tracks = utracks[:26]
    # # select_bool = tmp['track'].isin(select_tracks).values
    # # select_bool = tmp['date'].isin(["2019-01-01"])
    # select_bool = np.ones(len(tmp), dtype='bool')
    #
    # lat, lon = tmp.loc[select_bool, lat_col].values, tmp.loc[select_bool, lon_col].values
    #
    # # z = tmp['elev'].values
    # # z[np.isnan(z)] = np.nanmean(z)
    #
    # # z = np.arange(len(tmp))
    # z = tmp.loc[select_bool, 'track'].values
    # print(len(np.unique(z)))
    # # TOOD: plot a random subset of tracts
    #
    # # first plot: heat map of observations
    #
    # # parameters for projections
    # projection = get_projection(pole)
    # extent = [-180, 180, 60, 90] if pole == "north" else [-180, 180, -60, -90]
    #
    # figsize = (20, 20)
    # fig = plt.figure(figsize=figsize)
    #
    # ax = fig.add_subplot(1, 1, 1,
    #                      projection=projection)
    #
    # cmap = 'YlGnBu_r'
    # cmap = 'nipy_spectral'
    # # TODO: allo for plotting the south pole
    # plot_pcolormesh(ax,
    #                 lon=lon,
    #                 lat=lat,
    #                 plot_data=z,
    #                 scatter=True,
    #                 s=5,
    #                 fig=fig,
    #                 cbar_label="track number",
    #                 cmap=cmap,
    #                 extent=extent)
    #
    # # - would it be faster to write to file
    # plt.show()
    #

