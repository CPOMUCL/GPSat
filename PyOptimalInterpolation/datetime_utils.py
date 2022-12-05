# functions to generate datetime64 arrays

import os
import datetime
import re

import numpy as np
import pandas as pd


def from_file_start_end_datetime_GPOD(f, df):
    """
    parse file name to get start and end datetimes
    - assume observations occur at equally spaced intervals over

    Parameters
    ----------
    f: str, filename
    df: pd.DataFrame

    Returns
    -------
    np.array, dtype datetime64[s]

    """

    f = os.path.basename(f)
    # get the file datetime - interval
    dt0, dt1 = f.split("_")[-5], f.split("_")[-4]
    # convert to datetime(64)

    dt0 = np.datetime64(datetime.datetime.strptime(dt0, "%Y%m%dT%H%M%S"))
    dt1 = np.datetime64(datetime.datetime.strptime(dt1, "%Y%m%dT%H%M%S"))

    delta_t = (dt1 - dt0) / len(df)

    dt = dt0 + np.arange(len(df)) * delta_t

    # convert to seconds
    dt = dt.astype('datetime64[s]')

    return dt


def from_file_start_end_datetime_SARAL(f, df):

    # TODO: add doc string
    f = os.path.basename(f)
    # example file name: SARAL_C139_0036_20200331_234125_20200401_003143_CS2mss_IL_v1.proc
    # found: /cpdata/SATS/RA/ALTIKA/SGDR_f/processed/north/by_month/YYYYMM/

    # get the file datetime - interval
    fsplit = f.split("_")
    date0, time0 = fsplit[3], fsplit[4]
    date1, time1 = fsplit[5], fsplit[6]

    # dt0, dt1 = f.split("_")[-5], f.split("_")[-4]
    # convert to datetime(64)

    dt0 = np.datetime64(datetime.datetime.strptime(date0+time0, "%Y%m%d%H%M%S"))
    dt1 = np.datetime64(datetime.datetime.strptime(date1+time1, "%Y%m%d%H%M%S"))

    delta_t = (dt1 - dt0) / len(df)

    dt = dt0 + np.arange(len(df)) * delta_t

    # convert to seconds
    dt = dt.astype('datetime64[s]')

    return dt


def datetime_from_float_column(float_datetime, epoch=(1950, 1, 1), time_unit='D'):

    # convert float_datetime to a timedelta (time since epoch)
    dt = pd.to_timedelta(float_datetime, unit=time_unit) + datetime.datetime(*epoch)
    return dt.values.astype('datetime64[s]')


def date_from_datetime(dt):
    # convert a datetime column with format YYYY-MM-DD HH:mm:SS
    # would it be faster use apply on a Series?
    remove_dash_and_time = lambda x: re.sub(" .*$|-", "", x)
    return np.array([remove_dash_and_time(_) for _ in dt])


if __name__ == "__main__":

    pass


