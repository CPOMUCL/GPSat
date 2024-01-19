# functions to generate datetime64 arrays

import os
import datetime
import re

import numpy as np
import pandas as pd


def from_file_start_end_datetime_GPOD(f, df):
    """
    Extract an implied sequence of evenly spaced time intervals based off of a 'processed' GPOD file name

    This function takes in a file path and a pandas dataframe as input.
    It extracts the start and end datetime from the file name and
    calculates the time interval between them.

    It then generates a datetime array with the same length as the dataframe,
    evenly spaced over the time interval. The resulting datetime array is returned.

    Parameters
    ----------
    f: str
        filename
    df: pd.DataFrame, pd.Series, np.array, tuple, list
        the len(df) is used to determine the number and size of the intervals


    Returns
    -------
    np.array
        dtype datetime64[ns]

    Examples
    --------
    >>> f = "/path/to/S3A_GPOD_SAR__SRA_A__20191031T233355_20191101T002424_2019112_IL_v3.proc"
    >>> df = pd.DataFrame({"x": np.arange(11)})
    >>> from_file_start_end_datetime_GPOD(f, df)
    array(['2019-10-31T23:33:55.000000000', '2019-10-31T23:38:57.900000000',
           '2019-10-31T23:44:00.800000000', '2019-10-31T23:49:03.700000000',
           '2019-10-31T23:54:06.600000000', '2019-10-31T23:59:09.500000000',
           '2019-11-01T00:04:12.400000000', '2019-11-01T00:09:15.300000000',
           '2019-11-01T00:14:18.200000000', '2019-11-01T00:19:21.100000000',
           '2019-11-01T00:24:24.000000000'], dtype='datetime64[ns]')

    """

    f = os.path.basename(f)
    # get the file datetime - interval
    dt0, dt1 = f.split("_")[-5], f.split("_")[-4]
    # convert to datetime(64)

    dt0 = np.datetime64(datetime.datetime.strptime(dt0, "%Y%m%dT%H%M%S"))
    dt1 = np.datetime64(datetime.datetime.strptime(dt1, "%Y%m%dT%H%M%S"))

    assert isinstance(df, (pd.DataFrame, pd.Series, np.ndarray, list, tuple)), \
        f"df is wrong type, got: {type(df)}"
    assert len(df) > 0, "df has length 0"

    _ = (len(df) - 1)
    _ = 1 if _ == 0 else _

    delta_t = (dt1 - dt0) / _

    dt = dt0 + np.arange(len(df)) * delta_t

    # convert to nano seconds
    dt = dt.astype('datetime64[ns]')

    return dt


def from_file_start_end_datetime_SARAL(f, df):
    """
    This function takes in a file path to a file and
    a pandas dataframe and returns a numpy array of datetime objects.

    The file path is expected to be in the format of SARAL data files,
    with the datetime information encoded in the file name.
    The function extracts the start and end datetime information from the file name,
    calculates the time interval between them based on the length of the dataframe, and
    generates a numpy array of datetime objects with the same length as the dataframe.

    Parameters
    ----------
    f: str
        the file path of the SARAL data file
    df: pd.DataFrame
        the data contained in the SARAL data file

    Returns
    -------
    np.array
        datetime objects, representing the time stamps of the data in the SARAL data file
        with dtype: 'datetime64[s]'

    Examples
    --------
    >>> f = "/path/to/SARAL_C139_0036_20200331_234125_20200401_003143_CS2mss_IL_v1.proc"
    >>> df = pd.DataFrame({"x": np.arange(11)})
    >>> from_file_start_end_datetime_SARAL(f, df)
    array(['2020-03-31T23:41:25', '2020-03-31T23:46:26',
           '2020-03-31T23:51:28', '2020-03-31T23:56:30',
           '2020-04-01T00:01:32', '2020-04-01T00:06:34',
           '2020-04-01T00:11:35', '2020-04-01T00:16:37',
           '2020-04-01T00:21:39', '2020-04-01T00:26:41',
           '2020-04-01T00:31:43'], dtype='datetime64[s]')


    """
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

    assert isinstance(df, (pd.DataFrame, pd.Series, np.ndarray, list, tuple)), f"df is wrong type, got: {type(df)}"
    assert len(df) > 0, "df has length 0"

    _ = (len(df) - 1)
    _ = 1 if _ == 0 else _

    delta_t = (dt1 - dt0) / _

    dt = dt0 + np.arange(len(df)) * delta_t

    # convert to seconds
    dt = dt.astype('datetime64[s]')

    return dt


def datetime_from_float_column(float_datetime, epoch=(1950, 1, 1), time_unit='D'):
    """
    Converts a float datetime column to a datetime64 format.

    Parameters
    ----------
    float_datetime : pd.Series or np.array
        A pandas series or numpy array containing float values, corresponding to datetime.
    epoch : tuple, default is (1950, 1, 1).
        A tuple representing the epoch date in the format (year, month, day).
    time_unit : str, optional
        The time unit of the float datetime values. Default is 'D' (days).

    Returns
    -------
    numpy.ndarray
        A numpy array of datetime64 values, with dtype 'datetime64[s]'

    Examples
    --------
    >>> df = pd.DataFrame({'float_datetime': [18262.5, 18263.5, 18264.5]})
    >>> datetime_from_float_column(df['float_datetime'])
    array(['2000-01-01T12:00:00', '2000-01-02T12:00:00',
           '2000-01-03T12:00:00'], dtype='datetime64[s]')

    >>> df = pd.DataFrame({'float_datetime': [18262.5, 18263.5, 18264.5]})
    >>> datetime_from_float_column(df['float_datetime'], epoch=(1970, 1, 1))
    array(['2020-01-01T12:00:00', '2020-01-02T12:00:00',
           '2020-01-03T12:00:00'], dtype='datetime64[s]')

    >>> x = np.array([18262.5, 18263.5, 18264.5])
    >>> datetime_from_float_column(x, epoch=(1970, 1, 1))
    array(['2020-01-01T12:00:00', '2020-01-02T12:00:00',
           '2020-01-03T12:00:00'], dtype='datetime64[s]')
    """
    # convert float_datetime to a timedelta (time since epoch)
    dt = pd.to_timedelta(float_datetime, unit=time_unit) + datetime.datetime(*epoch)
    return dt.values.astype('datetime64[s]')


# TODO: remove date_from_datetime - not used in package
def date_from_datetime(dt):
    """
    Remove the time component of an array of datetimes (represented as strings) and just return the date

    The datetime format is expected to be YYYY-MM-DD HH:mm:SS
    The returned date format is YYYYMMDD

    Parameters
    ----------
    dt: list, np.array, pd.Series
        string with datetime format YYYY-MM-DD HH:mm:SS.

    Returns
    -------
    numpy.ndarray: A date column with format YYYY-MM-DD.

    Examples
    --------
        >>> dt = pd.Series(['2022-01-01 12:00:00', '2022-01-02 13:00:00', '2022-01-03 14:00:00'])
        >>> date_from_datetime(dt)
        array(['20220101', '20220102', '20220103'], dtype='<U8')

.. note::
    This function uses a lambda function to remove the time portion and the dash from the datetime column.
    It then returns a numpy array of the resulting date column.
    It is possible to use apply on a Series to achieve the same result,
    but it may not be as fast as using a lambda function and numpy array.
    """
    # convert a datetime column with format YYYY-MM-DD HH:mm:SS
    # would it be faster use apply on a Series?
    remove_dash_and_time = lambda x: re.sub(" .*$|-", "", x)
    return np.array([remove_dash_and_time(_) for _ in dt])


def datetime_from_ymd_cols(year, month, day, hhmmss):
    """
    Converts separate columns/arrays of year, month, day, and time (in hhmmss format)
    into a numpy array of datetime objects.

    Parameters
    ----------
    year : array-like
        An array of integers representing the year.
    month : array-like
        An array of integers representing the month (1-12).
    day : array-like
        An array of integers representing the day of the month.
    hhmmss : array-like
        An array of integers representing the time in hhmmss format.

    Returns
    -------
    datetime : numpy.ndarray
        An array of datetime objects representing the input dates and times.

    Raises
    ------
    AssertionError
        If the input arrays are not of equal length.

    Examples
    --------
    >>> year = [2021, 2021, 2021]
    >>> month = [1, 2, 3]
    >>> day = [10, 20, 30]
    >>> hhmmss = [123456, 234537, 165648]
    >>> datetime_from_ymd_cols(year, month, day, hhmmss)
    array(['2021-01-10T12:34:56', '2021-02-20T23:45:37',
           '2021-03-30T16:56:48'], dtype='datetime64[s]')

    """

    # NOTE: the following is likely slow...
    # check sizes
    assert len(year) == len(month)
    assert len(month) == len(day)
    assert len(day) == len(hhmmss)

    # add leading zero to hhmmss
    hhmmss = np.array([f"{_:06}" for _ in hhmmss])
    datetime = [f"{year[i]}-{month[i]:02}-{day[i]:02} {hhmmss[i][0:2]}:{hhmmss[i][2:4]}:{hhmmss[i][4:6]}"
                for i in range(len(year))]
    datetime = np.array(datetime).astype("datetime64[s]")

    return datetime


if __name__ == "__main__":

    pass

