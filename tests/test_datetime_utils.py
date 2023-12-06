# unit test for functions in datetime_utils

import pytest
import numpy as np
import pandas as pd
import datetime

from GPSat.datetime_utils import from_file_start_end_datetime_SARAL, \
    datetime_from_ymd_cols, datetime_from_float_column, from_file_start_end_datetime_GPOD

# -----
# datetime_from_float_column
# -----

@pytest.mark.parametrize(
    "float_datetime, epoch, time_unit, expected",
    [
        # Test basic functionality with default epoch and time_unit
        (
            pd.Series([18262.5, 18263.5, 18264.5]),
            (1950, 1, 1),
            'D',
            np.array(['2000-01-01T12:00:00', '2000-01-02T12:00:00', '2000-01-03T12:00:00'], dtype='datetime64[s]')
        ),
        # Test basic functionality with non-default epoch and time_unit
        (
            pd.Series([18262.5, 18263.5, 18264.5]),
            (1970, 1, 1),
            'D',
            np.array(['2020-01-01T12:00:00', '2020-01-02T12:00:00', '2020-01-03T12:00:00'], dtype='datetime64[s]')
        ),
        # Test input with NaN values
        (
            pd.Series([np.nan, 18262.5, 18263.5]),
            (1950, 1, 1),
            'D',
            np.array(['NaT', '2000-01-01T12:00:00', '2000-01-02T12:00:00'], dtype='datetime64[s]')
        ),
        # Test input with negative float values
        (
            pd.Series([-1.5, 0.0, 1.5]),
            (1950, 1, 1),
            'D',
            np.array(['1949-12-30T12:00:00', '1950-01-01T00:00:00', '1950-01-02T12:00:00'], dtype='datetime64[s]')
        ),
        # Test input with very large float values
        (
            pd.Series([1e9, 2e9, 3e9]),
            (1950, 1, 1),
            's',
            np.array(['1981-09-09T01:46:40', '2013-05-18T03:33:20',
                      '2045-01-24T05:20:00'], dtype='datetime64[s]')
        ),
        # Test input with float values outside of the representable range for datetime64
        (
            pd.Series([1e100, -1e100]),
            (1950, 1, 1),
            'D',
            np.array(['NaT', 'NaT'], dtype='datetime64[s]')
        ),
        # Test with basic input
        (pd.Series([18262.5, 18263.5, 18264.5]), (1950, 1, 1), 'D',
         np.array(['2000-01-01T12:00:00', '2000-01-02T12:00:00', '2000-01-03T12:00:00'], dtype='datetime64[s]')),

        # Test with different epoch
        (pd.Series([18262.5, 18263.5, 18264.5]), (1970, 1, 1), 'D',
         np.array(['2020-01-01T12:00:00', '2020-01-02T12:00:00', '2020-01-03T12:00:00'], dtype='datetime64[s]')),

        # Test with numpy array input
        (np.array([18262.5, 18263.5, 18264.5]), (1970, 1, 1), 'D',
         np.array(['2020-01-01T12:00:00', '2020-01-02T12:00:00', '2020-01-03T12:00:00'], dtype='datetime64[s]')),

        # Test with empty input
        (pd.Series([], dtype='object'), (1970, 1, 1), 'D', np.array([], dtype='datetime64[s]')),

        # Test with time_unit in hours
        (pd.Series([438001.0, 438025.0, 438049.0]), (1970, 1, 1), 'h',
         np.array(['2019-12-20T01:00:00', '2019-12-21T01:00:00', '2019-12-22T01:00:00'], dtype='datetime64[s]')),

        # Test with time_unit in minutes
        (pd.Series([26340161.0, 26340441.0, 26340721.0]), (1970, 1, 1), 'm',
         np.array(['2020-01-30T18:41:00', '2020-01-30T23:21:00', '2020-01-31T04:01:00'], dtype='datetime64[s]')),

        # Test with time_unit in seconds
        (pd.Series([1577920680, 1578007080, 1578093480]), (1970, 1, 1), 's',
         np.array(['2020-01-01T23:18:00', '2020-01-02T23:18:00', '2020-01-03T23:18:00'], dtype='datetime64[s]')),

    ]
)
def test_datetime_from_float_column(float_datetime, epoch, time_unit, expected):
    """
    Test the datetime_from_float_column function for various inputs and expected outputs
    """

    if isinstance(expected, type) and issubclass(expected, Exception):
        with pytest.raises(expected):
            result = datetime_from_float_column(float_datetime, epoch=epoch, time_unit=time_unit)
    else:
        # Run the function
        result = datetime_from_float_column(float_datetime, epoch=epoch, time_unit=time_unit)
        # Compare the result to the expected output
        np.testing.assert_array_equal(result, expected)


@pytest.mark.parametrize("float_datetime, epoch, time_unit, expected_exception", [
    # Test with invalid time_unit
    (pd.Series([18262.5, 18263.5, 18264.5]), (1970, 1, 1), 'X', ValueError),

    # Test with invalid epoch
    (pd.Series([18262.5, 18263.5, 18264.5]), '1970-01-01', 'D', TypeError),

    # Test with invalid float_datetime input
    (['2020-01-01T12:00:00', '2020-01-02T12:00:00', '2020-01-03T12:00:00'], (1970, 1, 1), 'D', ValueError),
])
def test_datetime_from_float_column_errors(float_datetime, epoch, time_unit, expected_exception):
    with pytest.raises(expected_exception):
        datetime_from_float_column(float_datetime, epoch, time_unit)

# -----
# datetime_from_ymd_cols
# -----

@pytest.mark.parametrize(
    "year, month, day, hhmmss, expected",
    [
        # Test basic functionality
        (
                [2021, 2021, 2021],
                [1, 2, 3],
                [10, 20, 30],
                [123456, 234537, 165648],
                np.array(['2021-01-10T12:34:56', '2021-02-20T23:45:37', '2021-03-30T16:56:48'], dtype='datetime64[s]')
        ),
        # Test input with leap year
        (
                [2024, 2024, 2024],
                [2, 2, 3],
                [28, 29, 1],
                [235959, 235959, 0],
                np.array(['2024-02-28T23:59:59', '2024-02-29T23:59:59', '2024-03-01T00:00:00'], dtype='datetime64[s]')
        ),
        # Test input with invalid month (too high)
        (
                [2021],
                [13],
                [1],
                [0],
                ValueError
        ),
        # Test input with invalid month (too low)
        (
                [2021],
                [0],
                [1],
                [0],
                ValueError
        ),
        # Test input with invalid day (too high)
        (
                [2021],
                [1],
                [32],
                [0],
                ValueError
        ),
        # Test input with invalid day (too low)
        (
                [2021],
                [1],
                [0],
                [0],
                ValueError
        ),
        # Test input with invalid hhmmss (too high)
        (
                [2021],
                [1],
                [1],
                [240000],
                ValueError
        ),
        # Test input with invalid hhmmss (too low)
        (
                [2021],
                [1],
                [1],
                [-1],
                ValueError
        ),
        # Test input with empty arrays
        (
                [],
                [],
                [],
                [],
                np.array([], dtype='datetime64[s]')
        ),
        # Test input with arrays of different lengths
        (
                [2021, 2022],
                [1, 2, 3],
                [1, 2, 3],
                [0, 0, 0],
                AssertionError
        ),
    ]
)
def test_datetime_from_ymd_cols(year, month, day, hhmmss, expected):
    """
    Test the datetime_from_ymd_cols function for various inputs and expected outputs
    """
    if isinstance(expected, type) and issubclass(expected, Exception):
        with pytest.raises(expected):
            datetime_from_ymd_cols(year, month, day, hhmmss)
    else:
        # Otherwise, we expect the output to match the expected value
        result = datetime_from_ymd_cols(year, month, day, hhmmss)
        # Compare the result to the expected output
        np.testing.assert_array_equal(result, expected)

# ----
# from_file_start_end_datetime_GPOD
# ----


@pytest.mark.parametrize("f, df, expected", [
    # Test with basic input
    ("/path/to/S3A_GPOD_SAR__SRA_A__20191031T233355_20191101T002424_2019112_IL_v3.proc",
     pd.DataFrame({"x": np.arange(11)}),
     np.array(['2019-10-31T23:33:55.000000000', '2019-10-31T23:38:57.900000000',
               '2019-10-31T23:44:00.800000000', '2019-10-31T23:49:03.700000000',
               '2019-10-31T23:54:06.600000000', '2019-10-31T23:59:09.500000000',
               '2019-11-01T00:04:12.400000000', '2019-11-01T00:09:15.300000000',
               '2019-11-01T00:14:18.200000000', '2019-11-01T00:19:21.100000000',
               '2019-11-01T00:24:24.000000000'], dtype='datetime64[ns]')),

    # Test with different length of df
    ("/path/to/S3A_GPOD_SAR__SRA_A__20191031T233355_20191101T002424_2019112_IL_v3.proc",
     pd.DataFrame({"x": np.arange(6)}),
     np.array(['2019-10-31T23:33:55.000000000', '2019-10-31T23:44:00.800000000',
               '2019-10-31T23:54:06.600000000', '2019-11-01T00:04:12.400000000',
               '2019-11-01T00:14:18.200000000', '2019-11-01T00:24:24.000000000'], dtype='datetime64[ns]')),

    # Test with empty df
    # ("/path/to/S3A_GPOD_SAR__SRA_A__20191031T233355_20191101T002424_2019112_IL_v3.proc",
    #  pd.DataFrame({"x": []}),
    #  np.array([], dtype='datetime64[ns]')),

    # Test with different filename format
    ("/path/to/COULD_BE_ANYTHING_HERE__20200101T000000_20200101T010000_202001_RL_v4.proc",
     pd.DataFrame({"x": np.arange(5)}),
     np.array(['2020-01-01T00:00:00.000000000', '2020-01-01T00:15:00.000000000',
               '2020-01-01T00:30:00.000000000', '2020-01-01T00:45:00.000000000',
               '2020-01-01T01:00:00.000000000'], dtype='datetime64[ns]')),
])
def test_from_file_start_end_datetime_GPOD(f, df, expected):
    result = from_file_start_end_datetime_GPOD(f, df)
    np.testing.assert_array_equal(result, expected)



@pytest.mark.parametrize("f, df, expected_exception", [
    # Test with invalid filename format
    ("/path/to/invalid_filename_format_20200101T000000_20200101T010000.proc",
     pd.DataFrame({"x": np.arange(5)}),
     ValueError),

    # Test with invalid datetime format in filename
    ("/path/to/S3A_GPOD_SAR__SRA_A__2020-01-01T00:00:00_2020-01-01T01:00:00_202001_IL_v3.proc",
     pd.DataFrame({"x": np.arange(5)}),
     ValueError),

    # Test with non-string file path
    (12345,
     pd.DataFrame({"x": np.arange(5)}),
     TypeError),

    # Test with non-DataFrame input
    ("/path/to/S3A_GPOD_SAR__SRA_A__20200101T000000_20200101T010000_202001_IL_v3.proc",
     {"x": np.arange(5)},
     AssertionError),
    # Test Empty DataFrame
    ("/path/to/S3A_GPOD_SAR__SRA_A__20191031T233355_20191101T002424_2019112_IL_v3.proc",
     pd.DataFrame({"x": []}),
     AssertionError),

])
def test_from_file_start_end_datetime_GPOD_errors(f, df, expected_exception):
    with pytest.raises(expected_exception):
        from_file_start_end_datetime_GPOD(f, df)


# ----
# from_file_start_end_datetime_SARAL
# ----


@pytest.mark.parametrize("f, df, expected_output", [
    # Test with DataFrame input
    ("/path/to/SARAL_C139_0036_20200331_234125_20200401_003143_CS2mss_IL_v1.proc",
     pd.DataFrame({"x": np.arange(11)}),
     np.array(['2020-03-31T23:41:25', '2020-03-31T23:46:26',
               '2020-03-31T23:51:28', '2020-03-31T23:56:30',
               '2020-04-01T00:01:32', '2020-04-01T00:06:34',
               '2020-04-01T00:11:35', '2020-04-01T00:16:37',
               '2020-04-01T00:21:39', '2020-04-01T00:26:41',
               '2020-04-01T00:31:43'], dtype='datetime64[s]')),
    # Test with incorrect file name format
    ("/path/to/incorrect_format_20200331_234125_20200401_003143_IL_v1.proc",
     pd.DataFrame({"x": np.arange(11)}),
     ValueError),
    # Test with empty dataframe
    ("/path/to/SARAL_C139_0036_20200331_234125_20200401_003143_CS2mss_IL_v1.proc",
                                           pd.DataFrame(), AssertionError),
    # Test with incorrect input types
    (123, pd.DataFrame({"x": np.arange(11)}), TypeError),
    ("/path/to/SARAL_C139_0036_20200331_234125_20200401_003143_CS2mss_IL_v1.proc", "123",AssertionError)

])
def test_from_file_start_end_datetime_SARAL(f, df, expected_output):

    if isinstance(expected_output, type) and issubclass(expected_output, Exception):
        with pytest.raises(expected_output):
            result = from_file_start_end_datetime_SARAL(f, df)
    else:
        result = from_file_start_end_datetime_SARAL(f, df)
        np.testing.assert_array_equal(result, expected_output)

