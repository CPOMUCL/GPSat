# unit test for functions in utils
# TODO: these tests should be reviewed and expanded where needed
import pytest
import datetime
import numpy as np
import pandas as pd
from numpy.testing import assert_array_equal
from pandas.testing import assert_frame_equal

import re
import os
from typing import Callable

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import tensorflow_probability as tfp

# import the function to be tested
from GPSat.utils import array_to_dataframe, to_array, \
    dataframe_to_array, match, pandas_to_dict, grid_2d_flatten, convert_lon_lat_str, \
    config_func, EASE2toWGS84, WGS84toEASE2, nested_dict_literal_eval, \
    dataframe_to_2d_array, sigmoid, inverse_sigmoid, softplus, inverse_softplus, \
    get_weighted_values

# -----
# convert_lon_lat_str
# -----


# @pytest.mark.parametrize(
#     "test_input, expected_output",  # the input and expected output for each test case
#     [
#         ('74 0.1878 N', 74.00313),
#         ('140 0.1198 W', -140.001997),
#         ('0 0 N', 0),
#         ('0 0 S', 0),
#         ('0 0 E', 0),
#         ('0 0 W', 0),
#         ('180 0 E', 180),
#         ('180 0 W', -180),
#         ('90 0 N', 90),
#         ('90 0 S', -90),
#         ('0 30 N', 0.5),
#         ('0 30 S', -0.5),
#         ('30 0 E', 30),
#         ('30 0 W', -30),
#         # ('0 0', None),
#         # ('0 0 X', None),
#         ('0 0 N ', 0),
#         (' 0 0 N', 0),
#         ('0 0 N', 0),
#         ('0 0 S ', 0),
#         (' 0 0 S', 0),
#         ('0 0 S', 0),
#         ('0 0 E ', 0),
#         (' 0 0 E', 0),
#         ('0 0 E', 0),
#         ('0 0 W ', 0),
#         (' 0 0 W', 0),
#         ('0 0 W', 0),
#         ('0 30 N ', 0.5),
#         (' 0 30 N', 0.5),
#         ('0 30 N', 0.5),
#         ('0 30 S ', -0.5),
#         (' 0 30 S', -0.5),
#         ('0 30 S', -0.5),
#         ('30 0 E ', 30),
#         (' 30 0 E', 30),
#         ('30 0 E', 30),
#         ('30 0 W ', -30),
#         (' 30 0 W', -30),
#         ('30 0 W', -30),
#         ('74 0.1878 N', 74.00313),
#         ('140 0.1198 W', -140.001997),
#         ('-74 0.1878 S', -74.00313),
#         ('-140 0.1198 E', 140.001997),
#         ('74 30 N', 74.5),
#         ('-74 30 S', -74.5),
#         ('30 30 E', 30.5),
#         ('-30 30 W', -30.5)
#     ],
# )
# def test_convert_lon_lat_str(test_input, expected_output):
#     assert np.abs(convert_lon_lat_str(test_input) - expected_output) < 1e-6


# test_convert_lon_lat_str()



def test_valid_latitude_conversion():
    # Test that the function correctly converts latitude strings in the format 'degrees minutes direction'
    # with direction values of 'N' and 'S'.
    test_cases = [
        ('45 30 N', 45.5),
        ('60 15.5 S', -60.25833),
        ('0 0 N', 0),
        ('90 0 N', 90),
        ('90 0 S', -90),
        ('45 0 N', 45),
        ('45 0 S', -45),
        ('89 59 N', 89.98333),
        ('89 59 S', -89.98333)
    ]
    for test_input, expected_output in test_cases:
        assert convert_lon_lat_str(test_input) == pytest.approx(expected_output, abs=0.001)

def test_assertion_error_on_non_string_input():
    # Test that the function raises an AssertionError when given an input that is not a string.
    with pytest.raises(AssertionError):
        convert_lon_lat_str(123)
    with pytest.raises(AssertionError):
        convert_lon_lat_str(['45 30 N'])



# TODO: add some simple test for changing: column name, dim value and reset_index=True
test_cases = [
    # test cases for various numpy arrays
    (np.array([1, 2, 3]), "data", "_dim_", False,
     pd.DataFrame({"data": [1, 2, 3]}, index=pd.MultiIndex.from_product([np.arange(3)], names=["_dim_0"]))),
    (np.array([[1, 2], [3, 4]]), "data", "_dim_", False, pd.DataFrame({"data": [1, 2, 3, 4]},
                                                                      index=pd.MultiIndex.from_product(
                                                                          [np.arange(2), np.arange(2)],
                                                                          names=["_dim_0", "_dim_1"]))),
    (np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]]), "data", "_dim_", False,
     pd.DataFrame({"data": [1, 2, 3, 4, 5, 6, 7, 8]},
                  index=pd.MultiIndex.from_product([np.arange(2), np.arange(2), np.arange(2)],
                                                   names=["_dim_0", "_dim_1", "_dim_2"]))),
    (np.ones((3, 3), dtype='float'), "data", "_dim_", False, pd.DataFrame({"data": [1.0] * 9},
                                                                          index=pd.MultiIndex.from_product(
                                                                              [np.arange(3), np.arange(3)],
                                                                              names=["_dim_0", "_dim_1"]))),
    (np.array([1, 2, 3, 4]), "data", "_dim_", False,
     pd.DataFrame({"data": [1, 2, 3, 4]}, index=pd.MultiIndex.from_product([np.arange(4)], names=["_dim_0"]))),
    (5, "data", "_dim_", False,
     pd.DataFrame({"data": [5]}, index=pd.MultiIndex.from_product([np.arange(1)], names=["_dim_0"]))),
    ("hello", "data", "_dim_", False,
     pd.DataFrame({"data": ["hello"]}, index=pd.MultiIndex.from_product([np.arange(1)], names=["_dim_0"]))),
    # test cases for edge cases and unexpected inputs
    (pd.DataFrame({"a": [1, 2, 3]}), "data", "_dim_", False, AssertionError),
    (np.array([]), "data", "_dim_", False,
     pd.DataFrame({"data": []}, dtype=object, index=pd.MultiIndex(levels=[[]], codes=[[]], names=["_dim_0"]))),
    (np.array([1]), "data", "_dim_", False,
     pd.DataFrame({"data": [1]}, index=pd.MultiIndex.from_product([np.arange(1)], names=["_dim_0"]))),
    (np.array([[1, 2, 3]]), "data", "_dim_", False, pd.DataFrame({"data": [1, 2, 3]}, index=pd.MultiIndex.from_product(
        [np.arange(1), np.arange(3)], names=["_dim_0", "_dim_1"]))),
    (np.array([[1], [2], [3]]), "data", "_dim_", False, pd.DataFrame({"data": [1, 2, 3]},
                                                                     index=pd.MultiIndex.from_product(
                                                                         [np.arange(3), np.arange(1)],
                                                                         names=["_dim_0", "_dim_1"]))),
    (np.array([[[1, 2]]]), "data", "_dim_", False, pd.DataFrame({"data": [1, 2]}, index=pd.MultiIndex.from_product(
        [np.arange(1), np.arange(1), np.arange(2)], names=["_dim_0", "_dim_1", "_dim_2"]))),
    (np.array([[[1], [2]]]), "data", "_dim_", False, pd.DataFrame({"data": [1, 2]}, index=pd.MultiIndex.from_product(
        [np.arange(1), np.arange(2), np.arange(1)], names=["_dim_0", "_dim_1", "_dim_2"]))),
    (np.array([[[1]], [[2]]]), "data", "_dim_", False, pd.DataFrame({"data": [1, 2]}, index=pd.MultiIndex.from_product(
        [np.arange(2), np.arange(1), np.arange(1)], names=["_dim_0", "_dim_1", "_dim_2"]))),
]

# define the function to be called for each test case
@pytest.mark.parametrize("x, name, dim_prefix, reset_index, expected", test_cases)
def test_array_to_dataframe(x, name, dim_prefix, reset_index, expected):
    # if the expected result is an AssertionError, check that the function raises an AssertionError
    if isinstance(expected, type):

        with pytest.raises(AssertionError):
            array_to_dataframe(x, name, dim_prefix, reset_index)
    # otherwise, check that the function returns the expected result
    else:
        result = array_to_dataframe(x, name, dim_prefix, reset_index)
        pd.testing.assert_frame_equal(result, expected)




@pytest.mark.parametrize(
    "df, val_col, idx_col, dropna, fill_val, expected_output",
    [
        # (
        #     # 1
        #     pd.DataFrame({"values": [1, 2, 3, 4]}, index=pd.MultiIndex.from_product([np.arange(2), np.arange(2)], names=['dim1', 'dim2'])),
        #     "values",
        #     None, #["dim1", "dim2"],
        #     True,
        #     np.nan,
        #     np.array([[1, 2], [3, 4]])
        # ),
        (
            pd.DataFrame({"dim1": [0, 0, 1, 1], "dim2": [0, 1, 0, 1], "values": [1., 2., 3., 4.]}),
            "values",
            ["dim1", "dim2"],
            True,
            np.nan,
            np.array([[1, 2], [3, 4]], dtype="float64")
        ),
        (
            pd.DataFrame({"dim1": [0, 0, 1, 1], "dim2": [0, 1, 0, 1], "values": [1, np.nan, 3, 4]}),
            "values",
            ["dim1", "dim2"],
            True,
            np.nan,
            np.array([[1, np.nan], [3, 4]])
        ),
        (   # fill missing location with 0 instead of nan
            pd.DataFrame({"dim1": [0, 1, 1], "dim2": [0, 0, 1], "values": [1, 3, 4]}),
            "values",
            ["dim1", "dim2"],
            False,
            0,
            np.array([[1, 0], [3, 4]])
        ),
        (
            # provide dataframe with index, don't specify idx_col
            pd.DataFrame({"dim0": [0, 0, 1, 1], "dim1": [0, 1, 0, 1], "data": [1, 2, 3, 4]}).set_index(["dim0", "dim1"]),
            "data",
            None,
            True,
            -9999,
            np.array([[1, 2], [3, 4]])
        ),
        (
            # idx_col specified by does not reference columns
            pd.DataFrame({"dim1": [0, 0, 1, 1], "dim2": [0, 1, 0, 1], "values": [1, 2, 3, 4]}).set_index(["dim1", "dim2"]),
            "values",
            [0, 1],
            True,
            np.nan,
            AssertionError
        ),
        (
            # specify the wrong columns for index / dimension
            pd.DataFrame({"dim1": [0, 0, 1, 1], "dim2": [0, 1, 0, 1], "values": [1, 2, 3, 4]}),
            "values",
            ["dim1", "dim3"],
            True,
            np.nan,
            AssertionError
        ),
        (
            # dim are float, must be int - as used to to specify index location in output array
            pd.DataFrame({"dim1": [0., 0., 1., 1.], "dim2": [0., 1., 0., 2.], "values": [1, 2, 3, 4]}),
            "values",
            ["dim1", "dim2"],
            True,
            np.nan,
            AssertionError
        ),
        (
                # when val_col is int should fill with int
                pd.DataFrame({"dim1": [0, 0, 1, 1], "dim2": [0, 1, 0, 2], "values": [1, 2, 3, 4]}),
                "values",
                ["dim1", "dim2"],
                True,
                -999,
                np.array([[1, 2, -999], [3, -999, 4]])
        ),
    ]
)
def test_dataframe_to_array(df, val_col, idx_col, dropna, fill_val, expected_output):
    if isinstance(expected_output, type) and issubclass(expected_output, Exception):
        with pytest.raises(expected_output):
            dataframe_to_array(df, val_col, idx_col, dropna, fill_val)
    else:
        result = dataframe_to_array(df, val_col, idx_col, dropna, fill_val)
        assert_array_equal(result, expected_output)


# -----
# match
# -----

# Test exact matching with integers
@pytest.mark.parametrize("x, y, expected", [
    ([1, 2, 3], [1, 2, 3], [0, 1, 2]),  # exact match
    (2, [1, 2, 3], [1]),  # check none array like values work
    ([1, 2, 3], [3, 2, 1], [2, 1, 0]),  # match in reverse order
    ([1, 2, 3], [1, 2, 2], AssertionError),  # trying to match a value in x not in y should give AssertionError
    ([1, 2, 2, 1], [1, 2], [0, 1, 1, 0]),  # y is shorter than x
    ([1, 3, 4, 1], [1, 2, 3, 3, 4, 4], [0, 2, 4, 0]),  # duplicate values in y, first location will be matched
    ([1, 2], [1, 2, 3, 4], [0, 1]),  # x is shorter than y
])
def test_exact_matching_integers(x, y, expected):
    if isinstance(expected, type) and issubclass(expected, Exception):
        with pytest.raises(expected):
            match(x, y)
    else:
        indices = match(x, y)
        np.testing.assert_array_equal(indices, expected)


# Test exact matching with strings
@pytest.mark.parametrize("x, y, expected", [
    (["apple", "banana", "cherry"], ["apple", "banana", "cherry"], [0, 1, 2]),  # exact match
    (["apple", "banana", "cherry"], ["cherry", "banana", "apple"], [2, 1, 0]),  # match in reverse order
    (["apple", "banana", "cherry"], ["apple", "banana", "banana"], AssertionError),  # x missing value in y
    (["apple", "banana", "banana"], ["apple", "banana", "cherry"], [0, 1, 1]),  # duplicate values in x
    (["apple", "banana"], ["apple", "banana", "apple", "banana"], [0, 1]), # duplicate values in y, first location will be matched
    (["apple", "banana"], ["apple", "banana", "cherry"], [0, 1]),  # x is shorter than y
])
def test_exact_matching_strings(x, y, expected):
    if isinstance(expected, type) and issubclass(expected, Exception):
        with pytest.raises(expected):
            match(x, y)
    else:
        indices = match(x, y)
        np.testing.assert_array_equal(indices, expected)

# Test exact matching with floats
@pytest.mark.parametrize("x, y, exact, expected", [
    ([1.0, 2.0, 3.0], [1.0, 2.0, 3.0], True, [0, 1, 2]),  # exact match
    ([1.0, 2.0, 3.0], [3.0, 2.0, 1.0], True, [2, 1, 0]),  # match in reverse order
    ([1.0, 2.0, 3.0], [1.0, 2.0, 2.0], True, AssertionError),  # x missing value in y
    ([1.0, 2.0, 3.0], [1.0000000005, 2.0, 3.0], False, [0, 1, 2]),  # match within tolerance
    ([1.0, 2.0, 3.0], [1.000001, 2.0, 3.0], True, AssertionError),  # no match within tolerance
])
def test_exact_matching_floats(x, y, exact, expected):
    if isinstance(expected, type) and issubclass(expected, Exception):
        with pytest.raises(expected):
            match(x, y, exact)
    else:
        indices = match(x, y, exact)
        np.testing.assert_array_equal(indices, expected)


# Test matching within tolerance with floats
@pytest.mark.parametrize("x, y, expected", [
    ([1.0, 2.0, 3.0], [1.0, 2.0, 3.0], [0, 1, 2]),  # exact match
    ([1.0, 2.0, 3.0], [3.0, 2.0, 1.0], [2, 1, 0]),  # match in reverse order
    ([1.0, 2.0, 3.0], [1.0, 2.0, 2.0], AssertionError),  # multiple matches
    ([1.0, 2.0, 3.0], [1.0000001, 2.0, 3.0], [0, 1, 2]),  # match within tolerance
    ([1.0, 2.0, 3.0], [1.00001, 2.0, 3.0], AssertionError),  # no match within tolerance
])
def test_matching_within_tolerance(x, y, expected):
    if isinstance(expected, type) and issubclass(expected, Exception):
        with pytest.raises(expected):
            match(x, y, exact=False, tol=1e-6)
    else:
        indices = match(x, y, exact=False, tol=1e-6)
        np.testing.assert_array_equal(indices, expected)


# Test that AssertionError is raised when element in x is not found in y
def test_assertion_error_not_found():
    with pytest.raises(AssertionError):
        match([1, 2, 3], [4, 5, 6])

# Test that AttributeError is raised when x is a str and y is an array
# def test_AttributeError_error_not_arrays():
#     with pytest.raises(AttributeError):
#         match("not an array", [1, 2, 3])

# Test that AssertionError is raised when exact=False and the differences is above tol
def test_assertion_error_not_floats():
    with pytest.raises(AssertionError):
        match([1, 2, 3], [1 + 1e-6, 2.0, 3.0], exact=False, tol=1e-7)



# ----
# pandas_to_dict
# -----


@pytest.mark.parametrize(
    "input_data, expected_output",
    [
        # Test case 1: Test for a pandas Series with string values
        (
            pd.Series(["John", "Jane"]),
            {0: "John", 1: "Jane"},
        ),
        # Test case 2: Test for a pandas Series with integer values
        (
            pd.Series([30, 25]),
            {0: 30, 1: 25},
        ),
        # Test case 3: Test for a pandas DataFrame with one row
        (
            pd.DataFrame({"name": ["John"], "age": [30]}),
            {"name": "John", "age": 30},
        ),
        # Test case 4: Test for a pandas DataFrame with multiple rows
        (
            pd.DataFrame({"name": ["John", "Jane"], "age": [30, 25]}),
            AssertionError
        ),
        # Test case 5: Test for a dictionary with string values
        (
            {"name": ["John", "Jane"], "age": [30, 25]},
            {"name": ["John", "Jane"], "age": [30, 25]},
        ),
        # Test case 6: Test for a dictionary with integer values
        (
            {"name": [30, 25], "age": [30, 25]},
            {"name": [30, 25], "age": [30, 25]},
        ),
        # Test case 7: Test for an empty dictionary
        (
            {},
            {},
        ),
        # # Test case 8: Test for a non-pandas, non-dictionary input
        # - should return input exactly, with user warning
        # (
        #     "John",
        #     "John",
        # ),
    ],
)
def test_pandas_to_dict(input_data, expected_output):

    if isinstance(expected_output, type) and issubclass(expected_output, Exception):
        with pytest.raises(expected_output):
            pandas_to_dict(input_data)
    else:
        assert pandas_to_dict(input_data) == expected_output

# Test case 9: Test for a pandas DataFrame with one row and multiple columns
def test_pandas_to_dict_single_row_multiple_columns():
    data = {"name": ["John"], "age": [30], "gender": ["male"]}
    df = pd.DataFrame(data)
    expected_output = {"name": "John", "age": 30, "gender": "male"}
    assert pandas_to_dict(df) == expected_output


# ----
# grid_2d_flatten
# ----


@pytest.mark.parametrize(
    "x_range, y_range, grid_res, step_size, num_step, center, expected_shape",
    [
        # Test basic functionality with default arguments
        ([-1, 1], [-1, 1], None, None, None, True, (9, 2)),
        # Test grid resolution argument
        ([-1, 1], [-1, 1], 0.5, None, None, True, (25, 2)),
        # Test step size argument
        ([-1, 1], [-1, 1], None, 0.5, None, True, (9, 2)),
        # Test num step argument
        ([-1, 1], [-1, 1], None, None, 5, True, (25, 2)),
        # Test edge case of single point grid
        ([0, 0], [0, 0], None, None, None, True, (1, 2)),
        # Test edge case of empty grid
        ([1, 0], [1, 0], None, None, None, True, (0, 2)),
        # Test edge case of non-square grid
        ([-1, 1], [-1, 0], None, None, None, True, (6, 2)),
        # Test edge case of non-centered grid
        ([-1, 1], [-1, 1], None, None, None, False, (8, 2)),
    ],
)
def test_grid_2d_flatten(x_range, y_range, grid_res, step_size, num_step, center, expected_shape):
    # Call the function with the given arguments
    result = grid_2d_flatten(x_range, y_range, grid_res, step_size, num_step, center)
    # Check that the shape of the result is as expected
    assert result.shape == expected_shape
    # Check that the x and y values are within the specified range
    assert np.all(result[:, 0] >= x_range[0])
    assert np.all(result[:, 0] <= x_range[1])
    assert np.all(result[:, 1] >= y_range[0])
    assert np.all(result[:, 1] <= y_range[1])


@pytest.mark.parametrize(
    "x_range, y_range, grid_res, step_size, num_step, center, expected_shape",
    [
        # Test basic functionality with default arguments
        ([-1, 1], [-1, 1], None, None, None, True, AssertionError),
        # Test grid resolution argument
        ([-1, 1], [-1, 1], 0.5, None, None, True, (16, 2)),
        # Test step size argument
        ([-1, 1], [-1, 1], None, 0.5, None, True, (16, 2)),
        # Test num step argument
        ([-1, 1], [-1, 1], None, None, 5, True, (16, 2)),
        # Test num step argument
        ([-1, 1], [-1, 1], None, None, 5, False, (25, 2)),
        # Test edge case of not space - taking center
        ([0, 0], [0, 0], None, None, 1, True, (0, 2)),
        # Test edge case of not space - taking edges will return the single point
        ([0, 0], [0, 0], None, None, 1, False, (1, 2)),
        # Unit square, num_step = 1, not taking center
        ([0, 1], [0, 1], None, None, 2, False, (4, 2)),
        # Test edge case of empty grid
        # ([1, 0], [1, 0], None, None, None, True, (0, 2)),
        # Test edge case of non-square grid
        ([-1, 1], [-1, 0], None, 0.5, None, True, (8, 2)),
        # Test edge case of non-square grid - with edges
        ([-1, 1], [-1, 0], None, 0.5, None, False, (5 * 3, 2)),
    ]
)
def test_grid_2d_flatten(x_range, y_range, grid_res, step_size, num_step, center, expected_shape):
    # if expecting and error
    if isinstance(expected_shape, type) and issubclass(expected_shape, Exception):
        with pytest.raises(expected_shape):
            grid_2d_flatten(x_range, y_range, grid_res, step_size, num_step, center)
    else:
        # Call the function with the given arguments
        result = grid_2d_flatten(x_range, y_range, grid_res, step_size, num_step, center)
        # Check that the result has the expected shape
        assert result.shape == expected_shape
        # Check that the result is a numpy array
        assert isinstance(result, np.ndarray)
        # Check that the result contains only finite values
        assert np.all(np.isfinite(result))


# ----
# dataframe_to_array
# ----


@pytest.fixture
def sample_df():
    return pd.DataFrame({
        'dim1': [0, 0, 1, 1],
        'dim2': [0, 1, 0, 1],
        'values': [1., 2., 3., 4.]
    })


@pytest.fixture
def sample_df_missing():
    return pd.DataFrame({
        'dim1': [0, 0, 1, 1, 2],
        'dim2': [0, 1, 0, 1, 0],
        'values': [1, 2, np.nan, 4, 5]
    })


@pytest.fixture
def sample_df_multiindex():
    return pd.DataFrame({
        'values': [1., 2., 3., 4.]
    }, index=pd.MultiIndex.from_tuples([(0, 0), (0, 1), (1, 0), (1, 1)], names=['dim1', 'dim2']))


@pytest.fixture
def sample_df_multiindex_missing():
    return pd.DataFrame({
        'values': [1, 2, np.nan, 4, 5]
    }, index=pd.MultiIndex.from_tuples([(0, 0), (0, 1), (1, 0), (1, 1), (2, 0)], names=['dim1', 'dim2']))


def test_dataframe_to_array_basic(sample_df):
    expected = np.array([[1, 2], [3, 4]], dtype='float64')
    result = dataframe_to_array(sample_df, 'values', ['dim1', 'dim2'])
    assert np.array_equal(result, expected)


def test_dataframe_to_array_missing(sample_df_missing):
    expected = np.array([[1, 2], [np.nan, 4], [5, np.nan]])
    result = dataframe_to_array(sample_df_missing, 'values', ['dim1', 'dim2'])
    # assert np.array_equal(result, expected)
    np.testing.assert_array_equal(result, expected)


def test_dataframe_to_array_multiindex(sample_df_multiindex):
    expected = np.array([[1, 2], [3, 4]])
    result = dataframe_to_array(sample_df_multiindex, 'values')
    np.testing.assert_array_equal(result, expected)


def test_dataframe_to_array_multiindex_missing(sample_df_multiindex_missing):
    expected = np.array([[1, 2], [np.nan, 4], [5, np.nan]])
    result = dataframe_to_array(sample_df_multiindex_missing, 'values', dropna=False)
    np.testing.assert_array_equal(result, expected)


def test_dataframe_to_array_dropna(sample_df_missing):
    # dropna is applied to the dataframe, will still have missing values in output array
    expected = np.array([[1, 2], [np.nan, 4], [5, np.nan]])
    result = dataframe_to_array(sample_df_missing, 'values', ['dim1', 'dim2'], dropna=True)
    np.testing.assert_array_equal(result, expected)


def test_dataframe_to_array_fill_val(sample_df_missing):
    expected = np.array([[1, 2], [0, 4], [5, 0]], dtype=float)
    result = dataframe_to_array(sample_df_missing, 'values', ['dim1', 'dim2'], fill_val=0.)
    assert np.array_equal(result, expected)


def test_dataframe_to_array_idx_col_not_in_df(sample_df):
    with pytest.raises(AssertionError):
        dataframe_to_array(sample_df, 'values', ['dim1', 'dim3'])


def test_dataframe_to_array_non_integer_dim(sample_df):
    sample_df['dim1'] = sample_df['dim1'].astype(float)
    with pytest.raises(AssertionError):
        dataframe_to_array(sample_df, 'values', ['dim1', 'dim2'])

# ----
# to_array
# ----

# TODO: test multiple inputs / outputs
# TODO: test a generator is returned
# TODO: use list comprehensions instead
@pytest.mark.parametrize(
    "inputs, expected_outputs",
    [
        # test input array of integers
        ([1, 2, 3], np.array([1, 2, 3])),
        # test input array of floats
        ([1.0, 2.5, 3.0], np.array([1.0, 2.5, 3.0])),
        # test input array of strings
        (["a", "b", "c"], np.array(["a", "b", "c"])),
        # test input array of booleans
        ([True, False, True], np.array([True, False, True])),
        # test input array of numpy booleans
        ([np.bool_(True), np.bool_(False)], np.array([True, False])),
        # test input array of numpy integers
        ([np.int32(1), np.int64(2)], np.array([1, 2])),
        # test input array of numpy floats
        ([np.float32(1.0), np.float64(2.5)], np.array([1.0, 2.5])),
        # test input array of datetime.date objects
        (
            [datetime.date(2021, 1, 1), datetime.date(2021, 1, 2)],
            np.array(["2021-01-01", "2021-01-02"], dtype="datetime64[D]"),
        ),
        # test input array of numpy datetime64 objects
        (
            [np.datetime64("2021-01-01"), np.datetime64("2021-01-02")],
            np.array(["2021-01-01", "2021-01-02"], dtype="datetime64[D]"),
        ),
        # test input array of mixed data types - if contains
        (
            [1, "a", True, np.int64(2), np.float32(2.5)],
            np.array(['1', 'a', 'True', '2', '2.5'], dtype='<U32'),
        ),
        # test input array with None value
        ([None], [np.array(None, dtype=object)]),
        # test multiple input arrays
        # (
        #     [1, 2, 3], [4, 5, 6], ["2021-01-01"]
        #     [np.array([1, 2, 3]), np.array([4, 5, 6]), np.array(["2021-01-01"])],
        # ),
        # pandas Series
        (pd.Series([1, 2, 3]), np.array([1, 2, 3]))
    ],
)
def test_to_array(inputs, expected_outputs):
    # convert the inputs to arrays and compare to expected outputs
    _, = to_array(inputs)
    np.testing.assert_array_equal(_, expected_outputs)

# TODO: move some of the below into the above
# def test_to_array():
#     # Test empty input
#     assert list(to_array()) == []
#
#     # Test a single numpy array
#     x = np.array([1, 2, 3])
#     assert list(to_array(x)) == [x]
#
#     # Test a single list
#     assert list(to_array([1, 2, 3])) == [np.array([1, 2, 3])]
#
#     # Test a single tuple
#     assert list(to_array((1, 2, 3))) == [np.array([1, 2, 3])]
#
#     # Test a single integer
#     assert list(to_array(1)) == [np.array([1], dtype=int)]
#
#     # Test a single float
#     assert list(to_array(1.0)) == [np.array([1.0], dtype=float)]
#
#     # Test a single boolean
#     assert list(to_array(True)) == [np.array([True], dtype=bool)]
#
#     # Test a single string
#     assert list(to_array("foo")) == [np.array(["foo"])]
#
#     # Test a single datetime.date
#     assert list(to_array(datetime.date(2022, 1, 1))) == [np.array(["2022-01-01"], dtype="datetime64[D]")]
#
#
#     # Test multiple arguments of various types
#     x = np.array([1, 2, 3])
#     y = np.array([4, 5, 6])
#     z = datetime.date(2021, 1, 1)
#
#     expected_output = [
#         np.array([1, 2, 3]),
#         np.array([4, 5, 6]),
#         np.array(["2021-01-01"], dtype="datetime64[D]")
#     ]
#
#     assert list(to_array(x, y, z)) == expected_output
#
#     # Test None input
#     assert list(to_array(None)) == [np.array([])]
#
#     # Test numpy integer inputs
#     assert list(to_array(np.int8(1))) == [np.array([1], dtype=np.int8)]
#     assert list(to_array(np.int16(1))) == [np.array([1], dtype=np.int16)]
#     assert list(to_array(np.int32(1))) == [np.array([1], dtype=np.int32)]
#     assert list(to_array(np.int64(1))) == [np.array([1], dtype=np.int64)]
#
#     # Test numpy float inputs
#     assert list(to_array(np.float16(1.0))) == [np.array([1.0], dtype=np.float16)]
#     assert list(to_array(np.float32(1.0))) == [np.array([1.0], dtype=np.float32)]
#     assert list(to_array(np.float64(1.0))) == [np.array([1.0], dtype=np.float64)]
#
#     # Test numpy boolean inputs
#     assert list(to_array(np.bool(True))) == [np.array([True], dtype=np.bool)]
#     assert list(to_array(np.bool_(True))) == [np.array([True], dtype=np.bool_)]
#     assert list(to_array(np.bool8(True))) == [np.array([True], dtype=np.bool8)]
#
#     # Test numpy datetime64 inputs
#     x = np.datetime64('2023-04-28')
#     assert list(to_array(x)) == [np.array([x], dtype="datetime64[D]")]
#
#
#     # # Test unsupported input type
#     # with pytest.warns(UserWarning, match="Data type <class 'set'> is not configured in to_array."):
#     #     assert list(to_array({1, 2, 3})) == [np.array([{1,


# ----
# config_func
# ----

# TODO: move combinations could be added for config_func
@pytest.mark.parametrize(
    "func, source, args, kwargs, col_args, col_kwargs, df, filename_as_arg, filename, col_numpy, expected_output", [
        # Test: Simple function (no DataFrame or filename)
        (np.sum, None, np.array([1, 2, 3]), {}, None, None, None, False, None, True, 6),

        # Test: Function with args and kwargs
        (np.arange, None, 3, {'dtype': float}, None, None, None, False, None, True, np.array([0., 1., 2.])),

        # Test: Function with source
        ("cumprod", "numpy", [np.array([1, 2, 3, 4])], {}, None, None, None, False, None, True, np.array([1, 2, 6, 24])),

        # Test: String / lambda function with operators
        ("lambda x, y: x * y", None, [2, 3], {}, None, None, None, False, None, True, 6),

        # Test: DataFrame as input (col_args and col_kwargs)
        (np.dot, None, [], {}, ['col1'], {'b': 'col2'}, pd.DataFrame({'col1': [2, 3], 'col2': [4, 5]}), False,
         None, True, 23),

        # Test: Filename as argument - provide a lambda function let the argument be the 'new' suffix / file type
        (lambda x, y: re.sub(r"\..*", f".{y}", x), None, "png", {}, None, None, None, True, "testfile.txt", True, "testfile.png")
    ])
def test_config_func(func: Callable, source: str, args: list, kwargs: dict, col_args: list, col_kwargs: dict,
                     df: pd.DataFrame, filename_as_arg: bool, filename: str, col_numpy: bool, expected_output):
    output = config_func(func, source, args, kwargs, col_args, col_kwargs, df, filename_as_arg, filename, col_numpy)
    assert np.all(output == expected_output), f"Expected {expected_output}, but got {output}"


@pytest.mark.parametrize("func, source, args, kwargs, col_args, col_kwargs, df, filename_as_arg, filename, col_numpy, expected_exception", [
    # Test: Invalid function (not a string or callable)
    (123, None, [], {}, None, None, None, False, None, True, AssertionError),

    # Test: DataFrame not provided, but col_args or col_kwargs are
    (np.sum, None, [], {}, ['col1'], {}, None, False, None, True, AssertionError),

    # Test: NameError on eval(func) and source is None
    ("non_existent_function", None, [], {}, None, None, None, False, None, True, AssertionError),
])
def test_config_func_exceptions(func: Callable, source: str, args: list, kwargs: dict, col_args: list, col_kwargs: dict, df: pd.DataFrame, filename_as_arg: bool, filename: str, col_numpy: bool, expected_exception):
    with pytest.raises(expected_exception):
        config_func(func, source, args, kwargs, col_args, col_kwargs, df, filename_as_arg, filename, col_numpy)


# ----
# EASE2toWGS84_New
# ----

@pytest.mark.parametrize("x, y, return_vals, lon_0, lat_0, expected_output", [
    # Test: Example from docstring
    (1000000, 2000000, 'both', 0, 90, (153.434948822922, 69.86894542225777)),
    # Test: Return only longitude
    (1000000, 2000000, 'lon', 0, 90, 153.434948822922),
    # Test: Return only latitude
    (1000000, 2000000, 'lat', 0, 90, 69.86894542225777),
    # Test: Different center of EASE2 grid - these values were just generated, not the best test
    (1000000, 2000000, 'both', 45, 45, (64.06588754736106, 61.89165587880854)),
    # Test: x = 0 and y = 0 for different centering positions
    (0, 0, 'both', 0, 90, (0, 90)),
    (0, 0, 'both', 0, 0, (0, 0)),
    (0, 0, 'both', 45, 45,  (45, 45)),
    (0, 0, 'both', -60, 80, (-60, 80)),
    # 60 nautical miles north from equator, should be one degree (?)
    (0, 110_573, 'both', 0, 0, (0, 1)),
    # 1 degree longitude at equator is (approximately) 111_321 meters (?_
    (111_321, 0, 'both', 0, 0, (1.000026, 0)),
])
def test_EASE2toWGS84_New(x: float, y: float, return_vals: str, lon_0: float, lat_0: float, expected_output):

    output = EASE2toWGS84(x, y, return_vals, lon_0, lat_0)
    np.testing.assert_array_almost_equal(output, expected_output, decimal=6)

# Test cases for expected exceptions
@pytest.mark.parametrize("x, y, return_vals, lon_0, lat_0, expected_exception", [
    # Test: Invalid return_vals option
    (1000000, 2000000, 'invalid_option', 0, 90, AssertionError),
])
def test_EASE2toWGS84_New_exceptions(x: float, y: float, return_vals: str, lon_0: float, lat_0: float,
                                     expected_exception):

    with pytest.raises(expected_exception):
        EASE2toWGS84(x, y, return_vals, lon_0, lat_0)


@pytest.mark.parametrize("x, y, lon_0, lat_0", [
    (1000_000, 2000_000, 0, 90),
    (0, 0, 0, 90),
    (0, 0, 0, 0),
    (0, 0, 0, -90),
    (-100_000, -200_000, 0, 90),
    (-2000_000, -1000_000, 0, 90),
    (100_000, -200_000, -0.136439, 51.507359),
    (-40_345, 55_124, -0.136439, 51.507359)
])
def test_EASE2toWGS84_New_check_inverse(x: float, y: float, lon_0: float, lat_0: float):
    # show WGS84toEASE2_New is the inverse of EASE2toWGS84_New
    lon, lat = EASE2toWGS84(x, y, 'both', lon_0, lat_0)
    x_, y_ = WGS84toEASE2(lon, lat, 'both', lon_0, lat_0)

    np.testing.assert_array_almost_equal([x, y], [x_, y_], decimal=3)


# -----
# WGS84toEASE2_New
# -----

# Helper function to create a transformer
# from pyproj import Transformer
# def _create_transformer(lon_0, lat_0):
#     EASE2 = f"+proj=laea +lon_0={lon_0} +lat_0={lat_0} +x_0=0 +y_0=0 +ellps=WGS84 +towgs84=0,0,0,0,0,0,0 +units=m +no_defs"
#     WGS84 = "+proj=longlat +ellps=WGS84 +datum=WGS84 +no_defs"
#     return Transformer.from_crs(WGS84, EASE2)

@pytest.mark.parametrize("lon, lat, return_vals, lon_0, lat_0, expected_output", [
    # Test: basic functionality
    (-105.01621, 39.57422, 'both', 0, 90, (-5254767.014984061, 1409604.1043472202)),
    # Test: return_vals = 'x'
    (-105.01621, 39.57422, 'x', 0, 90, -5254767.014984061),
    # Test: return_vals = 'y'
    (-105.01621, 39.57422, 'y', 0, 90, 1409604.1043472202),
    # Test: different central meridian (lon_0)
    (-105.01621, 39.57422, 'both', -90, 90, (-1409604.1043472204, -5254767.014984061)),
    # Test: different central parallel (lat_0)
    (-105.01621, 39.57422, 'both', 0, 45, (-5882390.5299294265, 4659494.127247092)),
])
def test_WGS84toEASE2_New(lon, lat, return_vals, lon_0, lat_0, expected_output):
    output = WGS84toEASE2(lon, lat, return_vals, lon_0, lat_0)
    np.testing.assert_array_equal(output, expected_output)

@pytest.mark.parametrize("lon, lat, return_vals, lon_0, lat_0, expected_exception", [
    # Test: Invalid return_vals
    (-105.01621, 39.57422, 'invalid', 0, 90, AssertionError),
])
def test_WGS84toEASE2_New_exceptions(lon, lat, return_vals, lon_0, lat_0, expected_exception):
    with pytest.raises(expected_exception):
        _ = WGS84toEASE2(lon, lat, return_vals, lon_0, lat_0)


# confirm WGS84toEASE2_New is the inverse of EASE2toWGS84_New

# ------
# nested_dict_literal_eval
# ------

@pytest.mark.parametrize("d, verbose, expected_output", [
    # Test: Basic nested dictionary with tuple string keys
    ({'(1, 2)': {'(3, 4)': 5}}, False, {(1, 2): {(3, 4): 5}}),

    # Test: Mixed keys with nested dictionaries
    ({'(1, 2)': {'(3, 4)': 5, 'key': 6}, 'other_key': 7}, False, {(1, 2): {(3, 4): 5, 'key': 6}, 'other_key': 7}),

    # Test: Nested dictionary with non-tuple string keys
    ({'key1': {'key2': 5}}, False, {'key1': {'key2': 5}}),

    # Test: Empty dictionary
    ({}, False, {}),
])
def test_nested_dict_literal_eval(d: dict, verbose: bool, expected_output: dict):
    output = nested_dict_literal_eval(d, verbose)
    assert output == expected_output, f"Expected {expected_output}, but got {output}"


# TODO: Test cases for expected exceptions
# @pytest.mark.parametrize("d, verbose, expected_exception", [
#     # Test: Invalid tuple string key
#     ({'(asdf, xcv)': {'(3, 4)': 5}}, False, ValueError),
# ])
# def test_nested_dict_literal_eval_exceptions(d: dict, verbose: bool, expected_exception):
#     with pytest.raises(expected_exception):
#         nested_dict_literal_eval(d, verbose)


# -----
# dataframe_to_2d_array
# -----

@pytest.mark.parametrize("df, x_col, y_col, val_col, tol, fill_val, dtype, expected_output", [
    # Test: Basic DataFrame with regularly spaced coordinates
    (pd.DataFrame({
        'x': [0, 0, 1, 1],
        'y': [0, 1, 0, 1],
        'val': [1, 2, 3, 4]
    }), 'x', 'y', 'val', 1e-9, np.nan, None, (np.array([[1, 3], [2, 4]], dtype=float),
                                              np.array([[0, 1], [0, 1]], dtype=float),
                                              np.array([[0, 0], [1, 1]], dtype=float))),
    # Test: Basic DataFrame with irregularly spaced coordinates
    (pd.DataFrame({
        'x': [0, 0, 1, 2],
        'y': [0, 1, 0, 1],
        'val': [1, 2, 3, 4]
    }), 'x', 'y', 'val', 1e-9, np.nan, None, (np.array([[1, 3, np.nan], [2, np.nan, 4]]),
                                              np.array([[0, 1, 2], [0, 1, 2]], dtype=float),
                                              np.array([[0, 0, 0], [1, 1, 1]], dtype=float))),
    # Test: DataFrame with float coordinates and dtype provided
    (pd.DataFrame({
        'x': [0.0, 0.0, 1.0, 1.0],
        'y': [0.0, 1.0, 0.0, 1.0],
        'val': [1.5, 2.5, 3.5, 4.5]
    }), 'x', 'y', 'val', 1e-9, np.nan, float,
     (np.array([[1.5, 3.5], [2.5, 4.5]]), np.array([[0.0, 1.0], [0.0, 1.0]]), np.array([[0.0, 0.0], [1.0, 1.0]]))),
    # Test: DataFrame with negative coordinates
    (pd.DataFrame({
        'x': [-1, -1, 0, 0],
        'y': [-1, 0, -1, 0],
        'val': [1, 2, 3, 4]
    }), 'x', 'y', 'val', 1e-9, np.nan, None,
     (np.array([[1, 3], [2, 4]]), np.array([[-1, 0], [-1, 0]]), np.array([[-1, -1], [0, 0]]))),
])
def test_dataframe_to_2d_array(df: pd.DataFrame, x_col: str, y_col: str, val_col: str, tol: float, fill_val: float, dtype, expected_output: tuple):
    output = dataframe_to_2d_array(df, x_col, y_col, val_col, tol, fill_val, dtype)
    assert np.allclose(output[0], expected_output[0], equal_nan=True), f"Expected {expected_output[0]}, but got {output[0]}"
    assert np.array_equal(output[1], expected_output[1]), f"Expected {expected_output[1]}, but got {output[1]}"
    assert np.array_equal(output[2], expected_output[2]), f"Expected {expected_output[2]}, but got {output[2]}"


# Test Exceptions
@pytest.mark.parametrize("df, x_col, y_col, val_col, tol, fill_val, dtype, expected_exception", [
    # Test: DataFrame missing required columns
    (pd.DataFrame({
        'x': [0, 0, 1, 1],
        'y': [0, 1, 0, 1],
    }), 'x', 'y', 'val', 1e-9, np.nan, None, AssertionError),
    # Test: DataFrame with more than one value per coordinate
    (pd.DataFrame({
        'x': [0, 0, 0, 1],
        'y': [0, 1, 1, 1],
        'val': [1, 2, 3, 4]
    }), 'x', 'y', 'val', 1e-9, np.nan, None, AssertionError),
])
def test_dataframe_to_2d_array_exceptions(df: pd.DataFrame, x_col: str, y_col: str, val_col: str, tol: float, fill_val: float, dtype, expected_exception: Exception):
    with pytest.raises(expected_exception):
        _ = dataframe_to_2d_array(df, x_col, y_col, val_col, tol, fill_val, dtype)


# -----
# transform functions
# -----

# softplus
def test_softplus_tf_validate_and_shift():
    x = np.linspace(-100, 100, 1000)
    y = softplus(x)
    # test against tensorflow for reference
    y2 = tf.math.softplus(tf.convert_to_tensor(x)).numpy()
    np.testing.assert_array_almost_equal(y, y2, decimal=14)

    # apply shift
    shift = 10.0
    y3 = softplus(x, shift=shift)

    # show adding shift is equal
    np.testing.assert_array_almost_equal(y + shift, y3, decimal=14)
    np.testing.assert_array_almost_equal(y, y3-shift, decimal=14)



def test_softplus_inverse():
    x = np.linspace(-100, 100, 1000)
    y = softplus(x)

    # test against tensorflow for reference
    x2 = tfp.math.softplus_inverse(tf.convert_to_tensor(y)).numpy()
    np.testing.assert_array_almost_equal(x, x2, decimal=14)

    # check utils
    x3 = inverse_softplus(y)
    np.testing.assert_array_almost_equal(x, x3, decimal=14)

    # check out of bounds values
    assert -np.inf == inverse_softplus(-1.0)

    # NOTE: here run into floating point precision error issues
    # shift = 10.0
    # y3 = softplus(x, shift=shift)
    # y = 1e-44, where y3-shift = 0,
    # x5 = inverse_softplus(y3-shift)
    # np.testing.assert_array_almost_equal(x, x5, decimal=14)

# sigmoid
def test_sigmoid():

    x = np.linspace(-10, 10, 1000)
    y = sigmoid(x)

    y2 = tf.math.sigmoid(tf.convert_to_tensor(x)).numpy()
    np.testing.assert_array_almost_equal(y, y2, decimal=14)

    x2 = inverse_sigmoid(y)
    # NOTE: lowered tolerance to 12 decimal places
    np.testing.assert_array_almost_equal(x, x2, decimal=12)

    high, low = 1, -1
    y = sigmoid(x, low, high)
    np.testing.assert_array_almost_equal(y, y2 * (high - low) + low, decimal=14)

    x2 = inverse_sigmoid(y, low, high)
    np.testing.assert_array_almost_equal(x, x2, decimal=12)

    # values outside of range
    assert -np.inf == inverse_sigmoid(-1.5, -1.0, 2.0)
    assert np.inf == inverse_sigmoid(2.0, -1.0, 2.0)


# ---
# get_weighted_values
# ---

def test_single_val_col_gaussian_weight():
    """Test with single value column and gaussian weight function."""
    df = pd.DataFrame({
        'ref_col': [0, 1, 2, 3],
        'dist_to_col': [1, 2, 3, 4],
        'value1': [10, 20, 30, 40]
    })
    result = get_weighted_values(df, 'ref_col', 'dist_to_col', 'value1', lengthscale=1.0)
    expected_columns = ['ref_col', 'value1']
    assert all(column in result.columns for column in expected_columns), "Output DataFrame should contain expected columns."


def test_zero_distance():
    """if the distances are zero then, single entry"""
    df = pd.DataFrame({
        'ref_col': [0, 1, 2, 3],
        'value1': [10, 20, 30, 40]
    })
    result = get_weighted_values(df, 'ref_col', 'ref_col',
                                'value1', lengthscale=1.0)

    chk = result.merge(df, on='ref_col', how='outer', suffixes=["", "_"])

    assert np.abs(chk['value1'] - chk['value1_']).max() < 1e-15


def test_zero_distance2():
    """if the distances are zero then, and multiple entries they should average"""
    df = pd.DataFrame({
        'ref_col': [0, 1, 2, 3, 0, 1, 2, 3],
        'value1': [10, 20, 30, 40, -10, -20, -30, -40]
    })
    result = get_weighted_values(df, 'ref_col', 'ref_col',
                                'value1', lengthscale=1.0)

    assert np.abs(result['value1']).max() < 1e-15


def test_multiple_val_cols_gaussian_weight():
    """Test with multiple value columns and gaussian weight function."""
    df = pd.DataFrame({
        'ref_col': [0, 1, 0, 1],
        'dist_to_col': [1, 2, 3, 4],
        'value1': [10, 20, 30, 40],
        'value2': [100, 200, 300, 400]
    })
    result = get_weighted_values(df, 'ref_col', 'dist_to_col', ['value1', 'value2'], lengthscale=1.0)
    expected_columns = ['ref_col', 'value1', 'value2']
    assert all(column in result.columns for column in expected_columns), "Output DataFrame should contain all expected value columns."

def test_assert_lengthscale_not_provided():
    """Ensure function raises AssertionError if lengthscale is not provided."""
    df = pd.DataFrame({
        'ref_col': [0, 1],
        'dist_to_col': [2, 3],
        'value1': [10, 20]
    })
    with pytest.raises(AssertionError):
        get_weighted_values(df, 'ref_col', 'dist_to_col', 'value1')

def test_assert_invalid_weight_function():
    """Test for NotImplementedError with an invalid weight function."""
    df = pd.DataFrame({
        'ref_col': [0, 1],
        'dist_to_col': [2, 3],
        'value1': [10, 20]
    })
    with pytest.raises(NotImplementedError):
        get_weighted_values(df, 'ref_col', 'dist_to_col', 'value1', weight_function="invalid", lengthscale=1.0)

def test_drop_weight_cols_parameter():
    """Test the effect of the drop_weight_cols parameter."""
    df = pd.DataFrame({
        'ref_col': [0, 1],
        'dist_to_col': [2, 3],
        'value1': [10, 20]
    })
    result = get_weighted_values(df, 'ref_col', 'dist_to_col', 'value1', lengthscale=1.0, drop_weight_cols=False)
    # Check if weight columns exist
    assert '_w' in result.columns and 'w_value1' in result.columns, "Weight columns should be present when drop_weight_cols is False."


@pytest.mark.parametrize("ref_col,dist_to_col", [
    (['ref_col', 'dummy_col'], ['dist_to_col']),
    (['ref_col', 'dist_to_col'], ['ref_col'])
])
def test_shape_mismatch(ref_col, dist_to_col):
    """Test for shape mismatch between ref_col and dist_to_col."""
    df = pd.DataFrame({
        'ref_col': [0, 1],
        'dummy_col': [2, 4],
        'dist_to_col': [2, 3],
        'value1': [10, 20]
    })
    with pytest.raises(AssertionError):
        get_weighted_values(df, ref_col, dist_to_col, 'value1', lengthscale=1.0)
