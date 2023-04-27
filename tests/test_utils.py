
import pytest
import numpy as np
import pandas as pd
from numpy.testing import assert_array_equal
from pandas.testing import assert_frame_equal

# import the function to be tested
from PyOptimalInterpolation.utils import array_to_dataframe, \
    dataframe_to_array, match, pandas_to_dict, grid_2d_flatten, convert_lon_lat_str

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
        (
            # 1
            pd.DataFrame({"values": [1, 2, 3, 4]}, index=pd.MultiIndex.from_product([np.arange(2), np.arange(2)], names=['dim1', 'dim2'])),
            "values",
            None, #["dim1", "dim2"],
            True,
            np.nan,
            np.array([[1, 2], [3, 4]])
        ),
        (
            pd.DataFrame({"dim1": [0, 0, 1, 1], "dim2": [0, 1, 0, 1], "values": [1, 2, 3, 4]}),
            "values",
            ["dim1", "dim2"],
            True,
            np.nan,
            np.array([[1, 2], [3, 4]])
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
            np.nan,
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
        assert_array_equal(dataframe_to_array(df, val_col, idx_col, dropna, fill_val), expected_output)


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
def test_AttributeError_error_not_arrays():
    with pytest.raises(AttributeError):
        match("not an array", [1, 2, 3])

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
        # Test case 8: Test for a non-pandas, non-dictionary input
        (
            "John",
            "John",
        ),
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
        'values': [1, 2, 3, 4]
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
        'values': [1, 2, 3, 4]
    }, index=pd.MultiIndex.from_tuples([(0, 0), (0, 1), (1, 0), (1, 1)], names=['dim1', 'dim2']))


@pytest.fixture
def sample_df_multiindex_missing():
    return pd.DataFrame({
        'values': [1, 2, np.nan, 4, 5]
    }, index=pd.MultiIndex.from_tuples([(0, 0), (0, 1), (1, 0), (1, 1), (2, 0)], names=['dim1', 'dim2']))


def test_dataframe_to_array_basic(sample_df):
    expected = np.array([[1, 2], [3, 4]])
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
    result = dataframe_to_array(sample_df_missing, 'values', ['dim1', 'dim2'], fill_val=0)
    assert np.array_equal(result, expected)


def test_dataframe_to_array_idx_col_not_in_df(sample_df):
    with pytest.raises(AssertionError):
        dataframe_to_array(sample_df, 'values', ['dim1', 'dim3'])


def test_dataframe_to_array_non_integer_dim(sample_df):
    sample_df['dim1'] = sample_df['dim1'].astype(float)
    with pytest.raises(AssertionError):
        dataframe_to_array(sample_df, 'values', ['dim1', 'dim2'])

