# chatgpt generated unittest for DataPrep
# - REQUIRES REVIEW!

# TODO: make some trivial data (not random) and validate binning value are as expected

import pytest
import pandas as pd
import numpy as np
import xarray as xr

from GPSat.dataprepper import DataPrep

@pytest.fixture
def sample_data():
    """Generates a simple dataset for testing purposes."""
    np.random.seed(42)  # For reproducibility
    data = {
        "category": np.random.choice(['A', 'B'], size=1000),
        "x": np.random.uniform(-100, 100, size=1000),
        "y": np.random.uniform(-100, 100, size=1000),
        "value": np.random.randn(1000),
    }
    return pd.DataFrame(data)


# ---
# bin_data
# ---


def test_bin_data_2d_mean(sample_data):
    """Test binning 2D data with 'mean' statistic."""
    binned_data, (x_bin, y_bin) = DataPrep.bin_data(df=sample_data, x_range=(-100, 100), y_range=(-100, 100),
                                                    grid_res=20, x_col="x", y_col="y", val_col="value",
                                                    bin_statistic="mean", bin_2d=True,
                                                    return_bin_center=True)
    assert binned_data.shape == (10, 10), "The shape of the binned data does not match the expected output."
    assert len(x_bin) == 10 and len(y_bin) == 10, "The length of bin centers does not match the expected output."

    # if not returning bin centers, then x/y_bin will be the edges, and thus longer by 1
    binned_data, (x_bin, y_bin) = DataPrep.bin_data(df=sample_data, x_range=(-100, 100), y_range=(-100, 100),
                                                    grid_res=20, x_col="x", y_col="y", val_col="value",
                                                    bin_statistic="mean", bin_2d=True,
                                                    return_bin_center=False)

    assert binned_data.shape == (10, 10), "The shape of the binned data does not match the expected output."
    assert len(x_bin) == 11 and len(y_bin) == 11, "The length of bin centers does not match the expected output."



def test_bin_data_2d_count(sample_data):
    """Test binning 2D data with 'count' statistic."""
    binned_data, (x_bin, y_bin) = DataPrep.bin_data(df=sample_data, x_range=(-100, 100), y_range=(-100, 100),
                                                    grid_res=20, x_col="x", y_col="y", val_col="value",
                                                    bin_statistic="count", bin_2d=True)
    assert np.all(binned_data >= 0), "Binned data contains negative counts."


def test_bin_data_1d_mean(sample_data):
    """Test binning 1D data with 'mean' statistic."""
    binned_data, x_bin = DataPrep.bin_data(df=sample_data, x_range=(-100, 100), grid_res=20, x_col="x",
                                           val_col="value", bin_statistic="mean", bin_2d=False)
    assert binned_data.shape == (10,), "The shape of the binned data does not match the expected output."
    assert len(x_bin) == 10, "The length of bin centers does not match the expected output."


def test_bin_data_invalid_val_col(sample_data):
    """Test binning with an invalid val_col."""
    with pytest.raises(AssertionError):
        DataPrep.bin_data(df=sample_data, x_range=(-100, 100), y_range=(-100, 100), grid_res=20, x_col="x", y_col="y",
                          val_col="nonexistent_column", bin_statistic="mean", bin_2d=True)


def test_bin_data_invalid_grid_res(sample_data):
    """Test binning with an invalid grid resolution."""
    with pytest.raises(AssertionError):
        DataPrep.bin_data(df=sample_data, x_range=(-100, 100), y_range=(-100, 100), grid_res=None, x_col="x", y_col="y",
                          val_col="value", bin_statistic="mean", bin_2d=True)


def test_bin_data_invalid_x_range(sample_data):
    """Test binning with an invalid x_range."""
    with pytest.raises(AssertionError):
        DataPrep.bin_data(df=sample_data, x_range=(100, -100), y_range=(-100, 100), grid_res=20, x_col="x", y_col="y",
                          val_col="value", bin_statistic="mean", bin_2d=True)


@pytest.mark.parametrize("statistic", ["mean", "median", "count", "sum"])
def test_bin_data_various_statistics(sample_data, statistic):
    """Test binning with various statistics."""
    binned_data, _ = DataPrep.bin_data(df=sample_data, x_range=(-100, 100), y_range=(-100, 100),
                                       grid_res=20, x_col="x", y_col="y", val_col="value",
                                       bin_statistic=statistic, bin_2d=True)
    assert binned_data.size > 0, f"Binned data is empty for statistic {statistic}."

# ---
# bin_data_by
# ---

def test_bin_data_by_2d_mean(sample_data):
    """Test 2D binning by category with 'mean' statistic."""

    # return Dataframe
    result = DataPrep.bin_data_by(df=sample_data, by_cols='category',
                                  x_col='x', y_col='y',
                                  val_col='value',
                                  x_range=(-100, 100), y_range=(-100, 100),
                                  grid_res=20, bin_statistic='mean', bin_2d=True,
                                  return_df=True)
    assert isinstance(result, pd.DataFrame), "The result should be a DataFrame"
    assert 'value' in result, "Resulting dataset should contain 'value_mean'."


# def test_bin_data_by_1d_sum(sample_data):
#     """Test 1D binning by category with 'sum' statistic."""
#     result = DataPrep.bin_data_by(df=sample_data, by_cols='category', x_col='x', val_col='value',
#                                   grid_res=20, bin_statistic='sum', bin_2d=False)
#     assert isinstance(result, pd.DataFrame) or isinstance(result, xr.Dataset), "The result should be a DataFrame or Dataset."
#     assert 'value_sum' in result.columns or 'value_sum' in result.data_vars, "Resulting dataset should contain 'value_sum'."


def test_bin_data_by_invalid_by_cols(sample_data):
    """Test binning with an invalid 'by_cols'."""
    with pytest.raises(AssertionError):
        DataPrep.bin_data_by(df=sample_data, by_cols='nonexistent_column', x_col='x', y_col='y', val_col='value',
                             grid_res=20, bin_statistic='mean', bin_2d=True)


def test_bin_data_by_invalid_val_col(sample_data):
    """Test binning with an invalid 'val_col'."""
    with pytest.raises(AssertionError):
        DataPrep.bin_data_by(df=sample_data, by_cols='category', x_col='x', y_col='y', val_col='nonexistent_value',
                             grid_res=20, bin_statistic='mean', bin_2d=True)


def test_bin_data_by_no_binning_column(sample_data):
    """Test binning without specifying 'by_cols', should raise an error."""
    with pytest.raises(AssertionError):
        DataPrep.bin_data_by(df=sample_data, x_col='x', y_col='y', val_col='value',
                             grid_res=20, bin_statistic='mean', bin_2d=True)


# @pytest.mark.parametrize("bin_statistic", ["mean", "median", "count", "sum"])
# def test_bin_data_by_various_statistics(sample_data, bin_statistic):
#     """Test binning with various statistics."""
#     result = DataPrep.bin_data_by(df=sample_data, by_cols='category', x_col='x', y_col='y', val_col='value',
#                                   grid_res=20, bin_statistic=bin_statistic, bin_2d=True)
#     assert isinstance(result, pd.DataFrame) or isinstance(result, xr.Dataset), "The result should be a DataFrame or Dataset."
#     expected_col = f'value_{bin_statistic}'
#     assert expected_col in result.columns or expected_col in result.data_vars, f"Resulting dataset should contain '{expected_col}'."
#

# def test_bin_data_by_return_df_flag(sample_data):
#     """Test the 'return_df' flag."""
#     result = DataPrep.bin_data_by(df=sample_data, by_cols='category', x_col='x', y_col='y', val_col='value',
#                                   grid_res=20, bin_statistic='mean', bin_2d=True, return_df=True)
#     assert isinstance(result, pd.DataFrame), "The result should be a DataFrame when 'return_df' is True."



# def test_bin_data_by_verbose_output(sample_data, capsys):
#     """Test the verbosity of the output."""
#     DataPrep.bin_data_by(df=sample_data, by_cols='category', x_col='x', y_col='y', val_col='value',
#                          grid_res=20, bin_statistic='mean', bin_2d=True, verbose=True)
#     captured = capsys.readouterr()
#     assert "adding new_col" in captured.out or "number unique values of by_cols" in captured.out, "Expected verbose output not captured."
