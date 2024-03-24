# DataLoader unit tests
# TODO: DataLoader unit test require review

import pandas as pd
import pytest

from GPSat.dataloader import DataLoader

# Define a fixture for a sample DataFrame
@pytest.fixture
def sample_df():
    return pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})

# trivial functions
add_one = lambda x: x + 1
times_two = lambda x: x * 2
add_one_times_two = lambda x: (x + 1, x*2)


# Sample data and functions for testing
def sample_function(a, b):
    return a + b

@pytest.fixture
def sample_ref_loc():
    return pd.DataFrame({'loc_col': [1, 2, 3], 'other_col': [4, 5, 6]})

@pytest.fixture
def global_static_select():
    return [{'col': 'test_col', 'comp': '==', 'val': 5}]

@pytest.fixture
def global_dynamic_select():
    return [{'loc_col': 'loc_col', 'src_col': 'test_col', 'func': 'sample_function'}]

@pytest.fixture
def local_select():
    return [{'col': 'loc_col', 'comp': '==', 'val': 1}]



def test_add_single_column(sample_df):

    DataLoader.add_cols(sample_df,
                        col_func_dict={
                            'C': {'func': add_one, "col_args": "A"}
                        }
                        )

    assert 'C' in sample_df.columns, "Column 'C' was not added to the DataFrame"
    assert (sample_df['C'] == (sample_df['A']+1)).all(), "The values in column 'C' are incorrect"


    # same function from string

    DataLoader.add_cols(sample_df,
                        col_func_dict={
                            'D': {'func': "lambda x: x + 1", "col_args": "A"}
                        }
                        )

    assert 'D' in sample_df.columns, "Column 'D' was not added to the DataFrame"
    assert (sample_df['D'] == sample_df['C']).all(), "The values in column 'C' are incorrect"


def test_add_multiple_columns(sample_df):

    DataLoader.add_cols(sample_df,
                        col_func_dict={
                            'C': {'func': add_one, "col_args": "A"},
                            'D': {'func': times_two, "col_args": "B"}
                        }
                        )

    assert 'C' in sample_df.columns, "Column 'C' was not added to the DataFrame"
    assert (sample_df['C'] == (sample_df['A']+1)).all(), "The values in column 'C' are incorrect"

    assert 'D' in sample_df.columns, "Column 'D' was not added to the DataFrame"
    assert (sample_df['D'] == (sample_df['B']*2)).all(), "The values in column 'C' are incorrect"



def test_add_multiple_columns_from_multi_output_function(sample_df):


    DataLoader.add_cols(sample_df,
                        col_func_dict={
                            ('C', 'D'): {'func': add_one_times_two, "col_args": "A"}
                        }
                        )

    assert 'C' in sample_df.columns, "Column 'C' was not added to the DataFrame"
    assert (sample_df['C'] == (sample_df['A']+1)).all(), "The values in column 'C' are incorrect"

    assert 'D' in sample_df.columns, "Column 'D' was not added to the DataFrame"
    assert (sample_df['D'] == (sample_df['A']*2)).all(), "The values in column 'C' are incorrect"


# def test_assertion_error_for_mismatched_lengths(sample_df):
#     with pytest.raises(AssertionError):
#         DataLoader.add_cols(sample_df, col_func_dict={('C', 'D'): {'func': lambda df, filename=None: (df['A'],)}})

def test_no_operation_for_none(sample_df):
    original_columns = sample_df.columns.copy()
    DataLoader.add_cols(sample_df)
    DataLoader.add_cols(sample_df, col_func_dict=None)
    assert all(sample_df.columns == original_columns), "The DataFrame should not change when `col_func_dict` is None or not provided"



def test_is_list_of_dict_with_list_of_dicts():
    input_list = [{"a": 1}, {"b": 2}]
    assert DataLoader.is_list_of_dict(input_list) == True, "Failed to identify a list of dictionaries"

def test_is_list_of_dict_with_empty_list():
    input_list = []
    assert DataLoader.is_list_of_dict(input_list) == True, "Failed to identify an empty list as a list of dictionaries"

def test_is_list_of_dict_with_list_of_ints():
    input_list = [1, 2, 3]
    assert DataLoader.is_list_of_dict(input_list) == False, "Incorrectly identified a list of ints as a list of dictionaries"

def test_is_list_of_dict_with_list_of_mixed_types():
    input_list = [{"a": 1}, 2, "string"]
    assert DataLoader.is_list_of_dict(input_list) == False, "Incorrectly identified a list of mixed types as a list of dictionaries"

def test_is_list_of_dict_with_non_list_input():
    inputs = ["not a list", 123, {"a": 1}, 12.34, None]
    for input_val in inputs:
        assert DataLoader.is_list_of_dict(input_val) == False, f"Incorrectly identified {type(input_val)} as a list of dictionaries"

def test_is_list_of_dict_with_list_containing_lists():
    input_list = [[{"a": 1}], [{"b": 2}]]
    assert DataLoader.is_list_of_dict(input_list) == False, "Incorrectly identified a list containing lists as a list of dictionaries"

def test_is_list_of_dict_with_list_containing_empty_dicts():
    input_list = [{}, {}]
    assert DataLoader.is_list_of_dict(input_list) == True, "Failed to identify a list of empty dictionaries as a list of dictionaries"



# def test_static_condition(sample_ref_loc, global_static_select):
#     result = DataLoader.get_where_list(global_select=global_static_select, ref_loc=sample_ref_loc)
#     assert isinstance(result, list), "Result should be a list"
#     assert result == global_static_select, "Static select condition was not processed correctly"

# def test_dynamic_condition(sample_ref_loc, global_dynamic_select, local_select):
#     expected_result = [{'col': 'test_col', 'comp': '==', 'val': sample_function(1, 1)}]
#     result = DataLoader.get_where_list(global_select=global_dynamic_select, local_select=local_select, ref_loc=sample_ref_loc)
#     assert isinstance(result, list), "Result should be a list"
#     assert result == expected_result, "Dynamic select condition was not processed correctly"

def test_missing_local_select_for_dynamic(sample_ref_loc, global_dynamic_select):
    with pytest.raises(AssertionError):
        DataLoader.get_where_list(global_select=global_dynamic_select, ref_loc=sample_ref_loc)

def test_missing_ref_loc_for_dynamic(global_dynamic_select, local_select):
    with pytest.raises(AssertionError):
        DataLoader.get_where_list(global_select=global_dynamic_select, local_select=local_select)

def test_incorrect_keys_in_dynamic_select(sample_ref_loc, local_select):
    incorrect_global_select = [{'loc_col': 'loc_col', 'wrong_key': 'test_col', 'func': 'sample_function'}]
    with pytest.raises(AssertionError):
        DataLoader.get_where_list(global_select=incorrect_global_select, local_select=local_select, ref_loc=sample_ref_loc)

def test_loc_col_not_in_ref_loc(global_dynamic_select, local_select):
    incorrect_ref_loc = pd.DataFrame({'wrong_loc_col': [1, 2, 3], 'other_col': [4, 5, 6]})
    with pytest.raises(AssertionError):
        DataLoader.get_where_list(global_select=global_dynamic_select, local_select=local_select, ref_loc=incorrect_ref_loc)
