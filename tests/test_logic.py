"""Unit tests for preprocessing module."""

import pytest
import numpy as np
from src import preprocessing


# FIXTURES

@pytest.fixture
def corrupted_string_list():
    """
    Fixture providing a sample list of numbers containing missing values
    """
    return [1, None, 2, np.nan, '', 3]

# UNIT TESTS

# Tests for "remove_missing_values"
def test_remove_missing_values_with_fixture(corrupted_string_list):
    result = preprocessing.remove_missing_values(corrupted_string_list)
    expected = [1, 2, 3]
    assert result == expected

@pytest.mark.parametrize("values,expected", [
    ([], []), # Empty List
    ([None, 1.0, np.nan], [1.0]), # Limit cases: None at start, np.nan at end
    ([np.nan, 1.0, ''], [1.0]), # Limit cases: np.nan at start, '' at end
    (['', 1.0, None], [1.0]), # Limit cases: '' at start, None at end
    (['', np.nan, None], []), # All input values are missing values. Result -> Empty list
    (['1.0', 'hey'], ['1.0', 'hey']), # Not designed for strings but should work
    ([1.0, '1.0'], [1.0, '1.0']) # Mixing data types
])
def test_remove_missing_values_parametrized(values, expected):
    result = preprocessing.remove_missing_values(values)
    assert result == expected

# Tests for "fill_missing_values"
def text_fill_missing_values_with_fixture(corrupted_string_list):
    result = preprocessing.fill_missing_values(corrupted_string_list)
    expected = [1, 0, 2, 0, 3]
    assert result == expected

@pytest.mark.parametrize("values,fill_value,expected", [
    ([], 0, []), # Empty list
    ([None, 1.0, np.nan], 0.0, [0.0, 1.0, 0.0]), # Limit cases: None at start, np.nan at end
    ([np.nan, 1.0, ''], 0.0, [0.0, 1.0, 0.0]), # Limit cases: np.nan at start, '' at end
    (['', 1.0, None], 0.0, [0.0, 1.0, 0.0]), # Limit cases: '' at start, None at end
    (['', np.nan, None], 0.0, [0.0, 0.0, 0.0]), # All input values are missing values.
    (['1.0', 'hey', ''], 'filled', ['1.0', 'hey', 'filled']), # Not designed for strings but should work
    ([1.0, '1.0', ''], 'filled', [1.0, '1.0', 'filled']), # Mixing data types, fill with str
    ([1.0, '1.0', ''], 1.0, [1.0, '1.0', 1.0]), # Mixig data types, fill with float (1.0)
    ([1.0, '1.0', ''], 0, [1.0, '1.0', 0]) # Mixing data types, fill with int (0)
])
def text_fill_missing_values_parametrized(values, fill_value, expected):
    result = preprocessing.fill_missing_values(values, fill_value)
    assert result == expected

# Test for "remove_duplicated_values"
@pytest.mark.parametrize("values,expected", [
    ([], []), # Empty list
    ([1], [1]), # One element list
    ([1, 1], [1]), # Two element list with duplicates
    ([1, 1, 2], [1, 2]), # Duplicates at start
    ([1, 2, 2], [1, 2]), # Duplicates at end
    ([1, 2, 1], [1, 2]), # Disordered duplicates
    ([1.0, 1.0, 2.0], [1.0, 2.0]), # With float type
    (['1', '1', '2'], ['1', '2']), # With string
    ([1, 1, 2.0, 2.0, '1.0', '1.0'], [1, 2.0, '1.0']), # Mixing data types
    ([1, 1, 1.0, 1.0, '1.0', '1.0'], [1, '1.0']), # Mixing data types
])
def test_remove_duplicated_values_parametrized(values, expected):
    result = preprocessing.remove_duplicated_values(values)
    assert result == expected

# Tests for "normalize_min_max"
@pytest.mark.parametrize("values,min,max,expected", [
    ([], 0.0, 1.0, []), # Empty list
    ([1], 0.0, 1.0, [0.0]), # One element
    ([0.0, 1.0], 0.0, 1.0, [0.0, 1.0]), # Two elements
    ([1, 2, 3, 4, 5], 0.0, 1.0, [0.0, 0.25, 0.5, 0.75, 1.0]), # Normal case
    ([5, 4, 3, 2, 1], 0.0, 1.0, [1.0, 0.75, 0.5, 0.25, 0.0]), # Descending order
    ([1, 5, 4, 2, 3], 0.0, 1.0, [0.0, 1.0, 0.75, 0.25, 0.5]), # Random order
    ([0.0, 1.0], -1.0, 1.0, [-1.0, 1.0]), # Negative min + Positive max
    ([0.0, 1.0], -5.0, -1.0, [-5.0, -1.0]), # Negative min and max
    ([-5.0, 0.0, 5.0], 0.0, 1.0, [0.0, 0.5, 1.0]), # Negative and positive inputs
    ([-15.0, -10.0, -5.0], 0.0, 1.0, [0.0, 0.5, 1.0]) # All negative inputs
])
def test_normalize_min_max_parametrized(values, min, max, expected):
    result = preprocessing.normalize_min_max(values, min, max)
    assert result == pytest.approx(expected, rel=1e-5)

