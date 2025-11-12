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

# Tests for "standardize_zscore"
@pytest.mark.parametrize("values,expected", [
    ([], []), # Empty list
    ([1.0], [0.0]), # One element
    ([1, 3], [-1.2247, 1.2247]), # Two elements
    ([1, 2, 3], [-1.2247, 0.0, 1.2247]), # Multiple elements
    ([3, 2, 1], [1.2247, 0.0, -1.2247]), # Descending order
    ([1, 3, 2], [-1.2247, 1.2247, 0.0]), # Random order
    ([-3.0, -2.0, -1.0], [-1.2247, 0.0, 1.2247]), # Negative values
    [[1, 1, 1], [0.0, 0.0, 0.0]] # std == 0
])
def test_standardize_zscore(values, expected):
    result = preprocessing.standardize_zscore(values)
    assert result == pytest.approx(expected, rel=1e-5)

# Tests for "clip_values"
@pytest.mark.parametrize("values,min,max,expected", [
    ([], []), # Empty list
    ([1], 0, 10, [1]), # One element
    ([1, 2], 0, 10, [1, 2]), # Two elements
    ([1, 2, 3], 0, 10, [1, 2, 3]), # Multiple elements
    ([1, 5, 10], 2, 8, [2, 5, 8]), # Normal case
    ([1, 1, 1], 2, 10, [2, 2, 2]), # All inputs clipped (lower thresh)
    ([2, 2, 2], 0, 1, [1, 1, 1]), # All inputs clipped (upper thresh)
    ([1, 2, 3], 2, 2, [2, 2, 2]), # Same min and max clip
    ([0.75, 1.25, 2.75], 1.0, 2.0, [1.0, 1.25, 2.0]), # Floats
    ([-3, -2, -1], -2, -2, [-2, -2, -2]), # Negative values
    ([-1.0, 0, 1], 0, 0.0, [0.0, 0.0, 0.0]) # Mixed data types
])
def test_clip_values(values, min, max, expected):
    result = preprocessing.clip_values(values, min, max)
    assert result == pytest.approx(expected, rel=1e-5)

# Tests for "convert_to_integers"
@pytest.mark.parametrize("values,expected", [
    ([], []), # Empty list
    (['1'], [1]), # One element
    (['1', '1'], [1, 1]), # Two elements
    (['1', '1', '1'], [1, 1, 1]), # Multiple elements
    (['1', 'a', '1'], [1, 1]), # list with non integers
    (['a', 'a', 'a'], []), # All non integers
    (['1.0'], [1.0]), # Float
    (['-1.0'], [-1.0]), # Negative float
    ([None, np.nan, '', 1, 1.0], []) # Missing, int and float values
])
def test_convert_to_integers(values, expected):
    result = preprocessing.remove_duplicated_values(values)
    assert result == expected

# Tests for "log_transform"
@pytest.mark.parametrize("values,expected", [
    ([], []), # Empty list
    ([1], [0.0]), # One element
    ([1, 1], [0.0, 0.0]), # Two elements
    ([1, 1, 1], [0.0, 0.0, 0.0]), # Multiple elements
    (([1, 10], [0.0, 2.302585092994046])), # Normal case
    ([1, 10, 0, -5], [0.0, 2.302585092994046]), # Valid and invalid values
    ([-1, 0], []), # All values <= 0
    ([-1e-10], []), # Very small negative value
    ([1e-10], [-23.02585]), # Very small positive value
    ([None, np.nan, '', 'a'], []), # Rare values
])
def test_log_transform(values, expected):
    result = preprocessing.log_transform(values)
    assert result == pytest.approx(expected, rel=1e-5)

# Tests for "tokenize_text"
@pytest.mark.parametrize("text,expected", [
    ('', []), # Empty text
    ('hello', ['hello']), # One lowercase word
    ('HELLO', ['hello']), # One uppercase word
    ('Hello', ['hello']), # One word starting with Maj.
    ('Hello world', ['hello', 'world']), # Two words
    ('Hello world 123', ['hello', 'world', '123']), # Multiple words
    ('Hello, world! 123!', ['hello', 'world', '123']) # With punct. symbols
])
def test_tokenize_text(text, expected):
    result = preprocessing.tokenize_text(text)
    assert result == expected

# Tests for "keep_alphanumeric_and_spaces"
@pytest.mark.parametrize("text,expected", [
    ('', ''), # Empty text
    ('hello', 'hello'), # Single word
    ('hello world', 'hello world'), # Text with space
    ('Hello! world', 'Hello world'), # Text with !
    ('3, 2, 1... Hello World!!!', '3 2 1 Hello World'), # Complex text
    ('      ', '      '), # 6 spaces
    ('123456', '123456'), # Only digits
    ('._,!-*|\\/+¿?¡', ''), # Punctuation symbols

])
def test_keep_alphanumeric_and_spaces(text, expected):
    result = preprocessing.keep_alphanumeric_and_spaces(text)
    assert result == expected

# Tests for "remove_stopwords"
@pytest.mark.parametrize("text,stopwords,expected", [
    ('', [], ''), # Empty text and empty stopword list
    ('', ['a'], ''), # Empty text
    ('hello', [], 'hello'), # One word
    ('hello world', [], 'hello world'), # Two words
    ('hello world', ['world'], 'hello'), # Remove one stopword
    ('hello world', ['hello', 'world'], ''), # All words are stopwords
    ('hello world', ['world, hello'], ''), # All words are stopwords (disordered)
    ('hello fakin world', ['fakin'], 'hello world'), # Normal use case
    ('hello world', [1, 1.0, None, np.nan, ''], 'hello world'), # Trash on stopwords
    ('HELLO World', [], 'hello world') # Testing lowercase
])
def test_remove_stopwords(text, stopwords, expected):
    result = preprocessing.remove_stopwords(text, stopwords)
    assert result == expected

# Tests for "flatten_list"
@pytest.mark.parametrize("values,expected", [
    ([], []), # Empty list
    ([[], [], []], []), # List of empty lists
    ([1], [1]), # One element list
    ([[], [1]], [1]), # Empty list + not empty list
    ([1, 2], [1, 2]), # Two element list
    ([[1], [2]], [1, 2]), # Two lists with one element each
    ([[1, 2], [3, 4]], [1, 2, 3, 4]), # Normal use case
    ([[1, 2, 3, 4], [5]], [1, 2, 3, 4, 5]), # Multiple list + one element list
    ([[1], [2, 3, 4, 5]], [1, 2, 3, 4, 5]), # Prev. case but switched
    ([[1, 2.0], ['3', -4]], [1, 2.0, '3', -4]), # Mutiple data types
    ([[None], ['']], [None, '']) # Rare cases
])
def test_flatten_list(values, expected):
    result = preprocessing.flatten_list(values)
    assert result == expected

# Tests for "shuffle_list"
def test_shuffle_list_properties():
    """Test that shuffle maintains list properties."""
    original = [1, 2, 3, 4, 5]
    
    # Test 1: Same length
    result = preprocessing.shuffle_list(original)
    assert len(result) == len(original)
    
    # Test 2: Same elements (different order)
    assert sorted(result) == sorted(original)
    
    # Test 3: Original list unchanged
    preprocessing.shuffle_list(original)
    assert original == [1, 2, 3, 4, 5]


def test_shuffle_list_randomness():
    """Test that shuffle actually randomizes (probabilistic)."""
    original = list(range(1000))
    
    # Shuffle without seed
    result = preprocessing.shuffle_list(original)
    
    # Very unlikely to be the same
    assert result != original


def test_shuffle_list_seed_reproducibility():
    """Test that same seed produces same result."""
    original = [1, 2, 3, 4, 5]
    
    result1 = preprocessing.shuffle_list(original, seed=42)
    result2 = preprocessing.shuffle_list(original, seed=42)
    
    # Same seed -> same shuffle
    assert result1 == result2
    
    # Different seed -> (probably) different shuffle
    result3 = preprocessing.shuffle_list(original, seed=123)
    assert result1 != result3