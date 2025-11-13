"""Integration tests for CLI commands."""

import pytest
from click.testing import CliRunner
from src.cli import cli # pylint: disable=import-error

# Fixture
@pytest.fixture
def runner():
    """Fixture to provide a CliRunner instance for all CLI tests."""
    return CliRunner()


# Test for "remove-missing" with successful performance
@pytest.mark.parametrize("values,expected", [
    ('[]', '[]'),
    ('["a", null, "b"]', '[\'a\', \'b\']'),
    ('[1, null, 2, 3]', '[1, 2, 3]'),
    ('[1, null, 2, "", 3, "hello"]', '[1, 2, 3, \'hello\']'),
    ('[null, null, ""]', '[]')
])
def test_remove_missing_exit_OK(runner, values, expected):
    """Test remove-missing command with valid inputs."""
    result = runner.invoke(cli, ['clean', 'remove-missing', values])
    assert result.exit_code == 0
    assert expected in result.output


# Test for "remove-missing" with erratic performance
@pytest.mark.parametrize("values", [
    'not-json',
    1,
    None,
    1.0,
    '\\{1, 2, 3\\}'
])
def test_remove_missing_error(runner, values):
    """Test remove-missing command with invalid inputs."""
    result = runner.invoke(cli, ['clean', 'remove-missing', values])
    assert ("Error" in result.output) or (result.exit_code != 0)


# Test for "fill-missing" with successful performance
@pytest.mark.parametrize("values,fill_value,expected", [
    ('[1, null, 3]', '0', '[1, 0, 3]'),
    ('["a", null, "b"]', '0', '[\'a\', 0, \'b\']'),
    ('[null, null, null]', '0', '[0, 0, 0]'),
    ('[]', '0', '[]'),
    ('[1, "", 3]', '-1', '[1, -1, 3]')
])
def test_fill_missing_exit_OK(runner, values, fill_value, expected):
    """Test fill-missing command with valid inputs."""
    result = runner.invoke(cli, ['clean', 'fill-missing', values, '--fill-value', fill_value])
    assert result.exit_code == 0
    assert expected in result.output


# Test for "fill-missing" with erratic performance
@pytest.mark.parametrize("values,fill_value", [
    ('not-json', '0'),
    ('\\{1, 2\\}', '0'),
    (1, '0'),
    (None, '0'),
    (1.0, '0')
])
def test_fill_missing_error(runner, values, fill_value):
    """Test fill-missing command with invalid inputs."""
    result = runner.invoke(cli, ['clean', 'fill-missing', values, '--fill-value', fill_value])
    assert ("Error" in result.output) or (result.exit_code != 0)


# Test for "remove-duplicates" with successful performance
@pytest.mark.parametrize("values,expected", [
    ('[1, 2, 2, 3, 3, 3]', '[1, 2, 3]'),
    ('["a", "a", "b"]', '[\'a\', \'b\']'),
    ('[]', '[]'),
    ('[1]', '[1]'),
    ('[1, 1, 1, 1]', '[1]')
])
def test_remove_duplicates_exit_OK(runner, values, expected):
    """Test remove-duplicates command with valid inputs."""
    result = runner.invoke(cli, ['clean', 'remove-duplicates', values])
    assert result.exit_code == 0
    assert expected in result.output


# Test for "remove-duplicates" with erratic performance
@pytest.mark.parametrize("values", [
    'not-json',
    '\\{1, 2, 3\\}',
    1,
    None,
    1.0
])
def test_remove_duplicates_error(runner, values):
    """Test remove-duplicates command with invalid inputs."""
    result = runner.invoke(cli, ['clean', 'remove-duplicates', values])
    assert ("Error" in result.output) or (result.exit_code != 0)


# Test for "normalize" with successful performance
@pytest.mark.parametrize("values,new_min,new_max,expected", [
    ('[1, 2, 3, 4, 5]', '0', '1', '0.0'),
    ('[1, 2, 3, 4, 5]', '0', '10', '2.5'),
    ('[0, 10]', '-1', '1', '-1.0'),
    ('[]', '0', '1', '[]'),
    ('[5]', '0', '1', '[')
])
def test_normalize_exit_OK(runner, values, new_min, new_max, expected):
    """Test normalize command with valid inputs."""
    result = runner.invoke(
        cli,
        ['numeric', 'normalize', values, '--new-min', new_min, '--new-max', new_max]
    )
    assert result.exit_code == 0
    assert expected in result.output


# Test for "normalize" with erratic performance
@pytest.mark.parametrize("values,new_min,new_max", [
    ('not-json', '0', '1'),
    ('\\{1, 2\\}', '0', '1'),
    (1, '0', '1'),
    (None, '0', '1'),
    (1.0, '0', '1')
])
def test_normalize_error(runner, values, new_min, new_max):
    """Test normalize command with invalid inputs."""
    result = runner.invoke(
        cli,
        ['numeric', 'normalize', values, '--new-min', new_min, '--new-max', new_max]
    )
    assert ("Error" in result.output) or (result.exit_code != 0)


# Test for "standardize" with successful performance
@pytest.mark.parametrize("values,expected", [
    ('[1, 3]', '-1.0'),
    ('[]', '[]'),
    ('[5]', '[0.0]'),
    ('[1, 1, 1]', '[0.0, 0.0, 0.0]')
])
def test_standardize_exit_OK(runner, values, expected):
    """Test standardize command with valid inputs."""
    result = runner.invoke(cli, ['numeric', 'standardize', values])
    assert result.exit_code == 0
    assert expected in result.output


# Test for "standardize" with erratic performance
@pytest.mark.parametrize("values", [
    'not-json',
    '\\{1, 2, 3\\}',
    1,
    None,
    1.0
])
def test_standardize_error(runner, values):
    """Test standardize command with invalid inputs."""
    result = runner.invoke(cli, ['numeric', 'standardize', values])
    assert ("Error" in result.output) or (result.exit_code != 0)


# Test for "clip" with successful performance
@pytest.mark.parametrize("values,min_value,max_value,expected", [
    ('[1, 5, 10, 15]', '2', '8', '2.0'),
    ('[1, 2, 3]', '0', '10', '1.0'),
    ('[]', '0', '1', '[]'),
    ('[5]', '2', '8', '5.0'),
    ('[1, 1, 1]', '2', '10', '2.0')
])
def test_clip_exit_OK(runner, values, min_value, max_value, expected):
    """Test clip command with valid inputs."""
    result = runner.invoke(
        cli,
        ['numeric', 'clip', values, '--min-value', min_value, '--max-value', max_value]
    )
    assert result.exit_code == 0
    assert expected in result.output


# Test for "clip" with erratic performance
@pytest.mark.parametrize("values,min_value,max_value", [
    ('not-json', '0', '1'),
    ('\\{1, 2\\}', '0', '1'),
    (1, '0', '1'),
    (None, '0', '1'),
    (1.0, '0', '1')
])
def test_clip_error(runner, values, min_value, max_value):
    """Test clip command with invalid inputs."""
    result = runner.invoke(
        cli,
        ['numeric', 'clip', values, '--min-value', min_value, '--max-value', max_value]
    )
    assert ("Error" in result.output) or (result.exit_code != 0)


# Test for "to-integer" with successful performance
@pytest.mark.parametrize("values,expected", [
    ('["1", "2", "3"]', '[1, 2, 3]'),
    ('[]', '[]'),
    ('["abc", "def"]', '[]'),
    ('["1"]', '[1]'),
    ('["1", "abc"]', '[1]')
])
def test_to_integer_exit_OK(runner, values, expected):
    """Test to-integer command with valid inputs."""
    result = runner.invoke(cli, ['numeric', 'to-integer', values])
    assert result.exit_code == 0
    assert expected in result.output


# Test for "to-integer" with erratic performance
@pytest.mark.parametrize("values", [
    'not-json',
    '\\{1, 2, 3\\}',
    1,
    None,
    1.0
])
def test_to_integer_error(runner, values):
    """Test to-integer command with invalid inputs."""
    result = runner.invoke(cli, ['numeric', 'to-integer', values])
    assert ("Error" in result.output) or (result.exit_code != 0)


# Test for "log-transform" with successful performance
@pytest.mark.parametrize("values,expected", [
    ('[1, 10]', '0.0'),
    ('[]', '[]'),
    ('[-1, 0]', '[]'),
    ('[1]', '[0.0]')
])
def test_log_transform_exit_OK(runner, values, expected):
    """Test log-transform command with valid inputs."""
    result = runner.invoke(cli, ['numeric', 'log-transform', values])
    assert result.exit_code == 0
    assert expected in result.output


# Test for "log-transform" with erratic performance
@pytest.mark.parametrize("values", [
    'not-json',
    '\\{1, 2, 3\\}',
    1,
    None,
    1.0
])
def test_log_transform_error(runner, values):
    """Test log-transform command with invalid inputs."""
    result = runner.invoke(cli, ['numeric', 'log-transform', values])
    assert ("Error" in result.output) or (result.exit_code != 0)


# Test for "tokenize" with successful performance
@pytest.mark.parametrize("input_text,expected", [
    ('Hello, World!', '[\'hello\', \'world\']'),
    ('', '[]'),
    ('HELLO', '[\'hello\']'),
    ('Hello, World! Test 123.', '[\'hello\', \'world\', \'test\', \'123\']')
])
def test_tokenize_exit_OK(runner, input_text, expected):
    """Test tokenize command with valid inputs."""
    result = runner.invoke(cli, ['text', 'tokenize', input_text])
    assert result.exit_code == 0
    assert expected in result.output


# Test for "tokenize" with erratic performance
@pytest.mark.parametrize("input_text", [
    1,
    None,
    1.0
])
def test_tokenize_error(runner, input_text):
    """Test tokenize command with invalid inputs."""
    result = runner.invoke(cli, ['text', 'tokenize', input_text])
    assert ("Error" in result.output) or (result.exit_code != 0)


# Test for "remove-punctuation" with successful performance
@pytest.mark.parametrize("input_text,expected", [
    ('Hello, World!', 'Hello World'),
    ('Test!!! 123...', 'Test 123'),
    ('', ''),
    ('abc', 'abc'),
    ('!!!', '')
])
def test_remove_punctuation_exit_OK(runner, input_text, expected):
    """Test remove-punctuation command with valid inputs."""
    result = runner.invoke(cli, ['text', 'remove-punctuation', input_text])
    assert result.exit_code == 0
    assert expected in result.output


# Test for "remove-punctuation" with erratic performance
@pytest.mark.parametrize("input_text", [
    1,
    None,
    1.0
])
def test_remove_punctuation_error(runner, input_text):
    """Test remove-punctuation command with invalid inputs."""
    result = runner.invoke(cli, ['text', 'remove-punctuation', input_text])
    assert ("Error" in result.output) or (result.exit_code != 0)


# Test for "remove-stopwords" with successful performance
@pytest.mark.parametrize("input_text,stopwords,expected", [
    ('this is a test', '["is", "a"]', 'this test'),
    ('hello world', '["world"]', 'hello'),
    ('hello world', '[]', 'hello world'),
    ('', '["a"]', ''),
    ('test', '["not", "present"]', 'test')
])
def test_remove_stopwords_exit_OK(runner, input_text, stopwords, expected):
    """Test remove-stopwords command with valid inputs."""
    result = runner.invoke(
        cli,
        ['text', 'remove-stopwords', input_text, '--stopwords', stopwords]
    )
    assert result.exit_code == 0
    assert expected in result.output


# Test for "remove-stopwords" with erratic performance
@pytest.mark.parametrize("input_text,stopwords", [
    ('hello world', 'not-json'),
    ('hello world', '\\{a, b\\}'),
    (1, '["a"]'),
    (1.0, '["a"]')
])
def test_remove_stopwords_error(runner, input_text, stopwords):
    """Test remove-stopwords command with invalid inputs."""
    result = runner.invoke(
        cli,
        ['text', 'remove-stopwords', input_text, '--stopwords', stopwords]
    )
    assert ("Error" in result.output) or (result.exit_code != 0)


# Test for "shuffle" with successful performance
@pytest.mark.parametrize("values,seed,check_value", [
    ('[1, 2, 3, 4, 5]', '42', '['),
    ('[1, 2, 3]', '123', '['),
    ('[]', '42', '[]'),
    ('[1]', '42', '[1]')
])
def test_shuffle_exit_OK(runner, values, seed, check_value):
    """Test shuffle command with valid inputs."""
    result = runner.invoke(cli, ['struct', 'shuffle', values, '--seed', seed])
    assert result.exit_code == 0
    assert check_value in result.output


# Test for "shuffle" with erratic performance
@pytest.mark.parametrize("values,seed", [
    ('not-json', '42'),
    ('\\{1, 2, 3\\}', '42'),
    (1, '42'),
    (None, '42')
])
def test_shuffle_error(runner, values, seed):
    """Test shuffle command with invalid inputs."""
    result = runner.invoke(cli, ['struct', 'shuffle', values, '--seed', seed])
    assert ("Error" in result.output) or (result.exit_code != 0)


# Test for "flatten" with successful performance
@pytest.mark.parametrize("values,expected", [
    ('[[1, 2], [3, 4], [5]]', '[1, 2, 3, 4, 5]'),
    ('[[1], [2], [3]]', '[1, 2, 3]'),
    ('[]', '[]'),
    ('[[]]', '[]'),
    ('[[1, 2, 3]]', '[1, 2, 3]')
])
def test_flatten_exit_OK(runner, values, expected):
    """Test flatten command with valid inputs."""
    result = runner.invoke(cli, ['struct', 'flatten', values])
    assert result.exit_code == 0
    assert expected in result.output


# Test for "flatten" with erratic performance
@pytest.mark.parametrize("values", [
    'not-json',
    '\\{1, 2, 3\\}',
    1,
    None,
    1.0
])
def test_flatten_error(runner, values):
    """Test flatten command with invalid inputs."""
    result = runner.invoke(cli, ['struct', 'flatten', values])
    assert ("Error" in result.output) or (result.exit_code != 0)


# Test for "unique" with successful performance
@pytest.mark.parametrize("values,expected", [
    ('[1, 2, 2, 3, 3, 3, 4]', '[1, 2, 3, 4]'),
    ('["a", "a", "b"]', '[\'a\', \'b\']'),
    ('[]', '[]'),
    ('[1]', '[1]'),
    ('[1, 1, 1]', '[1]')
])
def test_unique_exit_OK(runner, values, expected):
    """Test unique command with valid inputs."""
    result = runner.invoke(cli, ['struct', 'unique', values])
    assert result.exit_code == 0
    assert expected in result.output


# Test for "unique" with erratic performance
@pytest.mark.parametrize("values", [
    'not-json',
    '\\{1, 2, 3\\}',
    1,
    None,
    1.0
])
def test_unique_error(runner, values):
    """Test unique command with invalid inputs."""
    result = runner.invoke(cli, ['struct', 'unique', values])
    assert ("Error" in result.output) or (result.exit_code != 0)