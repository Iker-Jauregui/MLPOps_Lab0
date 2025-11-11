"""
Command Line Interface for data preprocessing utilities.

Examples:
# Clean operations
>>> python -m src.cli clean remove-missing '["a", null, "", "b"]'
>>> python -m src.cli clean fill-missing '[1, null, 3]' --fill-value 999

# Numeric operations
>>> python -m src.cli numeric normalize '[1, 2, 3, 4, 5]'
>>> python -m src.cli numeric clip '[1, 5, 10, 15]' --min-value 3 --max-value 12
>>> python -m src.cli numeric standardize '[10, 20, 30, 40, 50]'

# Text operations
>>> python -m src.cli text tokenize "Hello, World! Test 123"
>>> python -m src.cli text remove-stopwords "this is a test" --stopwords '["is", "a"]'

# Struct operations
>>> python -m src.cli struct shuffle '[1, 2, 3, 4, 5]' --seed 42
>>> python -m src.cli struct flatten '[[1, 2], [3, 4], [5]]'
"""

import json
import click
from src import preprocessing


@click.group()
def cli():
    """CLI for data preprocessing utilities."""
    pass


# ============================================================================
# CLEAN GROUP - Data cleaning operations
# ============================================================================


@cli.group()
def clean():
    """Commands for data cleaning operations."""
    pass


@clean.command()
@click.argument("values", type=str)
def remove_missing(values):
    """
    Remove missing values (None, '', nan) from a list.

    Example:
        cli clean remove-missing '["a", null, "", "b", "c"]'
    """
    try:
        values_list = json.loads(values)
        result = preprocessing.remove_missing_values(values_list)
        click.echo(f"Result: {result}")
    except json.JSONDecodeError:
        click.echo("Error: VALUES must be a valid JSON array", err=True)
    except Exception as e:
        click.echo(f"Error: {str(e)}", err=True)


@clean.command()
@click.argument("values", type=str)
@click.option(
    "--fill-value", default=0, help="Value to fill missing entries (default: 0)"
)
def fill_missing(values, fill_value):
    """
    Fill missing values with a specified value.

    Example:
        cli clean fill-missing '["a", null, "", "b"]' --fill-value "MISSING"
    """
    try:
        values_list = json.loads(values)
        result = preprocessing.fill_missing_values(values_list, fill_value)
        click.echo(f"Result: {result}")
    except json.JSONDecodeError:
        click.echo("Error: VALUES must be a valid JSON array", err=True)
    except Exception as e:
        click.echo(f"Error: {str(e)}", err=True)


@clean.command()
@click.argument("values", type=str)
def remove_duplicates(values):
    """
    Remove duplicate values from a list.

    Example:
        cli clean remove-duplicates '[1, 2, 2, 3, 3, 3]'
    """
    try:
        values_list = json.loads(values)
        result = preprocessing.remove_duplicated_values(values_list)
        click.echo(f"Result: {result}")
    except json.JSONDecodeError:
        click.echo("Error: VALUES must be a valid JSON array", err=True)
    except Exception as e:
        click.echo(f"Error: {str(e)}", err=True)


# ============================================================================
# NUMERIC GROUP - Numerical data operations
# ============================================================================


@cli.group()
def numeric():
    """Commands for numerical data preprocessing."""
    pass


@numeric.command()
@click.argument("values", type=str)
@click.option(
    "--new-min", default=0.0, type=float, help="New minimum value (default: 0.0)"
)
@click.option(
    "--new-max", default=1.0, type=float, help="New maximum value (default: 1.0)"
)
def normalize(values, new_min, new_max):
    """
    Normalize numerical values using min-max scaling.

    Example:
        cli numeric normalize '[1, 2, 3, 4, 5]' --new-min 0 --new-max 10
    """
    try:
        values_list = json.loads(values)
        result = preprocessing.normalize_min_max(values_list, new_min, new_max)
        click.echo(f"Result: {result}")
    except json.JSONDecodeError:
        click.echo("Error: VALUES must be a valid JSON array", err=True)
    except Exception as e:
        click.echo(f"Error: {str(e)}", err=True)


@numeric.command()
@click.argument("values", type=str)
def standardize(values):
    """
    Standardize numerical values using z-score method.

    Example:
        cli numeric standardize '[1, 2, 3, 4, 5]'
    """
    try:
        values_list = json.loads(values)
        result = preprocessing.standardize_zscore(values_list)
        click.echo(f"Result: {result}")
    except json.JSONDecodeError:
        click.echo("Error: VALUES must be a valid JSON array", err=True)
    except Exception as e:
        click.echo(f"Error: {str(e)}", err=True)


@numeric.command()
@click.argument("values", type=str)
@click.option(
    "--min-value", default=0, type=float, help="Minimum value to clip (default: 0)"
)
@click.option(
    "--max-value", default=1, type=float, help="Maximum value to clip (default: 1)"
)
def clip(values, min_value, max_value):
    """
    Clip numerical values to a specified range.

    Example:
        cli numeric clip '[1, 5, 10, 15]' --min-value 2 --max-value 8
    """
    try:
        values_list = json.loads(values)
        result = preprocessing.clip_values(values_list, min_value, max_value)
        click.echo(f"Result: {result}")
    except json.JSONDecodeError:
        click.echo("Error: VALUES must be a valid JSON array", err=True)
    except Exception as e:
        click.echo(f"Error: {str(e)}", err=True)


@numeric.command()
@click.argument("values", type=str)
def to_integer(values):
    """
    Convert string values to integers (non-numerical values excluded).

    Example:
        cli numeric to-integer '["1", "2.5", "abc", "3"]'
    """
    try:
        values_list = json.loads(values)
        result = preprocessing.convert_to_integers(values_list)
        click.echo(f"Result: {result}")
    except json.JSONDecodeError:
        click.echo("Error: VALUES must be a valid JSON array", err=True)
    except Exception as e:
        click.echo(f"Error: {str(e)}", err=True)


@numeric.command()
@click.argument("values", type=str)
def log_transform(values):
    """
    Transform numerical values to logarithmic scale (positive values only).

    Example:
        cli numeric log-transform '[1, 10, 100, 1000]'
    """
    try:
        values_list = json.loads(values)
        result = preprocessing.log_transform(values_list)
        click.echo(f"Result: {result}")
    except json.JSONDecodeError:
        click.echo("Error: VALUES must be a valid JSON array", err=True)
    except Exception as e:
        click.echo(f"Error: {str(e)}", err=True)


# ============================================================================
# TEXT GROUP - Text processing operations
# ============================================================================


@cli.group()
def text():
    """Commands for text data preprocessing."""
    pass


@text.command()
@click.argument("input_text", type=str)
def tokenize(input_text):
    """
    Tokenize text into lowercase alphanumeric words.

    Example:
        cli text tokenize "Hello, World! This is a TEST 123."
    """
    try:
        result = preprocessing.tokenize_text(input_text)
        click.echo(f"Result: {result}")
    except Exception as e:
        click.echo(f"Error: {str(e)}", err=True)


@text.command()
@click.argument("input_text", type=str)
def remove_punctuation(input_text):
    """
    Remove punctuation, keeping only alphanumeric characters and spaces.

    Example:
        cli text remove-punctuation "Hello, World! How are you?"
    """
    try:
        result = preprocessing.keep_alphanumeric_and_spaces(input_text)
        click.echo(f"Result: {result}")
    except Exception as e:
        click.echo(f"Error: {str(e)}", err=True)


@text.command()
@click.argument("input_text", type=str)
@click.option("--stopwords", type=str, help="JSON array of stopwords to remove")
def remove_stopwords(input_text, stopwords):
    """
    Remove stop-words from text.

    Example:
        cli text remove-stopwords "this is a test" --stopwords '["is", "a"]'
    """
    try:
        if stopwords:
            stopwords_list = json.loads(stopwords)
        else:
            stopwords_list = []

        result = preprocessing.remove_stopwords(input_text, stopwords_list)
        click.echo(f"Result: {result}")
    except json.JSONDecodeError:
        click.echo("Error: STOPWORDS must be a valid JSON array", err=True)
    except Exception as e:
        click.echo(f"Error: {str(e)}", err=True)


# ============================================================================
# STRUCT GROUP - Data structure operations
# ============================================================================


@cli.group()
def struct():
    """Commands for data structure operations."""
    pass


@struct.command()
@click.argument("values", type=str)
@click.option(
    "--seed",
    default=None,
    type=int,
    help="Random seed for reproducibility (default: None)",
)
def shuffle(values, seed):
    """
    Randomly shuffle a list of values.

    Example:
        cli struct shuffle '[1, 2, 3, 4, 5]' --seed 42
    """
    try:
        values_list = json.loads(values)
        result = preprocessing.shuffle_list(values_list, seed)
        click.echo(f"Result: {result}")
    except json.JSONDecodeError:
        click.echo("Error: VALUES must be a valid JSON array", err=True)
    except Exception as e:
        click.echo(f"Error: {str(e)}", err=True)


@struct.command()
@click.argument("values", type=str)
def flatten(values):
    """
    Flatten a list of lists into a single list.

    Example:
        cli struct flatten '[[1, 2], [3, 4], [5, 6]]'
    """
    try:
        values_list = json.loads(values)
        result = preprocessing.flatten_list(values_list)
        click.echo(f"Result: {result}")
    except json.JSONDecodeError:
        click.echo("Error: VALUES must be a valid JSON array", err=True)
    except Exception as e:
        click.echo(f"Error: {str(e)}", err=True)


@struct.command()
@click.argument("values", type=str)
def unique(values):
    """
    Get unique values from a list.

    Example:
        cli struct unique '[1, 2, 2, 3, 3, 3, 4]'
    """
    try:
        values_list = json.loads(values)
        result = preprocessing.remove_duplicated_values(values_list)
        click.echo(f"Result: {result}")
    except json.JSONDecodeError:
        click.echo("Error: VALUES must be a valid JSON array", err=True)
    except Exception as e:
        click.echo(f"Error: {str(e)}", err=True)


if __name__ == "__main__":
    cli()
