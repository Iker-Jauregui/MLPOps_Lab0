import math
import re
import random

import numpy as np

def remove_missing_values(list_of_values):
    """
    Remove missing values (None, NaN, and empty strings) from a list.

    Parameters
    ----------
    list_of_values : list
        List containing elements that may include `None`, `numpy.nan`, or empty strings `''`.

    Returns
    -------
    list
        A new list with all missing values removed.

    Examples
    --------
    >>> remove_missing_values([1, None, 2, np.nan, '', 3])
    [1, 2, 3]
    """

    return list(filter(lambda x: (x is not None) and (x != '') and not (isinstance(x, float) and math.isnan(x)), list_of_values))

def fill_missing_values(list_of_values, fill_value=0):
    """
    Fill missing values (None, NaN, and empty strings) in a list.

    Parameters
    ----------
    list_of_values : list
        List containing elements that may include `None`, `numpy.nan`, or empty strings `''`.
    fill_value : any, optional
        Value used to replace missing values. Default is 0.

    Returns
    -------
    list
        A new list with all missing values replaced by `fill_value`.

    Examples
    --------
    >>> fill_missing_values([1, None, 2, np.nan, '', 3], 100)
    [1, 100, 2, 100, 100, 3]
    """
    
    return [
        fill_value if (x is None or x == '' or (isinstance(x, float) and math.isnan(x))) else x
        for x in list_of_values
    ]

def remove_duplicated_values(list_of_values):
    """
    Remove duplicated values from a list.

    Parameters
    ----------
    list_of_values : list
        List that may contain duplicated values.

    Returns
    -------
    list
        A new list with all duplicated values removed.

    Examples
    --------
    >>> remove_duplicated_values([1, 1, 2, 2, 3, 3])
    [1, 2, 3]
    """
    
    return list(dict.fromkeys(list_of_values))

def normalize_min_max(values, new_min=0.0, new_max=1.0):
    """
    Normalize numerical values using the min-max method.

    Parameters
    ----------
    values : list
        List of numerical values.
    new_min : float, optional
        Desired minimum of the normalized values. Default is 0.0.
    new_max : float, optional
        Desired maximum of the normalized values. Default is 1.0.

    Returns
    -------
    list
        List of normalized values.

    Examples
    --------
    >>> normalize_min_max([1, 2, 3])
    [0.0, 0.5, 1.0]
    """
    arr = np.array(values, dtype=float)
    min_val, max_val = np.min(arr), np.max(arr)
    if min_val == max_val:
        return [new_min] * len(arr)
    return list(((arr - min_val) / (max_val - min_val)) * (new_max - new_min) + new_min)

def standardize_zscore(values):
    """
    Standardize numerical values using the z-score method.

    Parameters
    ----------
    values : list
        List of numerical values.

    Returns
    -------
    list
        List of standardized values.

    Examples
    --------
    >>> standardize_zscore([1, 2, 3])
    [-1.2247, 0.0, 1.2247]
    """
    arr = np.array(values, dtype=float)
    mean, std = np.mean(arr), np.std(arr)
    if std == 0:
        return [0.0] * len(arr)
    return list((arr - mean) / std)

def clip_values(values, min_val, max_val):
    """
    Clip numerical values to a specified range.

    Parameters
    ----------
    values : list
        List of numerical values.
    min_val : float
        Minimum allowed value.
    max_val : float
        Maximum allowed value.

    Returns
    -------
    list
        List of clipped values.

    Examples
    --------
    >>> clip_values([1, 5, 10], 2, 8)
    [2, 5, 8]
    """
    return list(np.clip(values, min_val, max_val))

def convert_to_integers(values):
    """
    Convert values to integers, excluding non-numerical entries.

    Parameters
    ----------
    values : list of str
        List of string values (numerical or non-numerical).

    Returns
    -------
    list
        List of integer values.

    Examples
    --------
    >>> convert_to_integers(['1', 'a', '2.5', '3'])
    [1, 3]
    """
    result = []
    for v in values:
        try:
            num = float(v)
            if num.is_integer():
                result.append(int(num))
        except (ValueError, TypeError):
            continue
    return result

def log_transform(values):
    """
    Apply logarithmic scale transformation to positive numerical values.

    Parameters
    ----------
    values : list
        List of numerical values.

    Returns
    -------
    list
        List of log-transformed values (only for positive inputs).

    Examples
    --------
    >>> log_transform([1, 10, 0, -5])
    [0.0, 2.302585092994046]
    """
    return [math.log(x) for x in values if isinstance(x, (int, float)) and x > 0]

def tokenize_text(text):
    """
    Tokenize text into lowercase alphanumeric words.

    Parameters
    ----------
    text : str
        Text to be processed.

    Returns
    -------
    list
        List of lowercase alphanumeric tokens.

    Examples
    --------
    >>> tokenize_text("Hello, World! 123")
    ['hello', 'world', '123']
    """
    return re.findall(r'\b[a-zA-Z0-9]+\b', text.lower())

def keep_alphanumeric_and_spaces(text):
    """
    Keep only alphanumeric characters and spaces in a text.

    Parameters
    ----------
    text : str
        Text to be processed.

    Returns
    -------
    str
        Processed text with only alphanumeric characters and spaces.

    Examples
    --------
    >>> keep_alphanumeric_and_spaces("Hello! World_123.")
    'Hello World123'
    """
    return re.sub(r'[^A-Za-z0-9 ]+', '', text)

def remove_stopwords(text, stopwords):
    """
    Remove stop-words from a text.

    Parameters
    ----------
    text : str
        Text to be processed (lowercased before filtering).
    stopwords : list
        List of stop-words to remove.

    Returns
    -------
    str
        Text with stop-words removed.

    Examples
    --------
    >>> remove_stopwords("this is a simple test", ["is", "a"])
    'this simple test'
    """
    words = text.lower().split()
    return ' '.join([w for w in words if w not in stopwords])

def flatten_list(list_of_lists):
    """
    Flatten a list of lists into a single list.

    Parameters
    ----------
    list_of_lists : list of lists
        A list containing sublists.

    Returns
    -------
    list
        Flattened list.

    Examples
    --------
    >>> flatten_list([[1, 2], [3, 4]])
    [1, 2, 3, 4]
    """
    return [item for sublist in list_of_lists for item in sublist]

def shuffle_list(values, seed=None):
    """
    Randomly shuffle a list of values.

    Parameters
    ----------
    values : list
        List of values to shuffle.
    seed : int, optional
        Random seed to ensure reproducibility.

    Returns
    -------
    list
        Shuffled list.

    Examples
    --------
    >>> shuffle_list([1, 2, 3], seed=42)
    [2, 1, 3]
    """
    rng = random.Random(seed)
    shuffled = values[:]
    rng.shuffle(shuffled)
    return shuffled

