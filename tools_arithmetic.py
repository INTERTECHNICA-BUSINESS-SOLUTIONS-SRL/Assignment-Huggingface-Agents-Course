import numpy as np
from typing import List

# ----------------------------------- Basic tools section -----------------------------------


def add_values(values_list: List[float]) -> float:
    """
    Adds values from a list, useful for calculations when dealing with adding multiple values. This can be used as a tool.

    Args:
        values: a list of values to be added. If you have multiple numbers to add, place them in the list.
    Returns:
        The sum of all values.
    """
    sum = np.sum(values_list)
    return sum
