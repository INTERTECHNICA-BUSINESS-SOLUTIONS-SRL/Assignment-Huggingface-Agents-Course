# Copyright (c) Iuga Marin
# This file is part of the HuggingFace free AI Agents course assignment.
# It contains utility functions for performing arithmetic operations.

import logging
import numpy as np

from typing import List

def add_values(a: float, b: float) -> float:
    """
    Adds two values together and returns the result. 
    Always use this tool when you need to add two values, do not try to add these values yourself. 
    This can be used as a tool.

    Args:
        a (float): The first value to add.
        b (float): The second value to add.

    Returns:
        float: The sum of the two values.
    """
    logging.debug(f"Adding values: {a} and {b}")
    result = float(a + b)
    logging.debug(f"Result of addition: {result}")
    return a + b


def subtract_values(a: float, b: float) -> float:
    """
    Subtracts the second value from the first and returns the result. 
    Always use this tool when you need to subtract two values, do not try to subtract these values yourself. 
    This can be used as a tool.

    Args:
        a (float): The first value - from what to subtract.
        b (float): The second value - what is subtracted.

    Returns:
        float: The result of subtracting the second value from the first value.
    """
    logging.debug(f"Subtracting values: {a} and {b}")
    result = float(a - b)
    logging.debug(f"Result of addition: {result}")
    return a - b


def add_multiple_values(values: List[float]) -> float:
    """
    Adds multiple values together and returns the result. 
    Always use this tool when you need to add multiple values, do not try to add these values yourself. 
    This can be used as a tool.

    Args:
        values List(float): The values to be added.

    Returns:
        float: The sum of the values in the list.
    """
    logging.debug(f"Adding multiple values tool called.")
    logging.debug(f"Adding values: {values}")
    result = float(np.sum(values))
    logging.debug(f"Result of addition: {result}")
    return result
