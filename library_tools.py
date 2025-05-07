# Copyright (c) Iuga Marin
# This file is part of the HuggingFace free AI Agents course assignment.
# It contains utility functions for handling base64 encoding and file retrieval.

import base64
from tools_hfhub import get_GAIA_dataset_file


def get_base_64_file_data_by_path(file_path: str) -> str:
    """
    Converts the content of a file at the given path to a Base64-encoded string.
    Args:
        file_path (str): The path to the file to be encoded.
    Returns:
        str: The Base64-encoded string representation of the file's content.
    """
    file_content = None
    with open(file_path, "rb") as f:
        file_content = f.read()

    return base64.b64encode(file_content).decode("utf-8")


def get_file_data_base_64(file_name: str) -> str:
    """
    Converts a file from the GAIA dataset to its Base64-encoded string representation.
    Args:
        file_name (str): The name of the file from the GAIA dataset to be encoded.
    Returns:
        str: The Base64-encoded string of the file's contents.
    """
    file_location = get_GAIA_dataset_file(file_name)
    base_64_data = get_base_64_file_data_by_path(file_location)

    return base_64_data
