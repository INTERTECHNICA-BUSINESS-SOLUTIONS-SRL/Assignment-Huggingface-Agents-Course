# Copyright (c) Iuga Marin
# This file is part of the HuggingFace free AI Agents course assignment.
# It contains utility functions for handling base64 encoding and file retrieval.

import base64
from inspect import signature
from typing import List, Callable
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


def get_tool_description(tool: Callable) -> str:
    """
    Generate a formatted description of a given tool function.
    This function extracts the name, signature, and docstring of the provided tool
    function and combines them into a single formatted string.
    Args:
        tool (Callable): The tool function to describe. It must be a callable object.
    Returns:
        str: A formatted string containing the tool's name, signature, and docstring.
    """
    tool_name = tool.__name__
    tool_signature = signature(tool)
    tool_doc = tool.__doc__.strip("\n").strip("\n")

    tool_description = f"{tool_name}{tool_signature}\n{tool_doc}"

    return tool_description


def get_tools_description(tools: List[Callable]) -> str:
    """
    Retrieves a description of all available tools.
    Returns:
        str: A string containing descriptions of all tools, separated by newlines.
    """
    tools = tools
    tools_description = ""
    for tool in tools:
        tools_description = tools_description + "\n" + get_tool_description(tool)

    return tools_description
