import pandas as pd
from tools_hfhub import get_GAIA_dataset_file

# ----------------------------------- Excel tools section -----------------------------------
"""
    Loads the content of an Excel file as markdown content
"""


def get_EXCEL_file_content_as_markdown(file_name: str):
    file_location = get_GAIA_dataset_file(file_name)
    data_frame = pd.read_excel(file_location, header=0)

    result = data_frame.to_markdown()

    return result


def get_excel_file_data(file_name: str) -> str:
    """
    Gets the Excel file content based on file name. This can be used as a tool.

    Args:
        file_name: The name of the excel file

    Returns:
        The content of the excel file
    """
    return get_EXCEL_file_content_as_markdown(file_name)
