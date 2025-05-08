# Copyright (c) Iuga Marin
# This file is part of the HuggingFace free AI Agents course assignment.
# It contains utility functions for reading Excel files and converting them to markdown format.

import logging
import pandas as pd

from langchain_core.messages import HumanMessage

from tools_hfhub import get_GAIA_dataset_file
from setup import get_EXCEL_calculation_LLM


def get_EXCEL_file_content_as_markdown(file_name: str) -> str:
    """
    Returns Excel file content in markdown table format. This can be used as a tool.

    Args:
        file_name (str): Name of the Excel file to read.

    Returns:
        str: Content of Excel file formatted as markdown table.
    """
    logging.debug(f"EXCEL file content extraction as markdown tool called.")
    logging.debug(f"Reading Excel file: {file_name}")

    file_location = get_GAIA_dataset_file(file_name)
    data_frame = pd.read_excel(file_location, header=0)

    result = data_frame.to_markdown()

    logging.debug(f"Extracted EXCEL file content as markdown: \n{result}")

    return result

def get_EXCEL_file_content_as_csv(file_name: str) -> str:
    """
    Converts an Excel file to CSV format and returns its content as a CSV string. This can be used as a tool.
    Args:
        file_name (str): Name of the Excel file to be converted.
    Returns:
        str: Content of the Excel file in CSV format.
    """
    logging.debug(f"EXCEL file content extraction as CSV tool called.")
    logging.debug(f"Reading Excel file: {file_name}")

    file_location = get_GAIA_dataset_file(file_name)
    data_frame = pd.read_excel(file_location, header=0)

    result = data_frame.to_csv(index=False)

    logging.debug(f"Extracted EXCEL file content as csv: \n{result}")

    return result


def process_EXCEL_file(file_name: str, query: str) -> str:
    """
    Performs a calculation on an EXCEL file using a query. This must be used as a tool when Excel files are processed.
    Args:
        file_name (str): Name of the Excel file to be converted.
        query (str): The query used to process the EXCEL file.
    Returns:
        str: Content of the Excel file in CSV format.
    """
    logging.debug(f"EXCEL file processing tool called.")
    logging.debug(f"EXCEL file: {file_name}")
    logging.debug(f"Processing query: {query}")

    file_content = get_EXCEL_file_content_as_csv(file_name)

    EXCEL_analysis_messages = HumanMessage(content=[
        {
            "type": "text",
            "text": f"""
                <role>
                    You are an agent specialized in Excel file analysis and performing data selection and calculations.
                    We will provide you the data for an Excel file using the CSV format. 
                </role>
                <task>
                    We will provide you the data for the EXCEL file in CSV format.
                    Perform the calculations specified in the query exactly as specified in instructions over the EXCEL data we provided.
                    Explain the steps you took to perform the calculations and provide the final result using the prefix FINAL RESULT:
                </task>
                <excel_data>
                    {file_content}
                </excel_data>
                <query>
                    {query}
                </query>
            """
        }
    ])

    excel_calculation_llm = get_EXCEL_calculation_LLM()

    output = excel_calculation_llm.invoke(
        [EXCEL_analysis_messages]
    )

    result = output.content

    logging.debug(f"The result of the calculation is: {result}")

    return result