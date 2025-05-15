# Copyright (c) Iuga Marin
# This file is part of the HuggingFace free AI Agents course assignment.
# It contains utility functions for analyzing images and extracting information based on queries.

import logging
from setup import get_chess_analysis_LLM
from tools_image import get_requested_information_from_image

from langchain_core.messages import HumanMessage


def get_chessboard_information(file_name: str) -> str:
    """
    Gets the chessboard information from an image file name. This can be used as a tool.

    Args:
        file_name: The name of the image file
        query: the query used for extracting the information from the image

    Returns:
        The chessboard information extracted from the image
    """
    logging.debug(f"Get chessboard information from image tool called!")
    logging.debug(f"File name: {file_name}")

    query = """
    You are a chess expert, specialized in transforming chessboard images into FEN (Forsyth–Edwards Notation).
    Analyze the provided image and extract the chessboard information into FEN notation.
    Once you extracted the FEN notation, verify it visually against the  provided image.
    
    Return only the chessboard information in FEN notation, without any additional text or explanation.
    """

    response = get_requested_information_from_image(file_name, query)
    logging.debug(f"Response: {response}")

    return response


def get_chess_analysis_information_from_image(file_name: str, query: str) -> str:
    """
    Queries a chess related information from an image.
    Always use this when asked to analyze a chessboard image and provide chess related information. 
    This can be used as a tool.

    Args:
        file_name: The name of the image file
        query: the query used for extracting the information from the image

    Returns:
        The chessboard information extracted from the image
    """
    logging.debug(f"Query chessboard information from image tool called!")
    logging.debug(f"File name: {file_name}")
    logging.debug(f"Query: {query}")

    chessboard_information = get_chessboard_information(file_name)

    image_analysis_messages = HumanMessage(content=[
        {
            "type": "text",
            "text": f"""
                <role>
                    You are an agent specialized in an in depth chess analysis. 
                    You work with chessboard information even if this information is incomplete or incorrect. 
                </role>
                <task>
                    You will receive the chessboard information into FEN notation. This information may be incomplete or incorrect.
                    Do your best to analyze the chessboard based on the information provided and provide the information requested in the query.
                </task>
                <chessboard_information>
                    {chessboard_information}
                </chessboard_information>    
                <query>
                    {query}
                </query>
            """
        }
    ])

    chess_llm = get_chess_analysis_LLM()

    output = chess_llm.invoke(
        [image_analysis_messages]
    )

    logging.debug(f"Obtained content is: {output.content}")

    return output.content
