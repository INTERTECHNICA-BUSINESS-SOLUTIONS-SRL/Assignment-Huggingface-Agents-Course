# Copyright (c) Iuga Marin
# This file is part of the HuggingFace free AI Agents course assignment.
# It contains utility functions for video file transcription and analysis.

import logging
import mimetypes

from library_tools import get_base_64_file_data_by_path
from setup import get_video_LLM
from langchain_core.messages import HumanMessage


def get_transcribed_video(video_file_path: str) -> str:
    """
    Gets the transcribed video file such as mp4, obtaining the transcription as text. This can be used as a tool.

    Args:
        video_file_path: The path of the video file.

    Returns:
        The transcription of the video file, returned as text.
    """
    logging.debug(f"Video transcription tool is called.")
    logging.debug(f"File path: {video_file_path}")

    mime_type = mimetypes.guess_type(video_file_path)[0]
    logging.debug(f"Inferred mime type is: {mime_type}")

    base64_video_data = get_base_64_file_data_by_path(video_file_path)

    video_transcription_messages = HumanMessage(content=[
        {
            "type": "text",
            "text": f"""
                <role>
                    You are an agent specialized in video transcription.
                </role>
                <task>
                    We will provide you the data for an video file.
                    Transcribe the content of the video file with the highest fidelity possible.
                </task>
            """
        },
        {
            "type": "media",
            "data": base64_video_data,
            "mime_type": mime_type
        }
    ])

    vision_llm = get_video_LLM()

    output = vision_llm.invoke(
        [video_transcription_messages]
    )

    transcription_content = output.content
    logging.debug(f"Obtained video transcription content: {transcription_content}")

    return transcription_content


def get_analysis_information_from_video(video_file_path: str, query: str) -> str:
    """
    Analyzes a video file such as mp4, obtaining the information from the file. This can be used as a tool.

    Args:
        video_file_path: The path of the video file.
        query: The query used for the video analysis.

    Returns:
        The information analysis from the video.
    """
    logging.debug(f"Video analysis tool is called.")
    logging.debug(f"File path: {video_file_path}")
    logging.debug(f"Query: {query}")

    mime_type = mimetypes.guess_type(video_file_path)[0]
    logging.debug(f"Inferred mime type is: {mime_type}")

    base64_video_data = get_base_64_file_data_by_path(video_file_path)

    video_analysis_messages = HumanMessage(content=[
        {
            "type": "text",
            "text": f"""
                <role>
                    You are an agent specialized in video analysis.
                    You analyze the video in great detail and provide exact and detailed analysis information.
                    We will provide detailed instructions in the task section.
                </role>
                <task>
                    We will provide you the data for an video file. Analyze the video file using the provided query and return the analysis information.
                    Analyze each video frame in detail before generating the final answer. Explain in detail your choice step by step.
                    Once the final answer is generated verify it against the entire video again to make sure is correct.
                </task>
                <query>
                    {query}
                </query>
            """
        },
        {
            "type": "media",
            "data": base64_video_data,
            "mime_type": mime_type
        }
    ])

    vision_llm = get_video_LLM()

    output = vision_llm.invoke(
        [video_analysis_messages]
    )

    analysis_content = output.content
    logging.debug(f"Obtained video analysis content: {analysis_content}")

    return analysis_content
