import logging
import mimetypes

from library_tools import get_file_data_base_64
from setup import get_audio_LLM
from langchain_core.messages import HumanMessage


def get_transcribed_audio_file_data(file_name: str) -> str:
    """
    Gets the transcribed audio file such as mp3 audio file. This can be used as a tool.

    Args:
        file_name: The name of the audio file.

    Returns:
        The transcribed text of the audio file.
    """
    logging.debug(f"File transcription tool called!")
    logging.debug(f"File name: {file_name}")

    mime_type = mimetypes.guess_type(file_name)[0]

    logging.debug(f"Inferred mime type is: {mime_type}")

    base64_audio_data = get_file_data_base_64(file_name)

    audio_analysis_messages = HumanMessage(content=[
        {
            "type": "text",
            "text": f"""
                <role>
                    You are an agent specialized in audio analysis and audio transcription.
                </role>
                <task>
                    We will provide you the data for an audio file.
                    Transcribe the content of the audio file with the highest fidelity possible.
                </task>
            """
        },
        {
            "type": "media",
            "data": base64_audio_data,
            "mime_type": mime_type
        }
    ])

    audio_llm = get_audio_LLM()

    output = audio_llm.invoke(
        [audio_analysis_messages]
    )

    logging.debug(f"The transcribed audio content is: {output.content}")

    return output.content

def get_analysis_information_from_audio_file(file_name: str, query: str) -> str:
    """
    Analyzes an audio file such as mp3, obtaining the information from the file. This can be used as a tool.

    Args:
        file_name: The name of the audio file.
        query: The query used for the audio file analysis.

    Returns:
        The information analysis from the audio file.
    """
    logging.debug(f"Audio file analysis tool is called.")
    logging.debug(f"File name: {file_name}")
    logging.debug(f"Query: {query}")

    mime_type = mimetypes.guess_type(file_name)[0]
    logging.debug(f"Inferred mime type is: {mime_type}")

    base64_audio_data = get_file_data_base_64(file_name)

    audio_analysis_messages = HumanMessage(content=[
        {
            "type": "text",
            "text": f"""
                <role>
                    You are an agent specialized in audio analysis.
                    You analyze the audio in great detail and provide exact and detailed analysis information.
                    We will provide detailed instructions in the task section.
                </role>
                <task>
                    We will provide you the data for an audio file. Analyze the audio file using the provided query and return the analysis information.
                    Explain in detail your choice step by step.
                    Once the final answer is generated verify it against the entire audio file again to make sure it is correct in regards with the audio content and the query.
                </task>
                <query>
                    {query}
                </query>
            """
        },
        {
            "type": "media",
            "data": base64_audio_data,
            "mime_type": mime_type
        }
    ])


    audio_llm = get_audio_LLM()

    output = audio_llm.invoke(
        [audio_analysis_messages]
    )

    analysis_content = output.content
    logging.debug(f"Obtained audio analysis content: {analysis_content}")
    
    return analysis_content