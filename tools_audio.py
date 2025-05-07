import logging
import mimetypes

from library_tools import get_file_data_base_64
from setup import get_audio_LLM
from langchain_core.messages import HumanMessage

# ----------------------------------- Audio analysis tool section -----------------------------------
def get_transcribed_audio_file_data(file_name: str) -> str:
    """
    Gets the transcribed audio file such as mp3 audio file. This can be used as a tool.

    Args:
        file_name: The name of the audio file.
        
    Returns:
        The transcribed text of the audio file.
    """
    base64_audio_data = get_file_data_base_64(file_name)
    mime_type = mimetypes.guess_type(file_name)[0]

    logging.debug(f"Using the audio transcription tool on the file: {file_name}")
    logging.debug(f"Inferred mime type is: {mime_type}")
    
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

    vision_llm = get_audio_LLM()

    output = vision_llm.invoke(
        [audio_analysis_messages]
    )

    logging.debug(f"The transcribed audio content is: {output.content}")
    
    return output.content