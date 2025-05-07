import logging
import mimetypes

from setup import get_vision_LLM
from library_tools import get_file_data_base_64

from langchain_core.messages import HumanMessage

# ----------------------------------- Image analysis tool section -----------------------------------
"""
    Loads the content of an image file as a base64 encoded information
"""


def get_requested_information_from_image(file_name: str, query: str) -> str:
    """
    Gets requested information from an image by using a filename and a query. This can be used as a tool.

    Args:
        file_name: The name of the image file
        query: the query used for extracting the information from the image

    Returns:
        The information from the image file.
    """
    base64_image_data = get_file_data_base_64(file_name)
    mime_type = mimetypes.guess_type(file_name)[0]


    logging.debug(f"Using the image information extraction tool on the file: {file_name}")
    logging.debug(f"Inferred mime type is: {mime_type}")


    image_analysis_messages = HumanMessage(content=[
        {
            "type": "text",
            "text": f"""
                <role>
                    You are an agent specialized in image analysis which explores image data and executes the requested task.
                </role>
                <task>
                    We will provide you the data for an image.
                    Perform the query exactly as specified in instructions.
                </task>
                <query>
                    {query}
                </query>
            """,
        },
        {
            "type": "image_url",
            "image_url": {
                "url": f"data:{mime_type};base64,{base64_image_data}"
            },
        }
    ])

    vision_llm = get_vision_LLM()

    output = vision_llm.invoke(
        [image_analysis_messages]
    )

    return output.content
