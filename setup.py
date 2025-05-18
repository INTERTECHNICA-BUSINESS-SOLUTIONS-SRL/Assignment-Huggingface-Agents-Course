# Copyright (c) Iuga Marin
# This file is part of the HuggingFace free AI Agents course assignment.
# This module sets up environment variables, logging, and LLM configuration functions.

import os
import logging

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

# load dotenv and check API keys are set
load_dotenv()

assert not os.environ["GOOGLE_API_KEY"] is None
assert not os.environ["HF_TOKEN"] is None
assert not os.environ["TAVILY_API_KEY"] is None

# retrieve environment variables
GOOGLE_API_KEY = os.environ["GOOGLE_API_KEY"]
HF_TOKEN = os.environ["HF_TOKEN"]

# Gemini models used for inference
GEMINI_PRO = "gemini-2.5-pro-exp-03-25"
GEMINI_FLASH = "gemini-2.0-flash"

# change global logging
TARGET_LOGGING_LEVEL = logging.DEBUG
logging.basicConfig(
    level=TARGET_LOGGING_LEVEL,
    filename='logging.log',
    filemode='w',
    format='%(asctime)s - %(levelname)s - %(module)s - %(funcName)s : %(message)s'
)

# forcefully disable some modules
FORCED_DISABLED_MODULES = [
    "_login"
]

# log only for modules of interest
ENABLED_MODULES = [
    "tools_arithmetic",
    "tools_web",
    "tools_excel",
    "tools_image",
    "tools_chess",
    "tools_audio",
    "tools_video",
    "tools_youtube",
    "agent_basic_tooling",
    "agent_final_answer",
    "generate_answers_database"
]

# enable high level logging only for modules which are not of interest
for name, _ in logging.root.manager.loggerDict.items():
    if name in ENABLED_MODULES:
        continue
    logging.getLogger(name).setLevel(logging.ERROR)

# completely disable logging for modules which are explicitly disabled
for name, _ in logging.root.manager.loggerDict.items():
    if name in FORCED_DISABLED_MODULES:
        logging.getLogger(name).disabled = True


def get_baseline_LLM() -> ChatGoogleGenerativeAI:
    """
    Returns a baseline language model instance suitable for general-purpose tasks.

    Returns:
        ChatGoogleGenerativeAI: A language model instance configured for baseline usage.
    """

    baseline_llm = ChatGoogleGenerativeAI(
        model=GEMINI_FLASH,
        temperature=0.25,
        max_tokens=None,
        timeout=None,
        max_retries=2
    )

    return baseline_llm


def get_EXCEL_calculation_LLM():
    """
    Returns a language model instance configured for Excel calculations.

    Returns:
        ChatGoogleGenerativeAI: A language model instance for Excel calculation tasks.
    """

    return ChatGoogleGenerativeAI(
        model=GEMINI_FLASH,
        temperature=0.25,
        max_tokens=None,
        timeout=None,
        max_retries=2
    )


def get_query_optimization_LLM() -> ChatGoogleGenerativeAI:
    """
    Returns a language model instance configured for query optimization tasks.

    Returns:
        ChatGoogleGenerativeAI: A language model instance for query optimization.
    """

    query_optimization_llm = ChatGoogleGenerativeAI(
        model=GEMINI_FLASH,
        max_tokens=None,
        temperature=0.25,
        top_p=0.95,
        timeout=None,
        max_retries=2,
    )

    return query_optimization_llm


def get_content_relevance_LLM() -> ChatGoogleGenerativeAI:
    """
    Initializes and returns a language model instance for content relevance evaluation.
    Returns:
        ChatGoogleGenerativeAI: An instance of a chat-based language model configured for content relevance tasks.
    """

    content_relevance_llm = ChatGoogleGenerativeAI(
        model=GEMINI_FLASH,
        max_tokens=None,
        temperature=0.25,
        top_p=0.95,
        timeout=None,
        max_retries=2,
    )

    return content_relevance_llm


def get_strict_content_analysis_LLM() -> ChatGoogleGenerativeAI:
    """
    Creates and returns a language model instance configured for strict content analysis.
    Returns:
        ChatGoogleGenerativeAI: An initialized language model for content analysis tasks.
    """

    strict_content_analysis_llm = ChatGoogleGenerativeAI(
        model=GEMINI_FLASH,
        max_tokens=None,
        temperature=0.25,
        top_p=0.95,
        timeout=None,
        max_retries=2,
    )

    return strict_content_analysis_llm


def get_loose_content_analysis_LLM() -> ChatGoogleGenerativeAI:
    """
    Creates and returns a language model instance for loose content analysis.
    Returns:
        ChatGoogleGenerativeAI: An initialized language model for content analysis tasks.
    """

    loose_content_analysis_llm = ChatGoogleGenerativeAI(
        model=GEMINI_FLASH,
        max_tokens=None,
        temperature=0.75,
        top_p=0.75,
        timeout=None,
        max_retries=2,
    )

    return loose_content_analysis_llm


def get_chess_analysis_LLM():
    """
    Creates and returns a language model instance specialized for chess analysis.
    Returns:
        An initialized language model object for analyzing chess games and positions.
    """

    chess_analysis_llm = ChatGoogleGenerativeAI(
        model=GEMINI_FLASH,
        max_tokens=None,
        timeout=None,
        max_retries=2
    )

    return chess_analysis_llm


def get_vision_LLM():
    """
    Initializes and returns a vision-capable language model interface.
    Returns:
        An instance of a vision-enabled language model ready for image processing tasks.
    """

    vision_llm = ChatGoogleGenerativeAI(
        model=GEMINI_FLASH,
        max_tokens=None,
        timeout=None,
        max_retries=2
    )

    return vision_llm


def get_video_LLM():
    """
    Creates and returns a language model instance optimized for video-related tasks.
    Returns:
        An initialized language model for video processing.
    """

    video_llm = ChatGoogleGenerativeAI(
        model=GEMINI_FLASH,
        max_tokens=None,
        timeout=None,
        max_retries=2
    )

    return video_llm


def get_audio_LLM():
    """
    Initializes and returns an audio language model interface for conversational tasks.
    Returns:
        An instance of a language model configured for audio-related interactions.
    """

    audio_llm = ChatGoogleGenerativeAI(
        model=GEMINI_FLASH,
        max_tokens=None,
        timeout=None,
        max_retries=2
    )

    return audio_llm


def get_final_answer_LLM():
    """
    Creates and returns a language model instance for generating final answers.
    Returns:
        An initialized language model object for answer generation.
    """

    final_answer_llm = ChatGoogleGenerativeAI(
        model=GEMINI_FLASH,
        top_p=0.95,
        max_tokens=None,
        timeout=None,
        max_retries=2
    )

    return final_answer_llm
