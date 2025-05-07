import logging

from pytubefix import YouTube
from pytubefix.cli import on_progress

from tools_video import get_analysis_information_from_video

# ----------------------------------- Youtube video download tool section -----------------------------------


def get_youtube_video(video_url: str) -> str:
    """
    Downloads a video from youtube using the video URL.

    Args:
        video_url: the URL for the YouTube video

    Returns:
        The path of the downloaded video file.
    """
    VIDEOS_DIRECTORY_CACHE = "./videos"

    you_tube_proxy = YouTube(video_url)
    you_tube_stream = you_tube_proxy.streams.get_highest_resolution()
    video_file_path = you_tube_stream.download(VIDEOS_DIRECTORY_CACHE)

    return video_file_path

# ----------------------------------- Youtube video analysis tool section -----------------------------------


def get_analysis_information_from_youtube_video(youtube_video_url: str, query: str) -> str:
    """
    This is the preferred tool when asked for information from an youtube video. 
    It should be used before any web searches are made.
    Analyzes a youtube vide using its url and a query. 
    If the Youtube video URL is not provided, it should be inferred from the query.
    This can be used as a tool.

    Args:
        youtube_video_url: The YouTube video URL. IF it is not explicitly provided, it should be inferred from the query.
        query: The query used for analysis.

    Returns:
        The information analysis from the video.
    """
    logging.debug(f"Youtube video analysis tool is called.")
    logging.debug(f"Youtube video URL: {youtube_video_url}")
    logging.debug(f"Query: {query}")

    video_file_path = get_youtube_video(youtube_video_url)
    video_analysis_content = get_analysis_information_from_video(video_file_path, query)
    logging.debug(f"Obtained video analysis content: {video_analysis_content}")

    return video_analysis_content
