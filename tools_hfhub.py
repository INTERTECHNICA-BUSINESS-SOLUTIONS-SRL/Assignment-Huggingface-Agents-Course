# Copyright (c) Iuga Marin
# This file is part of the HuggingFace free AI Agents course assignment.
# It contains utility functions for downloading GAIA dataset files from the Hugging Face Hub.

from setup import HF_TOKEN
from huggingface_hub import login, hf_hub_download 

def get_GAIA_dataset_validation_file(file_name: str) -> str:
    """
    Downloads a validation file from the GAIA dataset hosted on Hugging Face Hub.
    Args:
        file_name (str): The name of the validation file to download.
    Returns:
        str: The local file path to the downloaded validation file.
    """
    
    response = hf_hub_download(
        repo_id="gaia-benchmark/GAIA", 
        filename= f"2023/validation/{file_name}", 
        repo_type="dataset"
    )
    
    return response

def get_GAIA_dataset_test_file(file_name: str) -> str:
    """
    Downloads a specific test file from the GAIA dataset on Hugging Face Hub.
    Args:
        file_name (str): The name of the test file to download from the GAIA dataset.
    Returns:
        str: The local file path to the downloaded test file.
    """

    response = hf_hub_download(
        repo_id="gaia-benchmark/GAIA", 
        filename= f"2023/test/{file_name}", 
        repo_type="dataset"
    )
    
    return response

def get_GAIA_dataset_file(file_name: str) -> str:
    """
    Retrieves the specified GAIA dataset file, attempting to fetch the validation file first and falling back to the test file if necessary.
    Args:
        file_name (str): The name of the dataset file to retrieve.
    Returns:
        str: The path or identifier of the retrieved dataset file.
    """
    login(HF_TOKEN)
    response = None
    try:
        response = get_GAIA_dataset_validation_file(file_name)
    except:
        response = get_GAIA_dataset_test_file(file_name)
            
    return response
