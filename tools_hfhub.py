from setup import HF_TOKEN
from huggingface_hub import login, hf_hub_download 

# GAIA file download toolset
"""
    Basic function to download a GAIA dataset validation file
"""
def get_GAIA_dataset_validation_file(file_name: str):
    
    response = hf_hub_download(
        repo_id="gaia-benchmark/GAIA", 
        filename= f"2023/validation/{file_name}", 
        repo_type="dataset"
    )
    
    return response

"""
    Basic function to download a GAIA dataset test file
"""
def get_GAIA_dataset_test_file(file_name: str):

    response = hf_hub_download(
        repo_id="gaia-benchmark/GAIA", 
        filename= f"2023/test/{file_name}", 
        repo_type="dataset"
    )
    
    return response

"""
    Basic function to download a GAIA dataset file (validation attempted first)
"""
def get_GAIA_dataset_file(file_name: str):
    login(HF_TOKEN)
    response = None
    try:
        response = get_GAIA_dataset_validation_file(file_name)
    except:
        response = get_GAIA_dataset_test_file(file_name)
            
    return response
