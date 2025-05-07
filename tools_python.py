from tools_hfhub import get_GAIA_dataset_file

# ----------------------------------- Python tools section -----------------------------------


def get_python_file_data(file_name: str) -> str:
    """
    Gets a Python script file content based on the file name. This can be used as a tool.

    Args:
        file_name: The name of the Python script file

    Returns:
        The content of the Python file
    """
    file_location = get_GAIA_dataset_file(file_name)
    with open(file_location) as f:
        return f.read()
