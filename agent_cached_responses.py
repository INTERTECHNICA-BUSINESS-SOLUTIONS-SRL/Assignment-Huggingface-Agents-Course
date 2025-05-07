# Copyright (c) Iuga Marin
# This file is part of the HuggingFace free AI Agents course assignment.
# It contains the implementation of a cached response handler for an AI agent.

import os
import json
import cryptocode

class AgentCachedResponses:
    """
    A class to handle cached responses for an agent. This class reads encrypted 
    answers from a JSON file, decrypts them using a password stored in the 
    environment variable `ANSWERS_DATABASE_PASSWORD`, and provides access to 
    the answers based on a task ID.
    Attributes:
        DATABASE_ANSWERS_ENCRYPTED (str): The file path to the encrypted JSON 
            database containing the answers.
    Methods:
        __init__():
            Initializes the class by loading and decrypting the answers database.
        __call__(task_id: str, question: str, input_file: str) -> str:
            Retrieves the cached answer for a given task ID.
    """
    DATABASE_ANSWERS_ENCRYPTED = "./database/answers_encrypted.json"

    def __init__(self):
        """
        Initializes the class by loading and decrypting cached responses.
        - Reads encrypted JSON data from a file.
        - Decrypts the data using a password from environment variables.
        - Parses the decrypted JSON content into a Python dictionary.
        """
        json_data_raw = None
        with open(AgentCachedResponses.DATABASE_ANSWERS_ENCRYPTED, "r") as f:
            json_data_raw = json.load(f)

        _json_answers_data_content = cryptocode.decrypt(
            json_data_raw["encoded_answers_data"],
            os.environ["ANSWERS_DATABASE_PASSWORD"]
        )

        self._json_answers_data = json.loads(_json_answers_data_content)
        
    def __call__(self, task_id: str, question: str, input_file: str) -> str:
        """
        Execute the callable instance to retrieve a cached answer for a given task.

        Args:
            task_id (str): The unique identifier for the task.
            question (str): The question associated with the task (unused in this method).
            input_file (str): The input file associated with the task (unused in this method).

        Returns:
            str: The cached answer corresponding to the given task ID.
        """
        answer = self._json_answers_data.get(task_id)["answer"]
        return answer
