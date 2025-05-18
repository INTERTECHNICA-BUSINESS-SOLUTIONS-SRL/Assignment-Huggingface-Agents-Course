# Copyright (c) Iuga Marin
# This file is part of the HuggingFace free AI Agents course assignment.
# This module encrypts the answers database and saves it to a file.

import os
import json
import cryptocode
import datetime

import dotenv

from processing_generate_answers_database import DATABASE_ANSWERS

DATABASE_ANSWERS_ENCRYPTED = "./database/answers_encrypted.json"


def encrypt_answers_database():
    """
    Encrypts the answers from the database file, replaces the plain answers with the encrypted data,
    updates the date, and writes the result to an encrypted database file.
    Raises:
        ValueError: If the required password for encryption is not set.
    """
    if os.environ["ANSWERS_DATABASE_PASSWORD"] is None:
        dotenv.load_dotenv()
        if os.environ["ANSWERS_DATABASE_PASSWORD"] is None:
            raise ValueError("The environment variable 'ANSWERS_DATABASE_PASSWORD' is not set.")

    json_data = None
    with open(DATABASE_ANSWERS, "r") as f:
        json_data = json.load(f)

    json_answers = json_data["answers"]
    json_answers_string = json.dumps(json_answers, indent=2)
    encoded_answers = cryptocode.encrypt(json_answers_string, os.environ["ANSWERS_DATABASE_PASSWORD"])

    json_data["encoded_answers_data"] = encoded_answers
    del json_data["answers"]
    json_data["date"] = str(datetime.datetime.now()),

    with open(DATABASE_ANSWERS_ENCRYPTED, "w") as f:
        json.dump(json_data, f, indent=2)

    return


if __name__ == "__main__":
    encrypt_answers_database()
