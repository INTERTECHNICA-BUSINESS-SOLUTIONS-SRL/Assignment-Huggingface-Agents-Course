# Copyright (c) Iuga Marin
# This file is part of the HuggingFace free AI Agents course assignment.
# This module generates answers for a single question by its ID using the answers database.

from processing_generate_answers_database import GenerateAnswersDatabase

def generate_answers_for_one_item(item_id: str) -> None:
    """
    Generates answers for a single item specified by its ID.

    Args:
        item_id (str): The unique identifier of the item to process.

    Returns:
        The result of processing the question for the given item ID.
    """
    _generate_answers_database = GenerateAnswersDatabase()
    return _generate_answers_database.process_one_question_by_id(item_id)


if __name__ == "__main__":
    item_id = "example_item_id"  # Replace with the actual item ID you want to process
    result = generate_answers_for_one_item(item_id)
    
    if result[0] == 0:
        print("No response has been generated.")
    else: 
        print(f"The result of processing the question is: \n{result}")
