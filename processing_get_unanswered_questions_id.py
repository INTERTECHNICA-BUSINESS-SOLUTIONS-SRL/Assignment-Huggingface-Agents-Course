# Copyright (c) Iuga Marin
# This file is part of the HuggingFace free AI Agents course assignment.
# This module retrieves the IDs of unanswered questions from the answers database.

from typing import List
from processing_generate_answers_database import GenerateAnswersDatabase

def get_unanswered_questions_ids() -> List[str]:
    """
    Returns a list of question IDs that do not have answers in the answers database.

    Returns:
        List[str]: List of unanswered question IDs.
    """
    _generate_answers_database = GenerateAnswersDatabase()
    unanswered_questions_ids = _generate_answers_database.get_unanswered_questions_ids()
    return unanswered_questions_ids

if __name__ == "__main__":
    unanswered_questions_ids = get_unanswered_questions_ids()
    if len(unanswered_questions_ids) == 0:
        print("All questions have answers.")
    else:
        print(f"The following questions do not have answers: {unanswered_questions_ids}.")
