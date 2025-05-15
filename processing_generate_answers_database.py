# Copyright (c) Iuga Marin
# This file is part of the HuggingFace free AI Agents course assignment.
# This module manages the questions and answers databases for the assignment,
# including loading, caching, validating, and updating answers with agentic traces.

import os
import json
import time
import datetime
import hashlib
import base64
import logging

from typing import Dict, Tuple, List, Any

from tools_hfhub import get_GAIA_dataset_file
from agent_final_answer import AgentFinalAnswer


DATABASE_QUESTIONS = "./database/questions.json"
DATABASE_ANSWERS = "./database/answers.json"


class GenerateAnswersDatabase():
    """
    Manages the questions and answers databases for the HuggingFace Agents Course Assignment.

    Loads questions from a JSON file and maintains a cache of generated answers, including file integrity checks
    and agentic trace logging. Supports processing individual or all questions, caching results, and ensuring answer validity.
    """

    def __init__(self):
        """
        Initializes the instance.

        Ensures the required databases are available and loads their contents.
        Raises:
            Exception: If the questions database file is not found.
        """

        if not os.path.isfile(DATABASE_QUESTIONS):
            raise Exception(f"Questions database file not found: {DATABASE_QUESTIONS} ")
        if not os.path.isfile(DATABASE_ANSWERS):
            logging.warning(f"Answers database file will be created {DATABASE_ANSWERS}")
            with open(DATABASE_ANSWERS, "w", encoding="utf-8") as f:
                init_json = {
                    "title": "AGENTIC ANSWERS FILE",
                    "version": "1.0 RC1",
                    "description": """
                        Answers for the HuggingFace Agents Course Assignment.
                        The data was generated locally and cached in this file.
                        Agentic calls tracing is provided as proof of work.
                        Contact me for access to agentic implementation.
                    """,
                    "date": str(datetime.datetime.now()),
                    "answers": {}
                }
                json.dump(init_json, f, indent=2)

        with open(DATABASE_QUESTIONS, "r", encoding="utf-8") as f:
            self._questions_json = json.load(f)

        with open(DATABASE_ANSWERS, "r", encoding="utf-8") as f:
            self._answers_json = json.load(f)

    def _hash_file(self, file_name):
        """
        Computes a base64-encoded SHA-256 hash of the specified file.

        Args:
            file_name (str): The name of the file to hash.

        Returns:
            str: The base64-encoded SHA-256 hash of the file.
        """
        downloaded_file_name = get_GAIA_dataset_file(file_name)
        with open(downloaded_file_name, "rb") as f:
            file_digest = hashlib.file_digest(f, "sha256")
            return base64.b64encode(file_digest.digest()).decode("utf-8")

    def _update_answers(self):
        """
        Updates the answers database file with the current answers in JSON format.
        """
        with open(DATABASE_ANSWERS, "w", encoding="utf-8") as f:
            json.dump(self._answers_json, f,  indent=2)
            f.flush()

    def _validate_question_and_cached_answer(self, question_item: Dict, answer_item: Dict):
        """
        Validates that the question and answer items correspond to each other and, if files are attached, ensures the files have matching content.
        Args:
            question_item (Dict): The dictionary containing question data.
            answer_item (Dict): The dictionary containing answer data.
        Raises:
            AssertionError: If the question and answer do not match, or if attached files differ.
        """
        assert question_item["question"] == answer_item["question"], "Answer should fit the question"

        file_name = question_item["file_name"]

        if len(file_name) > 0:
            question_file_digest = self._hash_file(question_item["file_name"])
            answer_file_digest = self._hash_file(answer_item["file_name"])

            assert question_file_digest == answer_file_digest, "Question and answer attached files should have the same content"

    def _check_cached_answer(self, question_item: Dict) -> bool:

        logging.debug(f"Checking if an answer is cached for the question: {question_item}")

        if question_item["task_id"] in self._answers_json["answers"]:
            answer_item = self._answers_json["answers"][question_item["task_id"]]
            logging.debug(f"Cached answer was found: {answer_item["answer"]}")
            self._validate_question_and_cached_answer(question_item, answer_item)
            logging.debug(f"Cached answer was validated")

            return True
        else:
            logging.debug(f"No cached answer was found.")
            return False

    def _get_answer_for_question(self, question: str, input_file: str = None) -> Tuple[str, str]:
        """
        Retrieves the intermediate and final answers for a given question.
        Args:
            question (str): The question to be answered.
            input_file (str, optional): Path to an input file with additional context. Defaults to None.
        Returns:
            Tuple[str, str]: A tuple containing the intermediate answers and the final answer.
        """
        _agent_final_answer = AgentFinalAnswer()
        intermediate_answers, answer = _agent_final_answer(question, input_file)

        # sleep to prevent over quota processing
        time.sleep(15)

        return intermediate_answers, answer

    def _get_agentic_trace(self, intermediate_answers: List[Any], answer: str) -> str:
        """
        Generates a formatted trace log from a sequence of intermediate answers and a final answer.
        Args:
            intermediate_answers (List[Any]): A dictionary containing a list of intermediate message objects under the "messages" key.
            answer (str): The final answer to be included in the trace.
        Returns:
            str: A formatted string representing the trace of intermediate and final answers.
        """
        call_log = ""

        for intermediate_answer in intermediate_answers["messages"]:
            intermediate_answer_log = f"*** {intermediate_answer.__class__.__name__} *** \n{intermediate_answer.content} \n\n"
            call_log = call_log + intermediate_answer_log

        final_answer_log = f"=== FINAL ANSWER === \n{answer}"
        call_log = call_log + final_answer_log

        return call_log

    def process_one_question(self, question_item: Dict) -> Tuple[bool, Dict]:
        """
        Processes a single question item, generating an answer and updating the answers database.
        Args:
            question_item (Dict): A dictionary containing question data.
        Returns:
            Tuple[bool, Dict]: A tuple where the first element indicates if processing occurred,
            and the second is the answer item dictionary or None if skipped.
        """
        logging.debug(f"Processing question: {question_item}")

        if self._check_cached_answer(question_item):
            logging.debug("Cached response was found, skipping processing.")
            return False, None

        answer_item = {}
        answer_item["question"] = question_item["question"]
        answer_item["file_name"] = question_item["file_name"]

        file_name = answer_item["file_name"]

        if len(file_name) > 0:
            answer_item["file_digest"] = self._hash_file(file_name)
        else:
            answer_item["file_digest"] = ""

        question = question_item["question"]
        input_file = question_item["file_name"] if len(question_item["file_name"]) > 0 else None

        intermediate_answers, answer = self._get_answer_for_question(question, input_file)

        answer_item["agentic_trace"] = self._get_agentic_trace(intermediate_answers, answer)
        answer_item["answer"] = answer
        logging.debug(f"Obtained agentic answer: {answer_item["answer"]}")

        self._answers_json["answers"][question_item["task_id"]] = answer_item

        return True, answer_item

    def process_one_question_by_id(self, question_id: str) -> Tuple[bool, Dict]:
        """
        Processes a single question identified by its ID.
        Args:
            question_id (str): The unique identifier of the question to process.
        Returns:
            Tuple[bool, Dict]: A tuple containing a boolean indicating if the response was generated, and a dictionary with the answer item.
        Raises:
            Exception: If no question is found for the given ID.
        """
        logging.debug(f"Received request to process one item by id: {question_id}")
        question_item = None
        for question_json in self._questions_json:
            if question_json["task_id"] == question_id:
                question_item = question_json
                break

        if question_item is not None:
            logging.debug(f"Question item retrieved and sent to processing: \n {question_item}")
            is_response_generated, answer_item = self.process_one_question(question_item)
            if is_response_generated:
                self._update_answers()
            return is_response_generated, answer_item
        else:
            logging.error(f"Question item not found. Processing stopped.")
            raise Exception(f"No question found for id {question_id}")

    def get_unanswered_questions_ids(self) -> List[str]:
        """
        Returns a list of question IDs that do not have corresponding answers.
        Returns:
            List[str]: A list of unanswered question IDs.
        """
        logging.debug(f"Received request to get unanswered questions.")

        unanswered_questions = []
        for question_json in self._questions_json:
            question_id = question_json["task_id"]
            if question_id in self._answers_json["answers"]:
                continue
            unanswered_questions.append(question_id)

        logging.debug(f"Retrieved unanswered questions list: \n {unanswered_questions}")

        return unanswered_questions

    def process_all_questions(self) -> None:
        """
        Processes all questions in the dataset, updating answers if a response is generated for any question.
        """
        for question in self._questions_json:
            is_response_generated, _ = self.process_one_question(question)
            if is_response_generated:
                self._update_answers()
                break
