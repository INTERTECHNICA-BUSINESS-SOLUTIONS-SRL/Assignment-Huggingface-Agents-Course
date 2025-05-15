# Copyright (c) Iuga Marin
# This file is part of the HuggingFace free AI Agents course assignment.
# It contains utility functions for web search, content retrieval, and analysis.

import json
import logging
import re
import time
from typing import List, Tuple

import markdownify
import requests
from fake_useragent import UserAgent

from langchain_community.tools import DuckDuckGoSearchResults
from langchain_tavily import TavilySearch

from setup import get_content_relevance_LLM
from setup import get_loose_content_analysis_LLM
from setup import get_query_optimization_LLM
from setup import get_strict_content_analysis_LLM


def get_optimized_web_query(query: str) -> str:
    """
    Optimizes a given web search query to improve relevance and effectiveness for web engine searches.
    Args:
        query (str): The initial web search query to be optimized.
    Returns:
        str: The optimized web search query.
    """

    query_optimization_llm = get_query_optimization_LLM()

    logging.debug(f"Query optimization tool is called.")
    logging.debug(f"Query: {query}]")

    prompt = f"""
    <role>
        You are an agent highly specialized in web query optimization.
        You take long and ambiguous queries and make them optimal for web engine search. 
        You focus on creating queries that can get the most relevant information.
        Avoid using quotes and logical operators.
        If you are asked for a count of items make sure you include this in the optimized query.
        If searching a website was mentioned, preserve the website search in the query using "site:" keyword.
    </role>
    <task>
        We will provide you with an initial web query.
        Avoid very narrow queries.
        Use only the content of the initial web query for optimization and nothing else. 
        Do not return anything besides the optimized query. 
    </task>
    <query>
        {query}
    </query>
    """

    result = query_optimization_llm.invoke(prompt)
    optimized_query = result.content
    logging.debug(f"Created optimized query: {optimized_query}")

    return optimized_query


def get_web_search_results_links_duckduckgo(query: str) -> str:
    """
    Searches the web based on a query and retrieves the search results page links.
    This tool uses DuckDuckGo search API.
    This can be used when information does not seem readily available.
    This can be used as a tool.

    Args:
        query: the query used for searching information on the web

    Returns:
        The search results list of search results page links.
    """
    ddg_tool = DuckDuckGoSearchResults(output_format="list")
    search_results = ddg_tool.invoke(query)

    search_results_links = []
    for search_result_item in search_results:
        search_results_links.append(search_result_item["link"])

    return search_results_links


def get_web_search_results_links_tavily(query: str) -> str:
    """
    Searches the web based on a query and retrieves the search results page links.
    This can be used when information does not seem readily available.
    This can be used as a tool.

    Args:
        query: the query used for searching information on the web

    Returns:
        The search results list of search results page links.
        """
    tavily_search_tool = TavilySearch(
        max_results=5,
        topic="general"
    )

    results = tavily_search_tool.invoke({"query": query})

    logging.debug(f"Obtained Tavily search results \n {results} \n")

    results_links = []
    results_scores = []
    for result in results["results"]:
        results_links.append(result["url"])
        results_scores.append(result["score"])

    logging.debug(f"Obtained Tavily search results links \n {results_links} \n")
    logging.debug(f"Obtained Tavily search results scores \n {results_scores} \n")

    return results_links, results_scores


def get_web_page_content(url: str) -> str:
    """
    Gets a WEB page content using an URL. 
    The content is transformed using markdown. 
    This can be used as a tool. 

    Args:
        url: the url to the page

    Returns:
        The content of the WEB page designated by the URL.
    """
    logging.debug(f"Get Web page content tools is called")
    logging.debug(f"URL: {url}")

    user_agent = UserAgent().firefox
    request_headers = {
        'user-agent': user_agent
    }

    logging.debug(f"Using user agent: {user_agent}")

    response = requests.get(url, headers=request_headers)
    response.raise_for_status()

    logging.debug(f"Content successfully retrieved.")

    html_content = response.content
    page_content = markdownify.markdownify(html_content, heading_style="ATX")

    logging.debug(f"Content successfully transformed to markdown.")

    return page_content


def compare_content_relevance(source_content: str, target_content: str, query: str) -> int:
    """
    Compares the source and target content for relevance towards a query.
    This can be used as a tool. 

    Args:
        source_content: the source content
        source_content: the source content

    Returns:
        0 if the source content is more relevant, otherwise 1
    """

    logging.debug(f"Content relevance tool is called.")
    logging.debug(f"Source content: \n {source_content[:500]} ...")
    logging.debug(f"Target content: \n {target_content[:500]} ...")
    logging.debug(f"Query: \n {query}")

    content_relevance_llm = get_content_relevance_LLM()

    content_relevance_prompt = f"""
    <role>
        You are an agent highly specialized in content comparison and content relevance analysis.
    </role>
    <task>
        You will be provided with a source content, a target content and a query.
        You will be asked to determine which content is more relevant to the query: the source content or the target content. 
        Determine the content relevance based on the information and its relevance for the query.
        Return the result exactly in the format we request. 
    </task>
    <source_content>
        {source_content}
    </source_content>
    <target_content>
        {target_content}
    </target_content>
    <query>
        {query}
    </query>
    <format>
        Return an integer which is 0 if the source content is more relevant, 1 if the target content is more relevant.
        Return just the integer and nothing else.
    </format>
    """

    relevance_raw_response = content_relevance_llm.invoke(content_relevance_prompt).content
    logging.debug(f"Retrieved content relevance raw response: \n {relevance_raw_response}")

    relevance_flag = int(relevance_raw_response)
    logging.debug(f"Obtained relevance flag: {relevance_flag}")

    return relevance_flag


def analyze_content_strict_mode(page_content: str, query: str) -> Tuple[float, str]:
    """
    Analyzes the provided page content in strict mode to answer a given query.
    This function uses a language model to evaluate the relevance and accuracy of the page content
    in response to the specified query. It returns a confidence score (between 0 and 1) indicating
    how well the content answers the query, along with the generated response.
    Args:
        page_content (str): The textual content of the page to be analyzed.
        query (str): The question or query to be answered based on the page content.
    Returns:
        Tuple[float, str]: A tuple containing:
            - confidence (float): A score between 0 and 1 representing the confidence in the response.
            - response (str): The answer to the query based strictly on the provided content.
    """

    logging.debug(f"Strict mode content analysis tool called.")

    content_analysis_prompt = f"""
                <task>
                    We will provide you with a page content and with a query which needs evaluated towards the page content.
                    You must respond to the query as exactly and precisely as possible.
                    You will be also asked to evaluate the confidence that your response provides a good answer to the query, do this as precisely as possible.
                    Use only the content provided and nothing else.
                    Explain your reasoning step by step.
                </task>
                <query>
                    {query}
                </query>
                <format>
                    Return the information as a JSON item
                    {{
                        "confidence": a number between 0 and 1, the more useful the answer is - the higher the answer. If you cannot respond using the content, set the confidence to 0.
                        "response": the response returned,
                        "reasoning": a summarization of the reasoning for determining the response and confidence
                    }}
                    Always make sure that you return a valid JSON item.
                    Return only the JSON item and nothing else beside the JSON item. This is important.
                </format>
                <page_content>
                    {page_content}
                </page_content>
            """

    content_analysis_LLM = get_strict_content_analysis_LLM()

    analysis_content = content_analysis_LLM.invoke(content_analysis_prompt).content
    logging.debug(f"We have obtained the following raw analysis content: \n{analysis_content}\n")

    pattern = r'^```json\s*(.*?)\s*```$'
    cleaned_analysis_content = analysis_content.replace("\n", "")
    cleaned_analysis_content = re.sub(pattern, r'\1', cleaned_analysis_content, flags=re.DOTALL)

    json_content = json.loads(cleaned_analysis_content)
    logging.debug(f"We have obtained the following cleaned analysis content: \n{json_content}\n")

    response = json_content["response"]
    confidence = float(json_content["confidence"])
    reasoning = json_content["reasoning"]

    logging.debug(f"The response is {response}")
    logging.debug(f"The confidence is {confidence}")
    logging.debug(f"The reasoning is {reasoning}")

    return confidence, response


def analyze_content_loose_mode(page_content: str, query: str) -> Tuple[float, str]:
    """
    Analyzes the provided page content in relation to a given query using a language model in 'loose mode'.
    This function evaluates how well the page content can answer the specified query, inferring answers when necessary.
    It returns both a confidence score (between 0 and 1) indicating the reliability of the response, and the response itself.
    Args:
        page_content (str): The textual content of the page to be analyzed.
        query (str): The question or query to be evaluated against the page content.
    Returns:
        Tuple[float, str]: A tuple containing:
            - confidence (float): A score between 0 and 1 representing the confidence in the response.
            - response (str): The answer generated based on the page content and query.
    """

    logging.debug(f"Loose mode content analysis tool called.")

    content_analysis_prompt = f"""
                <task>
                    We will provide you with a page content and with a query which needs evaluated towards the page content.
                    You must respond to the query using the page content provided along with your best judgement.
                    If the answer is not immediately obvious, infer the possible answer using your judgement.
                    You will be also asked to evaluate the confidence that your response provides a good answer to the query, do this as precisely as possible.
                    If the answer is not obvious from the content, use a lower confidence.
                    If the answer is not from the content, use a higher confidence.                    
                    Use only the content provided and nothing else.
                    Explain your reasoning step by step.
                </task>
                <query>
                    {query}
                </query>
                <format>
                    Return the information as a JSON item
                    {{
                        "confidence": a number between 0 and 1, the more useful the answer is - the higher the answer. If you cannot respond using the content, set the confidence to 0.
                        "response": the response returned,
                        "reasoning": a summarization of the reasoning for determining the response and confidence
                    }}
                    Always make sure that you return a valid JSON item.
                    Return only the JSON item and nothing else beside the JSON item. This is important.
                </format>
                <page_content>
                    {page_content}
                </page_content>
            """

    content_analysis_LLM = get_loose_content_analysis_LLM()

    analysis_content = content_analysis_LLM.invoke(content_analysis_prompt).content
    logging.debug(f"We have obtained the following raw analysis content: \n{analysis_content}\n")

    pattern = r'^```json\s*(.*?)\s*```$'
    cleaned_analysis_content = analysis_content.replace("\n", "")
    cleaned_analysis_content = re.sub(pattern, r'\1', cleaned_analysis_content, flags=re.DOTALL)

    json_content = json.loads(cleaned_analysis_content)
    logging.debug(f"We have obtained the following cleaned analysis content: \n{json_content}\n")

    response = json_content["response"]
    confidence = float(json_content["confidence"])
    reasoning = json_content["reasoning"]

    logging.debug(f"The response is {response}")
    logging.debug(f"The confidence is {confidence}")
    logging.debug(f"The reasoning is {reasoning}")

    return confidence, response


def process_results_url_links(url_links: List[str], url_scores: List[float], query: str) -> str:
    """
    Processes a list of URL links and their associated scores to find the most relevant response to a given query.
    This function iterates through the provided URLs, analyzes the content of each page using different content analysis modes,
    and selects the response with the highest confidence score that is relevant to the query. If no sufficiently relevant response
    is found, a generic message is returned.
    Args:
        url_links (List[str]): A list of URLs to be processed.
        url_scores (List[float]): A list of scores corresponding to the relevance or quality of each URL.
        query (str): The query string to find relevant information for.
    Returns:
        str: The most relevant response found based on the query and URL content, or a generic message if no relevant answer is found.
    """
    last_meaningful_response_confidence = -1
    last_meaningful_response = None
    last_meaningful_page_content = None

    logging.debug(f"Processing URL links tool called.")
    logging.debug(f"URL links: \n{url_links}\n")
    logging.debug(f"URL scores: \n{url_scores}\n")
    logging.debug(f"Query: \n{query}\n")

    for analyze_content_mode in [analyze_content_strict_mode, analyze_content_loose_mode]:
        logging.debug(f"Processing URL links using analyze content mode: {analyze_content_mode.__name__}")

        for index in range(len(url_links)):
            try:
                url_link = url_links[index]
                url_score = url_scores[index]

                logging.debug(f"Current processed link: {url_link}")
                logging.debug(f"Current processed score: {url_score}")

                page_content = get_web_page_content(url_link)

                current_confidence, current_response = analyze_content_mode(page_content, query)

                if current_confidence > 0 and current_confidence >= last_meaningful_response_confidence:
                    content_relevance_flag = 1
                    if current_confidence == last_meaningful_response_confidence:
                        # force relevance comparison
                        content_relevance_flag = compare_content_relevance(
                            last_meaningful_page_content, page_content, query)

                    if content_relevance_flag == 1:
                        last_meaningful_response_confidence = current_confidence
                        last_meaningful_response = current_response
                        last_meaningful_page_content = page_content

                        logging.debug(
                            f"Using new meaningful response: \n{last_meaningful_response}\n with confidence {current_confidence}")
                    else:
                        logging.debug(f"Content relevance is low and will be skipped.")
                else:
                    logging.debug(f"Response confidence is low and will be skipped.")

                # !!!! Sleep in order to avoid quota issues
                logging.debug(f"Sleeping for preventing quota usage.")
                time.sleep(20)
            except Exception as e:
                logging.error(f"""
                    Failed to analyze the content of the web page:              
                    URL: {url_link},
                    Score: {url_score},
                    query: {query},
                    exception: {str(e)}
                    \n
                """)

                continue

        if (last_meaningful_response is not None) and (last_meaningful_response_confidence > 0.33):
            logging.debug(
                f"Found meaningful response with confidence {last_meaningful_response_confidence} using analyze content mode: {analyze_content_mode.__name__}")
            logging.debug(f"Found meaningful response: \n{last_meaningful_response}\n")

            # if the response is found, we can stop the processing
            return last_meaningful_response

    if last_meaningful_response is None:
        logging.warning(
            f"No relevant answer has been found while processing the URL links. We will use a generic no results answer.")
        last_meaningful_response = "No results have been found, the processing has failed."

    return last_meaningful_response


def search_web(query: str = None) -> str:
    """
    Searches WEB for for information using a query search string. 
    Use the tool only once per session, do not call it multiple times, do not call it recursively.
    To be used if you cannot figure out the answer by other means.
    This can be used as a tool. 

    Args:
        query: the query used to search the web, use exactly the text of the task you are performing, do not generate a new query.
        If searching a website was specified, restrict query to the website.
        Always provide the query.

    Returns:
        The best result obtained by analyzing the results retrieved from the web. Do not search the web further with this result.     
    """
    if query is None:
        return None

    logging.debug(f"Received request to search with the query: {query}]")
    optimized_query = get_optimized_web_query(query)
    logging.debug(f"Searching with optimized query: {optimized_query}]")

    url_links, url_scores = get_web_search_results_links_tavily(optimized_query)
    content = process_results_url_links(url_links, url_scores,  query)
    logging.debug(f"Received search response: {content}]")

    return content


def search_web_natural_language(query: str = None) -> str:
    """
    Searches the web in regards with a topic. 
    IF you don't understand a question, use this to clarify it (for example if it is missing information).
    The question must be expressed briefly in natural language.
    This can be used as a tool. 

    Args:
        query: the query used to search the web, must be in natural language, concise and precise. Do not use web search queries.
        Always provide the query.

    Returns:
        The best result obtained from the knowledge base. Do not search the knowledge base further with this result.     
    """
    if query is None:
        return None

    logging.debug(f"Received request to search web in natural language with the query: {query}]")
    optimized_query = get_optimized_web_query(query)
    logging.debug(f"Searching with optimized query: {optimized_query}]")

    url_links, url_scores = get_web_search_results_links_tavily(optimized_query)
    content = process_results_url_links(url_links, url_scores,  query)
    logging.debug(f"Received knowledge base search response: {content}]")

    return content
