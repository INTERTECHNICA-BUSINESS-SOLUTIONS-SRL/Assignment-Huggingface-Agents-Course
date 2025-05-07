# Copyright (c) Iuga Marin
# This file is part of the HuggingFace free AI Agents course assignment.
# It contains the implementation of an AI agent with basic tooling capabilities.

from inspect import signature
from typing import Annotated, Optional, TypedDict

from langchain_core.messages import AnyMessage, HumanMessage, SystemMessage
from langgraph.graph import START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

from setup import get_baseline_LLM

from tools_arithmetic import add_values
from tools_audio import get_transcribed_audio_file_data
from tools_excel import get_excel_file_data
from tools_image import get_requested_information_from_image
from tools_python import get_python_file_data
from tools_web import search_knowledge_base, search_web
from tools_youtube import get_analysis_information_from_youtube_video


class AgentState(TypedDict):
    """
    Represents the state of the agent.
    Attributes:
        input_file (Optional[str]): Contains the input file, if provided.
        messages (Annotated[list[AnyMessage], add_messages]): Contains the messages exchanged with the agent.
    """
    input_file: Optional[str]
    messages: Annotated[list[AnyMessage], add_messages]


def get_tools():
    """
    Retrieves a list of tool functions that can be used for various operations.
    Returns:
        list: A list of callable tool functions.
    """
    tools = [
        add_values,
        get_excel_file_data,
        get_python_file_data,
        get_excel_file_data,
        get_python_file_data,
        get_transcribed_audio_file_data,
        get_requested_information_from_image,
        get_analysis_information_from_youtube_video,
        search_knowledge_base
    ]

    return tools


def get_tool_description(tool):
    """
    Generate a formatted description of a given tool function.
    This function extracts the name, signature, and docstring of the provided tool
    function and combines them into a single formatted string.
    Args:
        tool (Callable): The tool function to describe. It must be a callable object.
    Returns:
        str: A formatted string containing the tool's name, signature, and docstring.
    """
    tool_name = tool.__name__
    tool_signature = signature(tool)
    tool_doc = tool.__doc__.strip("\n").strip("\n")

    tool_description = f"{tool_name}{tool_signature}\n{tool_doc}"

    return tool_description


def get_tools_description() -> str:
    """
    Retrieves a description of all available tools.
    Returns:
        str: A string containing descriptions of all tools, separated by newlines.
    """
    tools = get_tools()
    tools_description = ""
    for tool in tools:
        tools_description = tools_description + "\n" + get_tool_description(tool)

    return tools_description


def create_tooling_LLM():
    """
    Creates and returns a tooling-enabled language model (LLM) by binding tools
    to a baseline LLM.
    Returns:
        An instance of a tooling-enabled LLM.
    """
    baseline_LLM = get_baseline_LLM()
    tooling_LLM = baseline_LLM.bind_tools(get_tools())

    return tooling_LLM


def assistant(state: AgentState) -> AgentState:
    """
    Processes the given agent state to analyze input files and execute tasks using available tools.
    Args:
        state (AgentState): A dictionary containing the current state of the agent, including:
            - "input_file": The file to be analyzed (can be None if no file is provided).
            - "messages": A list of messages representing the conversation history.
    Returns:
        dict: A dictionary containing:
            - "messages": A list of processed messages, including the system's response.
            - "input_file": The input file provided in the state.
    """
    input_file = state["input_file"]
    if input_file is None:
        input_file = "No input file was provided."

    sys_msg = SystemMessage(
        content=f"""
            <role>
                You are a very capable AI Agent used for complex task specified in natural language.
                You can analyse documents and run computations with provided tools:
            <role>
            <tools>
                You are provided with the following tools:
---
{get_tools_description()}.
---
                You will call any tools as many times as needed in order to fulfill a requested task.
                Always prioritize the available tools over your calculations and reasoning.
                If an input file is provided, analyze the input file using the best tool provided.
                If you cannot analyze the file with the specified tool, answer that you cannot process the task.
                Use the tools outcome to assemble the final answer.
            </tools>
            <input_file>
                {input_file}
            </input_file>
            <final_answer>
                It is very important that the final answer respects in details all the initial requirements you were provided.
                Double check for answer format amd answer fit to the initial requirements.
            </final_answer>
        """)

    tooling_llm = create_tooling_LLM()

    return {
        "messages": [tooling_llm.invoke([sys_msg] + state["messages"])],
        "input_file": state["input_file"]
    }


class AgentBasicTooling():
    """
    A class that provides basic tooling for interacting with a REACT-based graph.
    This class is designed to handle queries and optionally process input files
    by invoking a REACT graph and returning the response.
    Methods
    -------
    __init__():
        Initializes the REACT graph for processing queries.
    __call__(query: str, input_file: str = None) -> str:
        Executes a query against the REACT graph and returns the response.
    """

    def _create_REACT_graph(self) -> StateGraph:
        """
        Creates and compiles a REACT (Reasoning + Acting) state graph for an agent.
        The graph defines the flow of interaction between the assistant and tools,
        enabling the agent to decide whether to respond directly or utilize tools
        based on the context of the latest message.
        Returns:
            StateGraph: A compiled state graph representing the REACT workflow.
        """
        builder = StateGraph(AgentState)

        # Add nodes for assistant logic and tools.
        builder.add_node("assistant", assistant)
        builder.add_node("tools", ToolNode(get_tools()))

        # Define graph flow: start -> assistant -> tools (if needed) -> assistant.
        builder.add_edge(START, "assistant")
        builder.add_conditional_edges("assistant", tools_condition)
        builder.add_edge("tools", "assistant")

        # Compile and return the state graph.
        return builder.compile()

    def __init__(self):
        """Initializes the instance and sets up the REACT graph."""
        self._react_graph = self._create_REACT_graph()

    def __call__(self, query: str, input_file: str = None) -> str:
        """
        Executes the callable object with the given query and optional input file.
        This method processes a query by creating a list of messages, invoking the 
        `_react_graph` with the provided messages and optional input file, and 
        returning the response messages and the content of the last response message.
        Args:
            query (str): The input query string to be processed.
            input_file (str, optional): An optional file path to provide additional input. 
                Defaults to None.
        Returns:
            tuple: A tuple containing:
                - response_messages (dict): The full response messages from `_react_graph`.
                - response_content (str): The content of the last message in the response.
        """
        messages = [HumanMessage(content=query)]
        response_messages = self._react_graph.invoke({"messages": messages, "input_file": input_file})
        response_content = response_messages["messages"][-1].content

        return response_messages, response_content
