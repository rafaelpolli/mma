from langchain_core.runnables import RunnablePassthrough
from langchain_openai.chat_models import ChatOpenAI
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage 
import functools
import operator 
from langchain.agents import AgentExecutor, create_openai_functions_agent
from bs4 import BeautifulSoup 
import lxml
import json 
import requests 
from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder 
from typing import Any, Callable, List, Optional, TypedDict, Union, Annotated, Dict 
from langchain_core.runnables import ( RunnableSerializable, Runnable, RunnableConfig, ensure_config) 
from langchain.output_parsers.openai_functions import JsonOutputFunctionsParser
from langgraph.graph import END, StateGraph, START
from langchain.vectorstores import Chroma
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter
import os
import chromadb

from pathlib import Path
from tempfile import TemporaryDirectory


from langchain_experimental.utilities import PythonREPL
from typing_extensions import TypedDict

_TEMP_DIRECTORY = TemporaryDirectory()
WORKING_DIRECTORY = Path(_TEMP_DIRECTORY.name)

WORKING_DIRECTORY = Path('/home/studio-lab-user/multiagent/files')

def vector_store_init(persist_directory: str = "data",
                        collection_name: str = "gdp",
                        doc: str = "content.txt"):
    
    os.environ["PERSIST_DIRECTORY"] = persist_directory
    os.environ["COLLECTION_NAME"] = collection_name
    
    embeddings = OpenAIEmbeddings()
    # Load the Chroma database from disk
    chroma_db = Chroma(persist_directory=persist_directory, 
                       embedding_function=embeddings,
                       collection_name=collection_name)
    
    # Get the collection from the Chroma database
    collection = chroma_db.get(collection_name)
    
    with open(doc) as f:
        content = f.read()
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    pages = text_splitter.split_text(content)
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    docs = text_splitter.create_documents(pages)
    
    # If the collection is empty, create a new one
    if len(collection['ids']) == 0:
        # Create a new Chroma database from the documents
        chroma_db = Chroma.from_documents(
            documents=docs, 
            embedding=embeddings, 
            persist_directory=persist_directory,
            collection_name=collection_name
        )
    
        # Save the Chroma database to disk
        chroma_db.persist()
        

# Define the custom search tool
@tool
def chromadb_search(query: str) -> list:
    """
    Perform a search in the ChromaDB collection using OpenAI embeddings.

    Args:
        query (str): The search query.

    Returns:
        list: A list of search results.
    """
    # Initialize your embedding function
    embeddings = OpenAIEmbeddings()
  
    # Load the Chroma database from disk
    chroma_db = Chroma(persist_directory=os.environ["PERSIST_DIRECTORY"], embedding_function=embeddings, collection_name = os.environ["COLLECTION_NAME"])
    # Convert the query to an embedding using the OpenAIEmbeddings instance
    
    # Perform the search using embeddings within the specified collection
    results = chroma_db.similarity_search(query, k = 4)

    # Process and return the results
    return results


def create_agent(
    llm: ChatOpenAI,
    tools: list,
    system_prompt: str,
) -> str:
    """Create a function-calling agent and add it to the graph."""
    system_prompt += "\nWork autonomously according to your specialty, using the tools available to you."
    " Do not ask for clarification."
    " Your other team members (and other teams) will collaborate with you with their own specialties."
    " You are chosen for a reason! You are one of the following team members: {team_members}."
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                system_prompt,
            ),
            MessagesPlaceholder(variable_name="messages"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ]
    )
    agent = create_openai_functions_agent(llm, tools, prompt)
    executor = AgentExecutor(agent=agent, tools=tools)
    return executor

def agent_node(state, agent, name):
    result = agent.invoke(state)
    return {"messages": [HumanMessage(content=result["output"], name=name)]}


def agent_node(state, agent, name):
    result = agent.invoke(state)
    return {"messages": [HumanMessage(content=result["output"], name=name)]}



def create_team_supervisor(llm: ChatOpenAI, system_prompt, members) -> str:
    """An LLM-based router."""
    options = ["FINISH"] + members
    function_def = {
        "name": "route",
        "description": "Select the next role.",
        "parameters": {
            "title": "routeSchema",
            "type": "object",
            "properties": {
                "next": {
                    "title": "Next",
                    "anyOf": [
                        {"enum": options},
                    ],
                },
            },
            "required": ["next"],
        },
    }
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            MessagesPlaceholder(variable_name="messages"),
            (
                "system",
                "Given the conversation above, who should act next?"
                " Or should we FINISH? Select one of: {options}",
            ),
        ]
    ).partial(options=str(options), team_members=", ".join(members))
    return (
        prompt
        | llm.bind_functions(functions=[function_def], function_call="route")
        | JsonOutputFunctionsParser()
    )




class ResearchTeamState(TypedDict):
    # A message is added after each team member finishes
    messages: Annotated[List[BaseMessage], operator.add]
    # The team members are tracked so they are aware of
    # the others' skill-sets
    team_members: List[str]
    # Used to route work. The supervisor calls a function
    # that will update this every time it makes a decision
    next: str

    
@tool
def query_on_google(question: Annotated[str, "The user question to be searced in the index."],
                   page_num = 0, page_limit=10, language="en", country="br",
                    headers={"User-Agent": "Mozilla/5.0 (Windows NT 6.1; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/108.0.0.0 Safari/537.36"}):
    
    """Use this to search the internet to gete more information about any question. The information might be wrong"""
    
    params = {
    "q" : question + ' do Itaú Unibanco',
    "h1" : language,
    "g1" : country,
    "start" : 0
    }
    
    data = []
    
    while True:
        page_num += 1
        
        khtml = requests.get("http://www.google.com/search",
                            params=params, headers = headers, timeout=30)
        soup = BeautifulSoup(khtml.text, 'lxml')
        
        for result in soup.select(".tF2Cxc"):
            title = result.select_one(".DKV0Md").text
            try:
                snippet = result.select_one(".lEBKkf span").text
            except:
                snippet = None
            links = result.select_one(".yuRUbf a")["href"]
            
            data.append({
            "title": title,
            "snippet": snippet,
            "links": links
            })
            
            
        if page_num == page_limit:
            break
        if soup.select_one(".d6cvqb a[id=pnnext]"):
            params["start"] += 10
        else:
            break
    
    return json.dumps(data, indent=2, ensure_ascii=False)   



def enter_chain(message: str):
    results = {
        "messages": [HumanMessage(content=message)],
    }
    return results

@tool
def create_outline(
    points: Annotated[List[str], "List of main points or sections."],
    file_name: Annotated[str, "File path to save the outline."],
) -> Annotated[str, "Path of the saved outline file."]:
    """Create and save an outline."""
    with (WORKING_DIRECTORY / file_name).open("w") as file:
        for i, point in enumerate(points):
            file.write(f"{i + 1}. {point}\n")
    return f"Outline saved to {file_name}"


@tool
def read_document(
    file_name: Annotated[str, "File path to save the document."],
    start: Annotated[Optional[int], "The start line. Default is 0"] = None,
    end: Annotated[Optional[int], "The end line. Default is None"] = None,
) -> str:
    """Read the specified document."""
    with (WORKING_DIRECTORY / file_name).open("r") as file:
        lines = file.readlines()
    if start is not None:
        start = 0
    return "\n".join(lines[start:end])


@tool
def write_document(
    content: Annotated[str, "Text content to be written into the document."],
    file_name: Annotated[str, "File path to save the document."],
) -> Annotated[str, "Path of the saved document file."]:
    """Create and save a text document."""
    with (WORKING_DIRECTORY / file_name).open("w") as file:
        file.write(content)
    return f"Document saved to {file_name}"


@tool
def edit_document(
    file_name: Annotated[str, "Path of the document to be edited."],
    inserts: Annotated[
        Dict[int, str],
        "Dictionary where key is the line number (1-indexed) and value is the text to be inserted at that line.",
    ],
) -> Annotated[str, "Path of the edited document file."]:
    """Edit a document by inserting text at specific line numbers."""

    with (WORKING_DIRECTORY / file_name).open("r") as file:
        lines = file.readlines()

    sorted_inserts = sorted(inserts.items())

    for line_number, text in sorted_inserts:
        if 1 <= line_number <= len(lines) + 1:
            lines.insert(line_number - 1, text + "\n")
        else:
            return f"Error: Line number {line_number} is out of range."

    with (WORKING_DIRECTORY / file_name).open("w") as file:
        file.writelines(lines)

    return f"Document edited and saved to {file_name}"


# Warning: This executes code locally, which can be unsafe when not sandboxed

repl = PythonREPL()


@tool
def python_repl(
    code: Annotated[str, "The python code to execute to generate your chart."],
):
    """Use this to execute python code. If you want to see the output of a value,
    you should print it out with `print(...)`. This is visible to the user."""
    try:
        result = repl.run(code)
    except BaseException as e:
        return f"Failed to execute. Error: {repr(e)}"
    return f"Successfully executed:\n```python\n{code}\n```\nStdout: {result}"



# Document writing team graph state
class DocWritingState(TypedDict):
    # This tracks the team's conversation internally
    messages: Annotated[List[BaseMessage], operator.add]
    # This provides each worker with context on the others' skill sets
    team_members: str
    # This is how the supervisor tells langgraph who to work next
    next: str
    # This tracks the shared directory state
    current_files: str


# This will be run before each worker agent begins work
# It makes it so they are more aware of the current state
# of the working directory.
def prelude(state):
    written_files = []
    if not WORKING_DIRECTORY.exists():
        WORKING_DIRECTORY.mkdir()
    try:
        written_files = [
            f.relative_to(WORKING_DIRECTORY) for f in WORKING_DIRECTORY.rglob("*")
        ]
    except Exception:
        pass
    if not written_files:
        return {**state, "current_files": "No files written."}
    return {
        **state,
        "current_files": "\nBelow are files your team has written to the directory:\n"
        + "\n".join([f" - {f}" for f in written_files]),
    }


# Top-level graph state
class State(TypedDict):
    messages: Annotated[List[BaseMessage], operator.add]
    next: str


def get_last_message(state: State) -> str:
    return state["messages"][-1].content


def join_graph(response: dict):
    return {"messages": [response["messages"][-1]]}













def get_clarify_answer(llm: ChatOpenAI):
    prompt_clarify = """You are an evaluator that provides answer to questions and make sure you are answering Exactly
    {input}"""
    
    clarify_prompt = ChatPromptTemplate.from_template(prompt_clarify)
    
    def output_message(output):
        return {'messages' : [AIMessage(output.content, name = 'ClarifyVerifier')]}
    
    clarify_chain = {'input' : RunnablePassthrough()} | clarify_prompt | llm | output_message
    return clarify_chain

def get_supervised_team(llm: ChatOpenAI):
    
    junior_node = create_node(llm, 
                             [chromadb_search],
                             "You're a junior analyst who can answer simple questions"
                             "JuniorAnalyst")
    
    senior_node = create_node(llm, 
                             [chromadb_search],
                             "You're a senior analyst who can answer more complex questions"
                             "SeniorAnalyst")
    
    nodes = [junior_node, senior_node]
    chain = SupervisorTeamGraph(nodes = nodes, llm = llm)
    subject_specs_chain = enter_chain | chain
    return subject_specs_chain