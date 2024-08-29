from typing import Annotated, List

from langchain_community.document_loaders import WebBaseLoader
#from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.tools import tool
import requests
from bs4 import BeautifulSoup 
import lxml
import json
import getpass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Dict, Optional
import os

from langchain_experimental.utilities import PythonREPL
from typing_extensions import TypedDict

from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
import chromadb
from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter
from langchain_openai.chat_models import ChatOpenAI


from typing import List, Optional

from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain.output_parsers.openai_functions import JsonOutputFunctionsParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage, HumanMessage
from langchain.pydantic_v1 import BaseModel, Field

from langgraph.graph import END, StateGraph, START
from langchain.schema.output_parser import StrOutputParser


WORKING_DIRECTORY = Path(os.environ["WORKING_DIRECTORY"])

class SearchInput(BaseModel):
    query: str = Field(description="should be a search query")

@tool
def search_on_web(question: Annotated[str, "The user question to be searced in the index."],
                   page_num = 0, page_limit=10, language="en", country="br",
                    headers={"User-Agent": "Mozilla/5.0 (Windows NT 6.1; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/108.0.0.0 Safari/537.36"}):
    
    """Use this to search the internet to gete more information about any question. The information might be wrong"""
    
    params = {
    "q" : question #+ ' do Itaú Unibanco'
        ,
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

def vector_store_init(persist_directory: str = "data",
                        collection_name: str = "gdp",
                        doc: str = "content.txt",
                     append: bool = False):
    
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
    if append:
        # Create a new Chroma database from the documents
        chroma_db = Chroma.from_documents(
            documents=docs, 
            embedding=embeddings, 
            persist_directory=persist_directory,
            collection_name=collection_name
        )
    
        # Save the Chroma database to disk
        chroma_db.persist()
    return chroma_db

# Define the custom search tool
@tool("search-tool", args_schema=SearchInput, return_direct=False)
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

def create_agent(
    llm: ChatOpenAI,
    tools: list,
    system_prompt: str,
) -> str:
    if len(tools) > 0:
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
    else:
        system_prompt += "\nWork autonomously according to your specialty"
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
            ]
        )
    
        return prompt | llm | StrOutputParser() | parse_output
    
def parse_output(output):
    return {"output": output}


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