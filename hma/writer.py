#import operator
#from pathlib import Path
#from hma.utils import *
#import functools
#
#
#WORKING_DIRECTORY = Path(os.environ["WORKING_DIRECTORY"])
#
#
#class Writer():
#    def __init__(self):
#        self.llm = ChatOpenAI(model=os.environ["MODEL"])
#        
#        self.doc_writer_agent = create_agent(
#            self.llm,
#            [write_document, edit_document, read_document],
#            "You are an expert writing a customer service a detailed and complete call guide or standard operation procedure. All content you write must be based on information brought on the message. Don't makeup information, if you don't have enought information to write about, return 'Need more context'\n"
#            # The {current_files} value is populated automatically by the graph state
#            "Below are files currently in your directory:\n{current_files}",
#        )
#        
#        self.context_aware_doc_writer_agent = prelude | self.doc_writer_agent
#        self.doc_writing_node = functools.partial(
#            agent_node, agent=self.context_aware_doc_writer_agent, name="DocWriter"
#        )
#        
#        self.note_taking_agent = create_agent(
#            self.llm,
#            [create_outline, read_document],
#            "You are an expert senior expert tasked with writing a customer service script outline and"
#            " taking notes to craft a perfect customer service script. All content you write must be based on information brought on the message. Don't makeup information, if you don't have enought information to write about, return 'Need more context'{current_files}",
#        )
#        
#        self.context_aware_note_taking_agent = prelude | self.note_taking_agent
#        self.note_taking_node = functools.partial(
#            agent_node, agent=self.context_aware_note_taking_agent, name="NoteTaker"
#        )
#        
#        
##        self.chart_generating_agent = create_agent(
##            self.llm,
##            [read_document, python_repl],
##            "You are a data viz expert tasked with generating charts for a research project."
##            "{current_files}",
##        )
#        
##        self.context_aware_chart_generating_agent = prelude | self.chart_generating_agent
##        self.chart_generating_node = functools.partial(
##            agent_node, agent=self.context_aware_note_taking_agent, name="ChartGenerator"
##        )
#        
#        self.doc_writing_supervisor = create_team_supervisor(
#            self.llm,
#            "You are a supervisor tasked with managing a conversation between the"
#            " following workers:  {team_members}. Given the following user request,"
#            " respond with the worker to act next. Each worker will perform a"
#            " task and respond with their results and status. When finished,"
#            " respond with FINISH.",
#            ["DocWriter", "NoteTaker"#, "ChartGenerator"
#            ],
#        )
#        
#        # Create the graph here:
#        # Note that we have unrolled the loop for the sake of this doc
#        self.authoring_graph = StateGraph(DocWritingState)
#        self.authoring_graph.add_node("DocWriter", self.doc_writing_node)
#        self.authoring_graph.add_node("NoteTaker", self.note_taking_node)
##        self.authoring_graph.add_node("ChartGenerator", self.chart_generating_node)
#        self.authoring_graph.add_node("supervisor", self.doc_writing_supervisor)
#        
#        # Add the edges that always occur
#        self.authoring_graph.add_edge("DocWriter", "supervisor")
#        self.authoring_graph.add_edge("NoteTaker", "supervisor")
##        self.authoring_graph.add_edge("ChartGenerator", "supervisor")
#        
#        # Add the edges where routing applies
#        self.authoring_graph.add_conditional_edges(
#            "supervisor",
#            lambda x: x["next"],
#            {
#                "DocWriter": "DocWriter",
#                "NoteTaker": "NoteTaker",
##                "ChartGenerator": "ChartGenerator",
#                "FINISH": END,
#            },
#        )
#        
#        self.authoring_graph.add_edge(START, "supervisor")
#        self.chain = self.authoring_graph.compile()
#        
#        # We reuse the enter/exit functions to wrap the graph
#        self.authoring_chain = (
#            functools.partial(enter_chain, members=self.authoring_graph.nodes)
#            | self.authoring_graph.compile()
#        )
#        
#
## The following functions interoperate between the top level graph state
## and the state of the research sub-graph
## this makes it so that the states of each graph don't get intermixed
#def enter_chain(message: str, members: List[str]):
#    results = {
#        "messages": [HumanMessage(content=message)],
#        "team_members": ", ".join(members),
#    }
#    return results
#
#
#
#        
#        
## This will be run before each worker agent begins work
## It makes it so they are more aware of the current state
## of the working directory.
#def prelude(state):
#    written_files = []
#    if not WORKING_DIRECTORY.exists():
#        WORKING_DIRECTORY.mkdir()
#    try:
#        written_files = [
#            f.relative_to(WORKING_DIRECTORY) for f in WORKING_DIRECTORY.rglob("*")
#        ]
#    except Exception:
#        pass
#    if not written_files:
#        return {**state, "current_files": "No files written."}
#    return {
#        **state,
#        "current_files": "\nBelow are files your team has written to the directory:\n"
#        + "\n".join([f" - {f}" for f in written_files]),
#    }
#
## Document writing team graph state
#class DocWritingState(TypedDict):
#    # This tracks the team's conversation internally
#    messages: Annotated[List[BaseMessage], operator.add]
#    # This provides each worker with context on the others' skill sets
#    team_members: str
#    # This is how the supervisor tells langgraph who to work next
#    next: str
#    # This tracks the shared directory state
#    current_files: str

import operator
from pathlib import Path
from hma.utils import *
import functools

WORKING_DIRECTORY = Path(os.environ["WORKING_DIRECTORY"])

class Writer():
    def __init__(self):
        self.llm = ChatOpenAI(model=os.environ["MODEL"])

        # Define the NoteTaker agent
        self.note_taking_agent = create_agent(
            self.llm,
            [create_outline, read_document],
            "You are an expert senior tasked with writing a customer service script outline and"
            " taking notes to craft a perfect customer service script. All content you write must be based on information brought on the message. Don't make up information. If you don't have enough information to write about, return 'Need more context'. Follow the outline document as a guide. \n{current_files}",
        )

        self.context_aware_note_taking_agent = prelude | self.note_taking_agent
        self.note_taking_node = functools.partial(
            agent_node, agent=self.context_aware_note_taking_agent, name="NoteTaker"
        )

        # Define the DocWriter agent
        self.doc_writer_agent = create_agent(
            self.llm,
            [write_document, edit_document, read_document],
            "You are an expert writing a customer service a detailed and complete call guide or standard operation procedure. All content you write must be based on information brought on the message. Don't make up information. If you don't have enough information to write about, return 'Need more context'\n"
            "Below are files currently in your directory:\n{current_files}",
        )

        self.context_aware_doc_writer_agent = prelude | self.doc_writer_agent
        self.doc_writing_node = functools.partial(
            agent_node, agent=self.context_aware_doc_writer_agent, name="DocWriter"
        )

        # Define the supervisor
        self.doc_writing_supervisor = create_team_supervisor(
            self.llm,
            "You are a supervisor tasked with managing a conversation between the"
            " following workers: {team_members}. Given the following user request,"
            " respond with the worker to act next. Each worker will perform a"
            " task and respond with their results and status. When finished,"
            " respond with FINISH.",
            ["NoteTaker", "DocWriter"],
        )

        # Create the state graph
        self.authoring_graph = StateGraph(DocWritingState)
        self.authoring_graph.add_node("NoteTaker", self.note_taking_node)
        self.authoring_graph.add_node("DocWriter", self.doc_writing_node)
        self.authoring_graph.add_node("supervisor", self.doc_writing_supervisor)

        # Define the sequence: NoteTaker runs first, then DocWriter
        self.authoring_graph.add_edge("NoteTaker", "supervisor")
        self.authoring_graph.add_edge("DocWriter", "supervisor")

        # Routing logic to ensure NoteTaker runs first, then DocWriter
        self.authoring_graph.add_conditional_edges(
            "supervisor",
            lambda state: "DocWriter" if state["next"] == "DocWriter" else "FINISH",
            {
                "DocWriter": "DocWriter",
                "FINISH": END,
            },
        )

        self.authoring_graph.add_edge(START, "NoteTaker")
        self.chain = self.authoring_graph.compile()

        # Wrap the graph
        self.authoring_chain = (
            functools.partial(enter_chain, members=self.authoring_graph.nodes)
            | self.authoring_graph.compile()
        )

# Functions to manage the flow and state
def enter_chain(message: str, members: List[str]):
    results = {
        "messages": [HumanMessage(content=message)],
        "team_members": ", ".join(members),
    }
    return results

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

# Document writing team graph state
class DocWritingState(TypedDict):
    messages: Annotated[List[BaseMessage], operator.add]
    team_members: str
    next: str
    current_files: str
