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
            "You are an expert writing a customer service a detailed and complete call guide or standard operation procedure in a way that an attendant must be able to answer all the customer questions using only this guide, include everything he may need in this guide. All content you write must be based on information brought on the message. Don't make up information. If you don't have enough information to write about, return 'Need more context'\n"
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
