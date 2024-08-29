import operator
from pathlib import Path
from hma.utils import *
import functools


WORKING_DIRECTORY = Path(os.environ["WORKING_DIRECTORY"])

class Validator():
    def __init__(self):
        self.llm = ChatOpenAI(model=os.environ["MODEL"])

        self.customer_agent = create_agent(
            self.llm,
            [],
            "Given the input message, you must act as a customer who is calling a service center. Your role is to gerate customer questions about the topic. Interact with the attendant."
        )
        self.customer_node = functools.partial(agent_node, agent=self.customer_agent, name="Customer")

        # Define the DocWriter agent
        self.attendant_agent = create_agent(
            self.llm,
            [read_document],
            "You are a call center attendant who must follow only the script and respond to the customer based only on the provided document. \n{current_files}"
        )

        self.context_aware_attendant_agent = prelude | self.attendant_agent
        self.attendant_agent_node = functools.partial(
            agent_node, agent=self.context_aware_attendant_agent, name="Attendant"
        )

        # Define the supervisor
        self.supervisor_agent = create_team_supervisor(
            self.llm,
            "You are a supervisor managing the call simulation between the Customer and the Attendant. Decide who should act next based on the conversation.",
            ["Customer", "Attendant"]
        )


        # Create the state graph
        self.authoring_graph = StateGraph(ValidatorState)
        self.authoring_graph.add_node("Customer", self.customer_node)
        self.authoring_graph.add_node("Attendant", self.attendant_agent_node)
        self.authoring_graph.add_node("supervisor", self.supervisor_agent)

        # Define the sequence: NoteTaker runs first, then DocWriter
        self.authoring_graph.add_edge("Customer", "supervisor")
        self.authoring_graph.add_edge("Attendant", "supervisor")

        # Routing logic to ensure NoteTaker runs first, then DocWriter
        self.authoring_graph.add_conditional_edges(
            "supervisor",
            lambda x: x["next"],
    {"Customer": "Customer", "Attendant": "Attendant", "FINISH": END}
        )

        self.authoring_graph.add_edge(START, "Customer")
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
class ValidatorState(TypedDict):
    messages: Annotated[List[BaseMessage], operator.add]
    team_members: str
    next: str
    current_files: str
