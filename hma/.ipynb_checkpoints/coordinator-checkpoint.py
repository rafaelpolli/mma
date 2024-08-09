import operator
from hma.utils import *
import functools
import hma.research as research
import hma.writer as writer

class Coordinator:
    def __init__(self):
        self.llm = ChatOpenAI(model=os.environ["MODEL"])
        self.writer_team = writer.Writer()
        self.research_team = research.Research()
        
        
        self.supervisor_node = create_team_supervisor(
            self.llm,
            "You are a supervisor tasked with managing a conversation between the"
            " following teams: {team_members}. Given the following user request,"
            " respond with the worker to act next. Each worker will perform a"
            " task and respond with their results and status. When finished,"
            " respond with FINISH.",
            ["ResearchTeam", "PaperWritingTeam"],
        )

        
        # Define the graph.
        self.super_graph = StateGraph(State)
        # First add the nodes, which will do the work
        self.super_graph.add_node("ResearchTeam", get_last_message | self.research_team.research_chain | join_graph)
        self.super_graph.add_node(
            "PaperWritingTeam", get_last_message | self.writer_team.authoring_chain | join_graph
        )
        self.super_graph.add_node("supervisor", self.supervisor_node)
        
        # Define the graph connections, which controls how the logic
        # propagates through the program
        self.super_graph.add_edge("ResearchTeam", "supervisor")
        self.super_graph.add_edge("PaperWritingTeam", "supervisor")
        self.super_graph.add_conditional_edges(
            "supervisor",
            lambda x: x["next"],
            {
                "PaperWritingTeam": "PaperWritingTeam",
                "ResearchTeam": "ResearchTeam",
                "FINISH": END,
            },
        )
        self.super_graph.add_edge(START, "supervisor")
        self.super_graph = self.super_graph.compile()
        
        
        
# Top-level graph state
class State(TypedDict):
    messages: Annotated[List[BaseMessage], operator.add]
    next: str


def get_last_message(state: State) -> str:
    return state["messages"][-1].content


def join_graph(response: dict):
    return {"messages": [response["messages"][-1]]}