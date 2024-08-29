import operator
from hma.utils import *
import functools
import hma.search as search
import hma.writer as writer
import hma.validator as validator

      

class Coordinator:
    def __init__(self):
        self.llm = ChatOpenAI(model=os.environ["MODEL"])
        self.writer_team = writer.Writer()
        self.search_team = search.Search()
        self.validator_team = validator.Validator()
        
        self.coordinator_node = create_team_supervisor(
            self.llm,
            "You are a coordinator managing a conversation between the following teams: {team_members}. Respond with the team to act next. Each team will perform a task and respond with their results and status. When finished, respond with FINISH.",
            ["SearchTeam", "WritingTeam", "ValidatorTeam"]
        )

        self.super_graph = StateGraph(State)
        self.super_graph.add_node("SearchTeam", get_last_message | self.search_team.search_chain | join_graph)
        self.super_graph.add_node("WritingTeam", get_last_message | self.writer_team.authoring_chain | join_graph)
        self.super_graph.add_node("ValidatorTeam", get_last_message | self.validator_team.authoring_chain | join_graph)

        self.super_graph.add_node("coordinator", self.coordinator_node)
        
        self.super_graph.add_edge("SearchTeam", "coordinator")
        self.super_graph.add_edge("WritingTeam", "coordinator")
        self.super_graph.add_edge("ValidatorTeam", "coordinator")
        
        # Ensure both teams are run at least once by defining a sequence
        self.super_graph.add_edge(START, "SearchTeam")
        self.super_graph.add_edge("SearchTeam", "WritingTeam")
        self.super_graph.add_edge("ValidatorTeam", "coordinator")
        self.super_graph.add_edge("WritingTeam", "coordinator")
        
        self.super_graph.add_conditional_edges(
            "coordinator",
            lambda x: x["next"],
            {
                "WritingTeam": "WritingTeam",
                "SearchTeam": "SearchTeam",
                "ValidatorTeam": "ValidatorTeam",
                "FINISH": END,
            },
        )
        
        self.super_graph = self.super_graph.compile()        
        
        
        
# Top-level graph state
class State(TypedDict):
    messages: Annotated[List[BaseMessage], operator.add]
    next: str


def get_last_message(state: State) -> str:
    return state["messages"][-1].content


def join_graph(response: dict):
    return {"messages": [response["messages"][-1]]}