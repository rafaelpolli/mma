from hma.utils import *
import functools
import operator



class Search:    
    def __init__(self):
        self.llm = ChatOpenAI(model=os.environ["MODEL"])
        
        self.search_agent = create_agent(
            self.llm,
            [chromadb_search],
            "You are a research assistant who can search for up-to-date info using the chromadb_search engine tool, you must use the tool. Don't makeup information, if you don't have enought information to retrieve, return 'I am not able to find information about this topic'",
        )
        
        self.search_node = functools.partial(agent_node, agent=self.search_agent, name="Search")

        self.scrapper_agent = create_agent(
            self.llm,
            [search_on_web],
            "You are a research assistant who can search for up-to-date info using the chromadb_search engine tool, you must use the tool. Don't makeup information, if you don't have enought information to retrieve, return 'I am not able to find information about this topic'",
        )
     
        self.scrapper_node = functools.partial(agent_node, agent=self.scrapper_agent, name="WebScraper")
        
        self.supervisor_agent = create_team_supervisor(
            self.llm,
            "You are a supervisor tasked with managing a conversation between the"
            " following workers:  Search, WebScraper. Given the following user request,"
            " respond with the worker to act next. Each worker will perform a"
            " task and respond with their results and status. When finished,"
            " respond with FINISH.",
            ["Search", "WebScraper"],
        )
        
        self.search_graph = StateGraph(SearchTeamState)
        self.search_graph.add_node("Search", self.search_node)
        self.search_graph.add_node("WebScraper", self.scrapper_node)
        self.search_graph.add_node("supervisor", self.supervisor_agent)
        
        # Define the control flow
        self.search_graph.add_edge("Search", "supervisor")
        self.search_graph.add_edge("WebScraper", "supervisor")
        self.search_graph.add_conditional_edges(
            "supervisor",
            lambda x: x["next"],
            {"Search": "Search", "WebScraper": "WebScraper", "FINISH": END},
        )
        
        
        self.search_graph.add_edge(START, "supervisor")
        self.chain = self.search_graph.compile()       
        
        self.search_chain = enter_chain | self.chain
        
# The following functions interoperate between the top level graph state
# and the state of the research sub-graph
# this makes it so that the states of each graph don't get intermixed
def enter_chain(message: str):
    results = {
        "messages": [HumanMessage(content=message)],
    }
    return results
        
# ResearchTeam graph state
class SearchTeamState(TypedDict):
    # A message is added after each team member finishes
    messages: Annotated[List[BaseMessage], operator.add]
    # The team members are tracked so they are aware of
    # the others' skill-sets
    team_members: List[str]
    # Used to route work. The supervisor calls a function
    # that will update this every time it makes a decision
    next: str