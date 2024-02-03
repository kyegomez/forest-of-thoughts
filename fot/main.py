
from dotenv import load_dotenv

# Import the OpenAIChat model and the Agent struct
from swarms import Agent, Mixtral

from fot.agent_name_creator import create_agent_name

# Load the environment variables
load_dotenv()


def agent_metadata(agent: Agent, task: str):
    """
    Returns the metadata for the agent.
    
    Returns:
        dict: The metadata for the agent.
    """
    out = {
        "Agent Name": f"{agent.ai_name}",
        "Agent ID": agent.id,
        "Agent History": agent.short_memory,
        "task": task,
    }
    return str


class ForestAgent:
    """
    Represents a forest of agents that can perform tasks.
    
    Args:
        num_agents (int): The number of agents in the forest.
        max_loops (int): The maximum number of loops each agent can run.
        max_new_tokens (int): The maximum number of new tokens each agent can generate.
    """
    def __init__(
        self,
        num_agents: int,
        max_loops: int,
        max_new_tokens: int,
    ):
        super().__init__()
        self.num_agents = num_agents
        self.max_loops = max_loops
        self.max_new_tokens = max_new_tokens
        self.forest = []
        
        for i in range(num_agents):
            self.forest.append(self.create_agent())
        
    def create_agent(self):
        """
        Creates a new agent with the specified parameters.
        
        Returns:
            Agent: The created agent.
        """
        return Agent(
            llm=Mixtral(
                max_new_tokens=self.max_new_tokens,
                load_in_4bit=True,
                use_flash_attention_2=True
            ),
            max_loops=self.max_loops,
            name=create_agent_name(),
            system_prompt=None,
            autosave=True,
            dashboard=True,
            tools=[None],
        )
        
    def create_agents(self):
        """
        Creates a list of agents based on the specified number of agents.
        
        Returns:
            list[Agent]: The list of created agents.
        """
        agents =  [self.create_agent() for _ in range(self.num_agents)]
        
        # Add the agents to the forest
        self.forest.extend(agents)
        
    
    def run(self, task: str, *args, **kwargs):
        """
        Runs the specified task on all agents in the forest.
        
        Args:
            task (str): The task to be performed.
            *args: Additional positional arguments for the task.
            **kwargs: Additional keyword arguments for the task.
        """
        agents = self.create_agents()
        for agent in agents:
            agent.start()
            agent.run()