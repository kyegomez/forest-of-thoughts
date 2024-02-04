import os
from swarms import Mixtral
from fot.main import ForestOfAgents
from dotenv import load_dotenv

# Load env
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

# create llm
llm = Mixtral(max_new_tokens=3000, )

# Create a forest of agents
forest = ForestOfAgents(llm, num_agents=5, max_loops=100, max_new_tokens=100)

# Distribute tasks to the agents
forest.distribute_tasks("What is the meaning of life?")
