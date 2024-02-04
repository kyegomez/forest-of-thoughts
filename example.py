import os
from swarms import OpenAIChat
from fot.main import ForestOfAgents
from dotenv import load_dotenv

# Load env
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

# create llm
llm = OpenAIChat(openai_api_key=api_key)

# Create a forest of agents
forest = ForestOfAgents(llm, num_agents=5, max_loops=100, max_new_tokens=100)

# Distribute tasks to the agents
forest.distribute_tasks("What is the meaning of life?")
