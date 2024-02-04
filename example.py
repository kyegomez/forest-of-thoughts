import os
from swarms import OpenAIChat, Mixtral
from fot.main import ForestOfAgents
from dotenv import load_dotenv
from fot.summarization_prompts import selective_reduction_type

# Load env
load_dotenv()

# OpenAI API key
api_key = os.getenv("OPENAI_API_KEY")

# create llm
openai = OpenAIChat(openai_api_base=api_key)
llm = Mixtral(max_new_tokens=3000, load_in_4bit=True)

# Create a forest of agents
forest = ForestOfAgents(
    openai,
    num_agents=5,
    max_loops=1,
    max_new_tokens=100,
    summarizer_prompt=selective_reduction_type,
)

# Distribute tasks to the agents
forest.run(
    "Solve the PNP prblem and give a detailed explanation of the"
    " solution."
)
