from fot.main import ForestAgent

# Create a forest of agents
forest = ForestAgent(llm, num_agents=5, max_loops=100, max_new_tokens=100)
