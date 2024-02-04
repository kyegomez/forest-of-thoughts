[![Multi-Modality](agorabanner.png)](https://discord.gg/qUtxnK2NMf)

# Forest of thoughts
The forest of thoughts project is building an AI system called Swarm that enables agents to communicate and collaborate. The goal is to showcase Swarm's capabilities by applying it to complex real-world problems like climate change, education, and world hunger. 

The system uses a vector database to store text embeddings generated from source materials. Each agent can access tools like starting prompts and summarization techniques. The agents communicate in a 1-on-1 fashion, distributing tasks among themselves. 

Key features of Swarm include:

- Agent IDs for identification
- Feedback loops to improve outputs 
- Message classification 
- History tracking
- Query prompts and reranking to refine responses
- Summarization techniques like selective reduction, analogical reasoning, progressive summarization, narrative synthesis, paraphrasing, and layered analysis.

The project aims to prove Swarm's collaborative approach is superior to individual systems like ChatGPT. The demo will show Swarm applied to the 3 real-world problem scenarios, with side-by-side comparisons to ChatGPT. 

The goal is to highlight Swarm's ability to synthesize complex information more efficiently through agent communication and tailored summarization techniques. This process-oriented approach draws on research in knowledge synthesis and transfer learning.

Overall, the forest of thoughts project demonstrates how an AI system like Swarm that utilizes agent collaboration, vector embeddings, and smart summarization can outperform individual systems on complex reasoning tasks.

## Install
`$ pip install forest-of-thoughts`

## Usage
```python
import os
from swarms import OpenAIChat, Mixtral
from fot.main import ForestOfAgents
from dotenv import load_dotenv

# Load env
load_dotenv()

# OpenAI API key
api_key = os.getenv("OPENAI_API_KEY")

# create llm
openai = OpenAIChat(openai_api_base=api_key)
llm = Mixtral(max_new_tokens=3000, load_in_4bit=True)

# Create a forest of agents
forest = ForestOfAgents(
    openai, num_agents=5, max_loops=1, max_new_tokens=100
)

# Distribute tasks to the agents
forest.run("Solve the PNP prblem and give a detailed explanation of the solution.")


```


# License
MIT

