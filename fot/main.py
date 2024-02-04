import random
from typing import List
import uuid
import os

import chromadb
import chromadb.utils.embedding_functions as embedding_functions

from dotenv import load_dotenv

# Import the OpenAIChat model and the Agent struct
from swarms import Agent, data_to_text

from fot.agent_name_creator import create_agent_name

# Load the environment variables
load_dotenv()

intro_statements = [
    "Take a breath.",
"Attack this problem step-by-step.",
"Think this through.",
"Answer from a process-oriented mindset.",
"Channel Sherlock's deductive insight.",
"View through Einstein's eyes.",
"Embrace Da Vinci's curiosity.",
"Apply a historian's lens.",
"Employ philosopher's skepticism.",
"Utilize mathematical precision.",
"Engage in rigorous debate.",
"Forecast like a futurist.",
"Embody child-like curiosity.",
"Critique with an editor's eye.",
"Sift like an archaeologist.",
"Integrate cultural perspectives.",
"Unravel as a mystery.",
"Explore like a deep-sea diver.",
"Seek Zen enlightenment.",
"Strategize with game theory.",
"Experiment scientifically.",
"Craft a compelling story.",
"Construct like an architect.",
"Investigate as a detective.",
"Simplify complex ideas.",
"Navigate with care.",
"Apply Occam's Razor.",
"Reflect deeply on the issue.",
"Critically analyze details.",
"Recognize underlying patterns.",
"Question every assumption.",
"Synthesize ideas creatively.",
"Deduce logically.",
"Anticipate outcomes strategically.",
"Brainstorm innovative solutions.",
"Insight from cultural understanding.",
"Challenge with intellectual rigor.",
"Demand empirical evidence.",
"Adopt a multidisciplinary angle.",
"Theorize expansively.",
"Inquire with philosophical depth.",
"Dissect the problem analytically.",
"Methodize like Curie.",
"Narrate with effectiveness.",
"Require precision.",
"Draw historical parallels.",
"Analyze economically.",
"Think like a tech visionary.",
"Apply scientific rigor.",
"Imagine with artistic freedom.",
]


def agent_metadata(agent: Agent, task: str, output: str):
    """
    Returns the metadata for the agent.

    Returns:
        dict: The metadata for the agent.
    """
    out = {
        "Agent Name": f"{agent.agent_name}",
        "Agent ID": agent.id,
        "Agent History": agent.short_memory,
        "Agent System Prompt": agent.system_prompt,
        "task": task,
        "output": output,
    }
    return str(out)


class ForestOfAgents:
    """
    Represents a forest of agents that can perform tasks.

    Args:
        num_agents (int): The number of agents in the forest.
        max_loops (int): The maximum number of loops each agent can run.
        max_new_tokens (int): The maximum number of new tokens each agent can generate.
    """

    def __init__(
        self,
        llm,
        summarizer_lmm,
        query_creator_llm,
        num_agents: int,
        max_loops: int,
        max_new_tokens: int,
        docs: str = "/Users/miles/Documents/github/forest-of-thoughts/tweet_files",
    ):
        super().__init__()
        self.llm = llm
        self.num_agents = num_agents
        self.max_loops = max_loops
        self.max_new_tokens = max_new_tokens

        # A list of agents in the forest
        self.forest = []

        openai_ef = embedding_functions.OpenAIEmbeddingFunction(
                api_key=os.getenv("OPENAI_API_KEY"),
                model_name="text-embedding-3-small"
            )


        # Connect
        self.db = chromadb.Client(
        )

        # Create a collection
        self.collection = self.db.create_collection(
            name="forest-of-thoughts",
            embedding_function=openai_ef,
            )

        # Convert all files in folders to text
        for i in range(num_agents):
            self.forest.append(self.create_agent())
            
        self.summarizer = Agent(
            llm=summarizer_lmm,
            max_loops=1,
            agent_name=f"Summarizer - {create_agent_name()}",
            autosave=True,
        )
        
        self.query_creator_llm = Agent(
            llm=query_creator_llm,
            max_loops=1,
            max_new_tokens=100,
            agent_name=f"Query Creator - {create_agent_name()}",
            autosave=True,
        )
        

        if docs:
            print(f"Traversing directory: {docs}")
            self.traverse_directory(docs)

    def create_agent(self):
        """
        Creates a new agent with the specified parameters.

        Returns:
            Agent: The created agent.
        """
        return Agent(
            llm=self.llm,
            max_loops=self.max_loops,
            agent_name=create_agent_name(),
            system_prompt=None,
            autosave=True,
        )

    def create_agents(self):
        """
        Creates a list of agents based on the specified number of agents.

        Returns:
            list[Agent]: The list of created agents.
        """
        agents = [self.create_agent() for _ in range(self.num_agents)]

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
        outputs = self.distribute_task_to_agents(task, *args, **kwargs)
        return self.summarize_outputs(task, outputs, *args, **kwargs)

    def distribute_task_to_agents(self, task: str, *args, **kwargs):
        """
        Distributes the specified task to all agents in the forest.

        Args:
            task (str): The task to be performed.
            *args: Additional positional arguments for the task.
            **kwargs: Additional keyword arguments for the task.
        """
        
        queries = self.create_queries(task)
        print(f"\n\n===Queries===:\n {', '.join(queries)}\n\n")
        
        outputs = []
        for agent, query in zip(self.forest, queries):
            docs = self.query_documents(query, 5)
            
            doc_texts = "\n".join([doc for doc in docs])
            
            print(f"\n\n===Docs===:\n {doc_texts}\n\n")
            
            random_intro_statement = random.choice(intro_statements)
            
            print(f"\n\n===Random Intro Statement===:\n {random_intro_statement}\n\n")
            
            task_with_doc_context = f"""{doc_texts}\n\n
            Use the above information to answer the following question:
            {random_intro_statement}
            {task}"""
            
            out = agent.run(task_with_doc_context, *args, **kwargs)
            save_metadata = self.get_agent_metadata(agent, task, out)
            self.add_document(save_metadata)
            
            outputs.append(out)
        return outputs
    
    def create_queries(self, task: str):
        """
        Creates queries for the agents to use.

        Args:
            task (str): The task to be performed.

        Returns:
            list: The list of queries.
        """
        QUERY_CREATOR_PROMPT = f"""
            Create a unique search query whose result will help answer the following question:
            {task}
        """
        
        queries = []
        for _ in range(self.num_agents):
            self.query_creator_llm.set_system_prompt(QUERY_CREATOR_PROMPT)
            query = self.query_creator_llm.run(task)
            queries.append(query)
        return queries
    
    def summarize_outputs(self, question: str, outputs: List[str], *args, **kwargs):
        """
        Summarizes the outputs from the agents.

        Args:
            outputs (list): The list of outputs from the agents.

        Returns:
            str: The summarized output.
        """
        
        task = self.format_answers(outputs)
        self.summarizer.set_system_prompt(f"""
        Answer the following question:
        {question}   
        
        Identify and extract the key sentences or phrases from the following text that encapsulate its core message. Focus particularly on sentences that highlight thematic importance, have a high frequency of concept occurrence, or are explicitly emphasized by the author. Provide a concise rationale for each selection to illustrate why it's crucial to the text's overall understanding.
        
        By extracting the relevant information from the following thoughts:
        Be concise and opinionated, and answer the question directly, without vague language.           
        """
        )
        return self.summarizer.run(task, *args, **kwargs)

    def add_document(self, document: str):
        doc_id = str(uuid.uuid4())
        self.collection.add(ids=[doc_id], documents=[document])

        return doc_id
    
    def add_documents(self, documents: List[str]):
        doc_ids = [str(uuid.uuid4()) for _ in range(len(documents))]
        self.collection.add(ids=doc_ids, documents=documents)

        return doc_ids

    def query_documents(self, query: str, n_docs: int = 1):
        docs = self.collection.query(query_texts=[query], n_results=n_docs)["documents"]

        return docs[0]

    def get_agent_metadata(self, agent: Agent, task: str, output: str):
        """
        Returns the metadata for the specified agent.

        Args:
            agent (Agent): The agent to get metadata for.
            task (str): The task the agent is performing.

        Returns:
            dict: The metadata for the agent.
        """
        return agent_metadata(agent, task, output)
    
    def format_answers(self, answers: List[str]):
        """
        Formats the answers for the user.

        Args:
            answers (list): The list of answers from the agents.

        Returns:
            str: The formatted answers.
        """
        out = ""
        for i, answer in enumerate(answers):
            out += f"Thought {i + 1}: {answer}\n"
        return out

    def traverse_directory(self, directory_name: str):
        """
        Traverse through every file in the given directory and its subdirectories,
        and add each file to the forest.

        Parameters:
        - directory_name (str): The name of the directory to traverse.

        Returns:
        - list: A list of paths to each file in the directory and its subdirectories.
        """
        datas = []
        for root, dirs, files in os.walk(directory_name):
            for file in files:
                filepath = os.path.join(root, file)
                data = data_to_text(filepath)
                datas.append(data)
                
        random.shuffle(datas)
        
        num_docs = len(datas) // 10
        
        print(f"Adding {num_docs} documents to the database")
        self.add_documents(datas[:num_docs])
        print(f"Added {num_docs} documents to the database")
        