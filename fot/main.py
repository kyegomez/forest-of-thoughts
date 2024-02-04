import os
import uuid

import chromadb
from dotenv import load_dotenv

# Import the OpenAIChat model and the Agent struct
from swarms import Agent, data_to_text
from termcolor import colored


def create_agent_name():
    return f"tree_{uuid.uuid4()}"


# Load the environment variables
load_dotenv()


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
        num_agents: int,
        max_loops: int,
        max_new_tokens: int,
        docs: str = None,
        n_results: str = 2,
        *args,
        **kwargs,
    ):
        super().__init__()
        self.llm = llm
        self.num_agents = num_agents
        self.max_loops = max_loops
        self.max_new_tokens = max_new_tokens
        self.docs = docs
        self.n_results = n_results

        # A list of agents in the forest
        self.forest = []

        # Connect
        self.db = chromadb.Client()

        # Create a collection
        colored("Creating collection", "green")
        self.collection = self.db.create_collection(
            name="forest-of-thoughts"
        )

        # Convert all files in folders to text
        for i in range(num_agents):
            self.forest.append(self.create_agent())
            colored(f"Agent {i} created", "green")

        # Docs
        if docs:
            self.traverse_directory()

        # Seperate the total number of agents into duos for conversations
        self.duos = [
            self.forest[i : i + 2]
            for i in range(0, len(self.forest), 2)
        ]
        colored(f"Number of duos: {len(self.duos)}", "green")

        # Create the summarizer agent
        self.summarizer = self.summarizer_agent()

    def create_agent(self, *args, **kwargs):
        """
        Creates a new agent with the specified parameters.

        Returns:
            Agent: The created agent.
        """
        name = (str(create_agent_name()),)
        return Agent(
            llm=self.llm,
            max_loops=self.max_loops,
            agent_name=name,
            system_prompt=(
                "You're an tree in a forest of thoughts. Work with"
                f" your peers to solve problems. Your name is {name}"
            ),
            autosave=True,
            *args,
            **kwargs,
        )

    def run(self, task: str, *args, **kwargs):
        """
        Runs the specified task on all agents in the forest.

        Args:
            task (str): The task to be performed.
            *args: Additional positional arguments for the task.
            **kwargs: Additional keyword arguments for the task.
        """
        colored(
            f"Distributing tasks: {task} to all {self.num_agents}",
            "green",
        )
        self.distribute_task_to_agents(task, *args, **kwargs)

        # Then engage in duos
        out = self.seperate_agents_into_conversations(task)

        # Print out the output of the conversation
        for i in range(len(out)):
            colored(f"Conversation {i} output: {out[i]}", "cyan")

        # Summarize the conversation and save to database
        # summary = self.summarizer.run(f"Summarize: {task}", out)

        return out

    def distribute_task_to_agents(self, task: str, *args, **kwargs):
        """
        Distributes the specified task to all agents in the forest.

        Args:
            task (str): The task to be performed.
            *args: Additional positional arguments for the task.
            **kwargs: Additional keyword arguments for the task.
        """

        outputs = []
        for agent in self.forest:
            out = agent.run(
                f"{task}: History{self.context_history(task)}",
                *args,
                **kwargs,
            )
            save_metadata = self.get_agent_metadata(agent, task, out)
            self.add_document(save_metadata)

            outputs.append(out)
        return outputs

    def add_document(self, document: str):
        doc_id = str(uuid.uuid4())
        self.collection.add(ids=[doc_id], documents=[document])

        return doc_id

    def query_documents(self, query: str, n_docs: int = 1):
        docs = self.collection.query(
            query_texts=[query], n_results=n_docs
        )["documents"]

        return docs[0]

    def get_agent_metadata(
        self, agent: Agent, task: str, output: str
    ):
        """
        Returns the metadata for the specified agent.

        Args:
            agent (Agent): The agent to get metadata for.
            task (str): The task the agent is performing.

        Returns:
            dict: The metadata for the agent.
        """
        return agent_metadata(agent, task, output)

    def traverse_directory(self):
        """
        Traverse through every file in the given directory and its subdirectories,
        and return the paths of all files.
        Parameters:
        - directory_name (str): The name of the directory to traverse.
        Returns:
        - list: A list of paths to each file in the directory and its subdirectories.
        """
        for root, dirs, files in os.walk(self.docs):
            for file in files:
                data = data_to_text(file)
                added_to_db = self.add_document(data)
                print("Document added to Database ")
        return added_to_db

    def seperate_agents_into_conversations(self, task: str) -> list:
        # Take the duos and engage them in conversation using their .run method that intakes a task param with string
        # return the output of the conversation
        all_outputs = []
        # The conversation prompt that shows the main task of the conversation and the agents involved
        for duo in self.duos:
            conversation_prompt = (
                f"Conversation between {duo[0].agent_name} and"
                f" {duo[1].agent_name} about {task}:"
                f" {self.context_history(task)}"
            )
            output = duo[0].run(conversation_prompt)
            save_metadata = self.get_agent_metadata(
                duo[0], duo[1].short_memory, output
            )
            self.add_document(save_metadata)
            print(
                f"Conversation between {duo[0].agent_name} and"
                f" {duo[1].agent_name} saved to database"
            )
            all_outputs.append(output)
        return all_outputs

    def context_history(self, query: str):
        """
        Generate the agent long term memory prompt

        Args:
            system_prompt (str): The system prompt
            history (List[str]): The history of the conversation

        Returns:
            str: The agent history prompt
        """
        ltr = self.query_documents(query, self.n_results)

        context = f"""
            ####### Forest Database ################
            {ltr}
        """

        return context

    def summarizer_agent(self, *args, **kwargs):
        # Summarizer agent
        summarizer = "Summarizer Tree"
        self.summarizer = Agent(
            llm=self.llm,
            max_loops=self.max_loops,
            agent_name=summarizer,
            system_prompt=(
                "You're an tree in a forest of thoughts. Work with"
                " your peers to solve problems. You will summarize"
                f" the conversations. Your name is {summarizer}"
            ),
            autosave=True,
            *args,
            **kwargs,
        )

        # Add summarizer to the forest
        self.forest.append(self.summarizer)
        colored(f"Summarizer {summarizer} added to forest", "green")

        # Add summarizer metadata to database
        save_metadata = self.get_agent_metadata(
            self.summarizer, "Summarizer", ""
        )

        # Save to database
        self.add_document(save_metadata)

        return self.summarizer
