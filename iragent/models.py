import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable, List

import numpy as np
import tiktoken
from googlesearch import search
from tqdm import tqdm

from .agent import Agent, AgentFactory
from .memory import BaseMemory
from .message import Message
from .prompts import (
    AGENT_GENERATOR,
    AUTO_AGENT_PROMPT,
    GENERATOR_PROMPT,
    RETRIEVER_PROMPT,
    SMART_PROMPT_READER,
    SMART_PROMPT_WRITER,
    SUMMARIZER_PROMPT,
    TASK_GENERATOR,
)
from .tools import simple_termination
from .utility import chunker, create_agents, fetch_url


class SimpleSequentialAgents:
    """
    A lightweight wrapper for running multiple agents in a fixed, 
    predefined sequence.

    This class sets up a chain of agents where each agent's output is 
    automatically routed to the next agent in the list, until all 
    agents have executed in order.

    Internally, it uses `AgentManager` to handle message passing 
    and execution, with the number of rounds set to the number of agents.

    Attributes:
        history (list): Stores the conversation or execution history.
        agent_manager (AgentManager): Manages the sequential execution 
            of agents.

    Args:
        agents (List[Agent]): The list of agents to execute in sequence.  
            Each agent will have its `next_agent` attribute set to the 
            name of the following agent in the list.
        init_message (str): The initial message content passed to the 
            first agent.

    Methods:
        start() -> List[Message]:
            Runs the agents in sequential order, starting with the 
            initial message and passing outputs along the chain.
            Returns the list of `Message` objects representing the 
            results of each agent's execution.
    """    
    def __init__(self, agents: List[Agent], init_message: str):
        self.history = []
        # We just follow sequencially the agents.
        for i in range(len(agents) - 1):
            agents[i].next_agent = agents[i + 1].name
        self.agent_manager = AgentManager(
            init_message=init_message,
            agents=agents,
            max_round=len(agents),
            termination_fn=None,
            first_agent=agents[0],
        )

    def start(self) -> List[Message]:
        return self.agent_manager.start()


class AgentManager:
    """
    A simple multi-agent execution manager that routes messages between 
    agents in a fixed sequence, with optional early termination.

    Unlike `AutoAgentManager`, this class does not dynamically decide 
    the next agent to route to — it executes in a predefined order 
    based on the message's `reciever` field.

    Attributes:
        termination_fn (Callable): Optional function to determine when the 
            workflow should stop early.
        max_round (int): Maximum number of message-passing iterations allowed.
        agents (dict[str, Agent]): Dictionary of available agents, keyed by 
            agent name.
        init_msg (Message): The initial message passed to the first agent.

    Args:
        init_message (str): The initial request or instruction from the user.
        agents (List[Agent]): The list of agents participating in the workflow.
        first_agent (Agent): The first agent to receive the initial message.
        max_round (int, optional): Maximum number of routing rounds. Defaults to 3.
        termination_fn (Callable, optional): A function to check if the process 
            should terminate early. Defaults to None.

    Methods:
        start() -> Message:
            Executes the multi-agent workflow starting with `init_message`.  
            The process:
                - Sends the message to the current agent.
                - Evaluates termination conditions after each response.
                - Passes the output directly to the next agent defined by the 
                  `reciever` field in the message.
                - Stops when the termination function returns True or the 
                  maximum round count is reached.
            Returns the final `Message` from the last executed agent.
    """    
    def __init__(
        self,
        init_message: str,
        agents: List[Agent],
        first_agent: Agent,
        max_round: int = 3,
        termination_fn: Callable = None,
    ) -> None:
        self.termination_fn = termination_fn
        self.max_round = max_round
        self.agents = {agent.name: agent for agent in agents}
        self.init_msg = Message(
            sender="user",
            reciever=first_agent.name,
            content=init_message,
            intent="User request",
            metadata={},
        )

    def start(self) -> Message:
        last_msg = self.init_msg
        for _ in range(self.max_round):
            if last_msg.reciever not in self.agents.keys():
                raise ValueError(f"No agent named {last_msg.reciever}")
            print(f"Routing from {last_msg.sender} -> {last_msg.reciever}")
            res = self.agents[last_msg.reciever].call_message(last_msg)
            if self.termination_fn is not None:
                if self.termination_fn(res):
                    return res
            last_msg = res

        return last_msg


class AutoAgentManager:
    """
    A multi-agent orchestration manager that routes messages between agents 
    in an automated workflow, with support for termination conditions and 
    dynamic agent selection.

    This class coordinates the execution of multiple agents in a round-based 
    loop. It starts with an initial message sent to the first agent, then uses 
    an internal `agent_manager` to determine the next agent to route the 
    response to. The process continues until:
        - The termination function returns True, or
        - The maximum number of rounds is reached.

    Attributes:
        auto_agent (Agent): An internal controller agent responsible for 
            deciding which agent should handle the next step.
        first_agent (Agent): The first agent to receive the initial message.
        termination_fn (Callable): Optional function to determine when the 
            workflow should stop.
        max_round (int): Maximum number of message-passing iterations allowed.
        agents (dict[str, Agent]): Dictionary of available agents, keyed by 
            agent name.
        init_msg (Message): The initial message passed to the first agent.
        termination_word (str): Optional keyword used by `termination_fn` to 
            detect completion.

    Args:
        init_message (str): The initial request or instruction from the user.
        agents (List[Agent]): The list of agents participating in the workflow.
        first_agent (Agent): The starting agent for message routing.
        max_round (int, optional): Maximum number of routing rounds. Defaults to 3.
        termination_fn (Callable, optional): A function to check if the process 
            should terminate early. Defaults to None.
        termination_word (str, optional): Keyword used in termination checks. 
            Defaults to None.

    Methods:
        start(message: str = None) -> Message:
            Executes the multi-agent workflow starting from either the 
            `init_message` or a provided `message`.  
            The process:
                - Sends the message to the current agent.
                - Evaluates termination conditions after each response.
                - Uses `auto_agent` to determine the next agent in the sequence.
                - Stops when the termination function returns True or the 
                  maximum round count is reached.
            Returns the final `Message` from the last executed agent.
    """    
    def __init__(
        self,
        agents: List[Agent],
        first_agent: Agent,
        max_round: int = 3,
        termination_fn: Callable = None,
        termination_word: str = None,
    ) -> None:
        self.auto_agent = Agent(
            "agent_manager",
            system_prompt="You are the agent manager. Who route the message between agents and user.",
            model=first_agent.model,
            base_url=first_agent.base_url,
            api_key=first_agent.api_key,
            temprature=0.1,
            max_token=4096,
            memory=BaseMemory
        )
        self.first_agent = first_agent
        self.termination_fn = termination_fn
        self.max_round = max_round
        self.agents = {agent.name: agent for agent in agents}
        self.termination_word = termination_word

    def start(self, message = None) -> Message:
        list_agents_info = "\n".join(
            f"- [{agent_name}]-> system_prompt :{self.agents[agent_name].system_prompt}"
            for agent_name in self.agents.keys()
        )
        last_msg = Message(sender="user",reciever=self.first_agent.name,content=message) if message is not None else self.init_msg
        for _ in range(self.max_round):
            if last_msg.reciever not in self.agents.keys():
                raise ValueError(f"No agent named {last_msg.reciever}")
            print(
                f"Routing from {last_msg.sender} -> {last_msg.reciever} \n content: {last_msg.content}"
            )
            res = self.agents[last_msg.reciever].call_message(last_msg)
            if self.termination_fn is not None:
                if self.termination_fn(self.termination_word, res):
                    res.content = res.content.replace(self.termination_word, "").strip()
                    return res
            last_msg = res

            for _ in range(self.max_round):
                next_agent = (self.auto_agent.call_message(
                    Message(
                        sender=self.first_agent.name,
                        reciever="agent_manager",
                        content=AUTO_AGENT_PROMPT.format(
                            list_agents_info, message ,last_msg.sender, last_msg.content
                        ),
                    )
                )).content.strip()
                if next_agent.lower() == "finish":
                    return last_msg
                if next_agent in self.agents.keys():
                    break
            last_msg.reciever = next_agent

        return last_msg


class InternetAgent:
    """
    InternetAgent is a tool for conducting web-based searches, retrieving relevant web pages,
    chunking their content, and summarizing them using a specified language model.

    Attributes:
        chunk_size (int): Maximum token size for each chunk of webpage content.
        summerize_agent (Agent): An instance of the summarization agent for generating summaries
                                 from text chunks based on a system prompt.

    Args:
        chunk_size (int): Token limit for text chunking.
        model (str): The name or identifier of the language model to be used.
        base_url (str): Base URL for the API that powers the summarization model.
        api_key (str): API key for authenticating with the model provider.
        temperature (float, optional): Sampling temperature for generation. Defaults to 0.1.
        max_token (int, optional): Maximum number of tokens allowed in the summary output. Defaults to 512.
        provider (str, optional): Name of the model provider (e.g., "openai"). Defaults to "openai".

    Methods:
        start(query: str, num_result: int) -> list:
            Executes a web search for the given query, retrieves the content of top results,
            splits them into chunks, summarizes them using the summarization agent,
            and returns a list of dictionaries with URL, title, and summarized content.
    """

    def __init__(
        self,
        chunk_size: int,
        model: str,
        base_url: str,
        api_key: str,
        temperature: float = 0.1,
        max_token: int = 512,
        provider: str = None,
    ) -> None:
        self.chunk_size = chunk_size
        self.summerize_agent = Agent(
            name="Summerize Agent",
            model=model,
            base_url=base_url,
            api_key=api_key,
            system_prompt=SUMMARIZER_PROMPT,
            temprature=temperature,
            max_token=max_token,
            provider=provider,
        )

    def start(self, query: str, num_result) -> str:
        tqdm.write(
            f"\nStarting search for query: '{query}' with top {num_result} results...\n"
        )
        search_results = search(query, advanced=True, num_results=num_result)
        final_result = []
        for result in tqdm(
            search_results, desc="Processing search results", unit="site"
        ):
            # Pass the seach with no title
            if result.title is None:
                tqdm.write(f"Skipping result with missing title: {result.url}")
                continue
            tqdm.write(f"\nFetching: {result.title} ({result.url})")
            try:
                page_text = fetch_url(result.url)
            except Exception as exc:
                tqdm.write(f"Skipping {result.url}: {exc}")
                continue

            if not page_text:
                tqdm.write(f"Skipping empty page: {result.url}")
                return None            
            chunks = chunker(page_text, token_limit=self.chunk_size)
            sum_list = []
            tqdm.write("Searching")
            for chunk in tqdm(chunks, desc="Reading chunks", unit="chunk"):
                msg = """
                    query: {}
                    context: {}
                    """
                sum_list.append(
                    self.summerize_agent.call_message(
                        Message(content=msg.format(query, chunk))
                    ).content
                )
            final_result.append(
                dict(
                    url=result.url,
                    title=result.title,
                    content="\n".join(
                        [
                            item
                            for item in sum_list
                            if item != "No relevant information found."
                        ]
                    ),  # Check item with relevant info.
                )
            )
            tqdm.write(f"Finished summarizing: {result.title}\n")
        tqdm.write("Done processing all search results.\n")
        return final_result

    def fast_start(
        self, query: str, num_result: int, max_workers: int | None = None
    ) -> list[dict]:
        """
        A convenience wrapper that searches the Web, fetches the content of each hit,
        breaks the text into token‑limited chunks, and asks a language‑model “summarizer”
        to extract only the information relevant to the user’s query.

        ----------
        Attributes
        ----------
        chunk_size : int
            Maximum token length for each text chunk before it is passed to the
            summarization model.
        summerize_agent : Agent
            A pre‑configured LLM “agent” used to turn a chunk of raw page text
            into a concise, query‑focused summary.

        ----------
        Parameters
        ----------
        chunk_size : int
            Token limit used when splitting page text.
        model : str
            Name / identifier of the language model (e.g. ``"gpt-4o-2025-05-13"``).
        base_url : str
            Base URL for the model’s API endpoint.
        api_key : str
            API key or access token.
        temperature : float, optional (default = 0.1)
            Sampling temperature.
        max_token : int, optional (default = 512)
            Maximum length of each summary returned by the LLM.
        provider : str, optional (default = ``"openai"``)
            Identifies the backend.  Special‑casing is included for ``"ollama"``
            because its local HTTP server dislikes shared clients in a pool of
            threads.

        ----------
        Methods
        ----------
        start(query, num_result)
            Serial implementation – easy to read, useful for debugging.
        fast_start(query, num_result, max_workers=None)
            Threaded implementation that parallelises I/O for speed.
        _summarize_page(result, query)
            Worker routine run by each thread in ``fast_start``.  Not public.

        All methods return a ``list[dict]`` whose items look like::

            {
                "url":     "<page URL>",
                "title":   "<page title>",
                "content": "<summarised text>",
            }
        """
        tqdm.write(
            f"\nStarting search for query: '{query}' with top {num_result} results...\n"
        )
        search_results = search(query, advanced=True, num_results=num_result)

        # Keep only results with a title so the progress bar is accurate
        valid_results = [r for r in search_results if r.title]

        final_result: list[dict] = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_url = {
                executor.submit(self._summarize_page, r, query): r.url
                for r in valid_results
            }

            for fut in tqdm(
                as_completed(future_to_url),
                total=len(future_to_url),
                desc="Processing search results",
                unit="site",
            ):
                try:
                    item = fut.result()
                    if item:
                        final_result.append(item)
                except Exception as exc:
                    tqdm.write(f"Error while processing {future_to_url[fut]}: {exc}")

        tqdm.write("Done processing all search results.\n")
        return final_result

    def _summarize_page(self, result, query: str):
        """Fetch one page and summarise it (runs in its own thread)."""
        if result.title is None:
            tqdm.write(f"Skipping result with missing title: {result.url}")
            return None

        tqdm.write(f"\nFetching: {result.title} ({result.url})")
        try:
            page_text = fetch_url(result.url)
        except Exception as exc:
            tqdm.write(f"Skipping {result.url}: {exc}")
            return None
        
        if not page_text:
            tqdm.write(f"Skipping empty page: {result.url}")
            return None        

        chunks = chunker(page_text, token_limit=self.chunk_size)

        # Ollama servers dislike shared clients; make one per thread
        if getattr(self.summerize_agent, "provider", "").lower() == "ollama" :
            summarizer = Agent(
                name="Summarize Agent (thread‑local)",
                model=self.summerize_agent.model,
                base_url=self.summerize_agent.base_url,
                api_key=self.summerize_agent.api_key,
                system_prompt=self.summerize_agent.system_prompt,
                temprature=self.summerize_agent.temprature,
                max_token=self.summerize_agent.max_token,
            )
        else:
            summarizer = self.summerize_agent  # safe to reuse for OpenAI etc.

        summaries = []
        for chunk in chunks:
            msg = f"""
            query: {query}
            context: {chunk}
            """
            summaries.append(summarizer.call_message(Message(content=msg)).content)

        tqdm.write(f"Finished summarizing: {result.title}\n")
        return dict(
            url=result.url,
            title=result.title,
            content="\n".join(
                [s for s in summaries if s.strip() != "No relevant information found."]
            ),
        )


class SmartPrompt:
    """
    A utility class for generating optimized system prompts based on example 
    inputs and desired outputs using a two-agent collaborative workflow.

    This class leverages:
        1. `writer` – An agent that crafts an initial prompt tailored to produce 
           the desired output from a given input.
        2. `reader` – An agent that reviews and refines the generated prompt for 
           clarity, accuracy, and adherence to the intended task.

    Both agents are orchestrated by an `AutoAgentManager` to enable iterative 
    collaboration until the prompt meets the defined termination condition.

    Attributes:
        writer (Agent): The prompt creation agent.
        reader (Agent): The prompt review and refinement agent.
        manager (AutoAgentManager): Coordinates the interaction between 
            `writer` and `reader` agents.

    Args:
        agent_factory (AgentFactory): The factory used to create new agents.

    Methods:
        generate(input: str, output: str) -> str:
            Generates a refined prompt based on an example input and its 
            expected output. The process:
                - Passes the input and output examples to the manager.
                - Iteratively runs writer and reader agents.
                - Returns the final crafted system prompt.
    """    
    def __init__(self, agent_factory: AgentFactory) -> None:
        self.writer = agent_factory.create_agent(
            name="prompt_maker",
            temprature=0.1,
            system_prompt=SMART_PROMPT_WRITER,
            max_token = 512
        )
        self.reader = agent_factory.create_agent(
            name="prompt_reader",
            temprature=0.1,
            system_prompt=SMART_PROMPT_READER,
            max_token = 512
        )
        self.manager = AutoAgentManager(
            agents= [self.writer,self.reader],
            first_agent=self.writer,
            max_round=5,
            termination_fn=simple_termination,
            termination_word="[#finish#]"
        )
    
    def generate(self, input: str, output: str) -> str:
        msg = """
        Here is the input example :
        {}
        
        Here is the output example i have expected from system.
        {}

        """
        return self.manager.start(msg.format(input, output))


class SmartAgentBuilder:
    """
    A utility class for automatically generating and managing task-specific agents 
    based on a given high-level task description.

    This class uses a sequential pipeline of two agents:
        1. `task_generator` – Breaks down a high-level task into smaller, 
           structured subtasks.
        2. `agent_generator` – Creates dedicated agents for each subtask with 
           specific system prompts and configurations.

    The resulting agents are combined into an `AutoAgentManager` for coordinated 
    execution, ensuring that the output from one agent feeds into the next.

    Attributes:
        agent_factory (AgentFactory): Factory for creating agents with predefined 
            settings.
        task_generator (Agent): The agent responsible for generating subtasks 
            from a high-level task description.
        agent_generator (Agent): The agent responsible for creating specialized 
            agents based on generated subtasks.
        sequencial_agent (SimpleSequentialAgents): Manages the execution of 
            `task_generator` followed by `agent_generator`.
        agents (list): Stores created agents.

    Args:
        agent_factory (AgentFactory): The factory used to create new agents.
        max_token (int, optional): Maximum token limit for each agent. Defaults 
            to 1024.

    Methods:
        create_agent(task: str, init_message: str = None) -> list[Agent]:
            Generates agents for a given high-level task, creates an 
            AutoAgentManager to run them, and returns the configured manager.
            The process:
                - Pass the task to the sequential pipeline.
                - Convert generated subtask definitions into real agents.
                - Initialize an AutoAgentManager for coordinated multi-agent execution.
    """
    def __init__(self, agent_factory: AgentFactory, max_token: int=1024) -> None:
        self.agent_factory = agent_factory
        self.task_generator = self.agent_factory.create_agent(
            name="task_generator",
            temprature=0.1,
            max_token = max_token,
            system_prompt = TASK_GENERATOR,
        )
        self.agent_generator = self.agent_factory.create_agent(
            name = "agent_creator",
            temprature=0.0,
            max_token = max_token, 
            system_prompt= AGENT_GENERATOR, 
            response_format = {"type": "json_object"}
        )
        self.sequencial_agent = SimpleSequentialAgents(
            agents= [self.task_generator, self.agent_generator],
            init_message= ""
        )
        self.agents = []
    
    def create_agent(self, task: str, init_message: str = None) -> list[Agent]:
        self.sequencial_agent.agent_manager.init_msg.content = task
        agents_list = (self.sequencial_agent.start()).content

        agents_object = create_agents(
            agents_list=agents_list["agents"],
            agent_factory= self.agent_factory
        )
        print(f"Agents are created : {' | '.join([a.name for a in agents_object])}")
        manager = AutoAgentManager(
            agents= agents_object,
            first_agent=agents_object[0],
            max_round=2 * len(agents_object),
            termination_fn=simple_termination,
            termination_word="[#finish#]"
        )
        return manager



class KnowledgeGraphBuilder:
    def __init__(self, embedding_model, chunk_size: int = 128 , chunk_overlap: int= 32,index_dir= "./vector_store/"):
        import faiss
        # -----------
        self.embedding_model = embedding_model
        self.index_path = os.path.join(index_dir, "faiss_index.index")
        self.index_dir = index_dir
        self.docs_path = os.path.join(index_dir, "faiss_index.index.docs")
        self.index = None
        self.document = []
        self.faiss = faiss
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        if os.path.exists(self.index_path):
            self.load_index()
    
    def _chunker(self, text: str) -> list[str]:
        """
        Splits text into overlapping chunks based on token count.

        Args:
            text: The input text to split.
            max_tokens: Maximum tokens per chunk.
            overlap: Number of tokens to overlap between chunks.
            model_name: Model name for tokenizer compatibility.

        Returns:
            List of text chunks.
        """        
        model_name= "gpt-3.5-turbo"
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap must be smaller that chunk_size.")
        
        enc = tiktoken.encoding_for_model(model_name)
        tokens = enc.encode(text)
        chunks = []
        start = 0
        while start < len(tokens):
            end = start + self.chunk_size
            chunk_tokens = tokens[start:end]
            chunk_text = enc.decode(chunk_tokens)
            chunks.append(chunk_text)

            start += self.chunk_size - self.chunk_overlap
        
        return chunks
    
    def _read_markdown_and_chunk(
            self,
            dir_path: str,
            strip_markdown: bool = True,
            recursive: bool=True
    ) -> list[str]:
        if not os.path.isdir(dir_path):
            raise NotADirectoryError(f"Directory not found: {dir_path}")
        
        all_chunks = []

        for root, _, files in os.walk(dir_path):
            for file in files:
                if file.lower().endswith(".md"):
                    file_path = os.path.join(root, file)
                    with open(file_path, "r", encoding="utf-8") as f:
                        text = f.read()
                    
                    if strip_markdown:
                        # Remove images
                        text = re.sub(r"!\[.*?\]\(.*?\)", "", text)
                        # Remove links but keep link text
                        text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)
                        # Remove inline/code blocks
                        text = re.sub(r"`{1,3}.*?`{1,3}", "", text, flags=re.DOTALL)
                        # Remove formatting symbols
                        text = re.sub(r"[#>*_~]", "", text)
                    
                    chunks = self._chunker(text)
                    all_chunks.extend(chunks)
            if not recursive:
                break
        return all_chunks

    def _normalize(self,vectors: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        return vectors / norms

    def build_index_from_markdown(self, dir_path: str, strip_markdown: bool = True, recursive: bool=True):
        chunks = self._read_markdown_and_chunk(
            dir_path=dir_path,
            strip_markdown=strip_markdown,
            recursive=recursive
        )
        self.document.extend(chunks)
        embeddings = self.embedding_model.encode(chunks)
        embeddings = embeddings.astype("float32")
        embeddings = self._normalize(embeddings)
        dim = embeddings.shape[1]

        if self.index is None:
            self.index = self.faiss.IndexFlatIP(dim)
        
        self.index.add(embeddings)
        self.save_index()        

    def build_index_from_texts(self, texts: list[str]):
        """Create a FAISS index from a list of texts"""
        self.document.extend(texts)
    
        embeddings = self.embedding_model.encode(texts)
        embeddings = embeddings.astype("float32")
        embeddings = self._normalize(embeddings)
        dim = embeddings.shape[1]

        if self.index is None:
            self.index = self.faiss.IndexFlatIP(dim)
        
        self.index.add(embeddings)
        self.save_index()
    
    def save_index(self):
        if self.index:
            os.makedirs(self.index_dir, exist_ok=True)
            self.faiss.write_index(self.index, self.index_path)
            with open(self.docs_path, "w", encoding="utf-8") as f:
                for doc in self.document:
                    f.write(doc + "\n")
    def load_index(self):
        """Load FAISS index from disk"""
        self.index = self.faiss.read_index(self.index_path)
        if os.path.exists(self.docs_path):
            with open(self.docs_path, "r", encoding="utf-8") as f:
                self.document = [line.strip() for line in f]

    def search(self, query: str, k: int) -> List[str]:
        """
        Search for nearest neighbors, this function get query: str and k:int as input. 
        """
        query_vec = self.embedding_model.encode([query])
        if not isinstance(query_vec, np.ndarray):
            query_vec = np.array(query_vec)
        query_vec = query_vec.astype("float32")
        query_vec = query_vec / np.linalg.norm(query_vec, axis=1, keepdims=True) # Normalize
        distances, indices = self.index.search(query_vec, k)
        results = [
            str(self.document[i])
            for pos, i in enumerate(indices[0])
            if i < len(self.document)
        ]
        return results


class SimpleAgenticRAG:
    def __init__(self, kg: KnowledgeGraphBuilder, agent_factory: AgentFactory) -> None:
        self.kg_search: Callable[[str, int], List[str]] = kg.search
        self.retriever_agent = agent_factory.create_agent(
            name="retirever",
            system_prompt=RETRIEVER_PROMPT,
            max_token = 4096,
            fn=[self.kg_search]
        )
        self.generator_agent = agent_factory.create_agent(
            name="generator",
            system_prompt=GENERATOR_PROMPT,
            max_token = 1024
        )
        self.manager = AutoAgentManager(
            agents=[self.retriever_agent, self.generator_agent],
            first_agent=self.retriever_agent,
            max_round=10,
            termination_fn=simple_termination,
            termination_word= "[#finish#]"
        )
    
    def ask(self, query):
        return (self.manager.start(message=query)).content
