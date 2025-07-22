from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable, List

from googlesearch import search
from tqdm import tqdm

from .agent import Agent
from .message import Message
from .prompts import AUTO_AGENT_PROMPT, SUMMARIZER_PROMPT
from .utility import chunker, fetch_url


class SimpleSequentialAgents:
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
    def __init__(
        self,
        init_message: str,
        agents: List[Agent],
        first_agent: Agent,
        max_round: int = 3,
        termination_fn: Callable = None,
        termination_word: str = None,
    ) -> None:
        self.auto_agent = Agent(
            "agent_manager",
            system_prompt="You are the Auto manager.",
            model=first_agent.model,
            base_url=first_agent.base_url,
            api_key=first_agent.api_key,
            temprature=0.1,
            max_token=32,
        )
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
        self.termination_word = termination_word

    def start(self) -> Message:
        list_agents_info = "\n".join(
            f"- [{agent_name}]-> system_prompt :{self.agents[agent_name].system_prompt}"
            for agent_name in self.agents.keys()
        )
        last_msg = self.init_msg
        for _ in range(self.max_round):
            if last_msg.reciever not in self.agents.keys():
                raise ValueError(f"No agent named {last_msg.reciever}")
            print(
                f"Routing from {last_msg.sender} -> {last_msg.reciever} \n content: {last_msg.content}"
            )
            res = self.agents[last_msg.reciever].call_message(last_msg)
            if self.termination_fn is not None:
                if self.termination_fn(self.termination_word, res):
                    return res
            last_msg = res

            for _ in range(self.max_round):
                next_agent = self.auto_agent.call_message(
                    Message(
                        sender="auto_router",
                        reciever="agent_manager",
                        content=AUTO_AGENT_PROMPT.format(
                            list_agents_info, last_msg.sender, last_msg.content
                        ),
                    )
                ).content
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
        provider: str = "openai",
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
            page_text = fetch_url(result.url)
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
        page_text = fetch_url(result.url)
        chunks = chunker(page_text, token_limit=self.chunk_size)

        # Ollama servers dislike shared clients; make one per thread
        if getattr(self.summerize_agent, "provider", "").lower() == "ollama":
            summarizer = Agent(
                name="Summarize Agent (thread‑local)",
                model=self.summerize_agent.model,
                base_url=self.summerize_agent.base_url,
                api_key=self.summerize_agent.api_key,
                system_prompt=self.summerize_agent.system_prompt,
                temprature=self.summerize_agent.temprature,
                max_token=self.summerize_agent.max_token,
                provider="ollama",
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
