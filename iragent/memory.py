from .message import Message
from .agent import AgentFactory
from .prompts import SMART_MEMORY

class BaseMemory:
    """
    BaseMemory is a foundational memory management class for conversational agents.

    It stores two types of information:
    - `history`: a list of message dictionaries representing role-based dialogue turns 
                 (e.g., user and assistant messages).
    - `messages`: a list of raw Message objects, useful for storing additional metadata or original input/output.

    This class supports adding, retrieving, and clearing both types of memory and 
    is designed to be extended for more advanced memory strategies.

    Attributes:
        history (list[dict]): List of role-content dictionaries used in LLM context (e.g., {"role": "user", "content": "Hi"}).
        messages (list[Message]): List of Message objects (structured user inputs/outputs).

    Methods:
        add_history(msg): Adds a single dict or list of dicts to the conversation history.
        get_history(): Returns the stored conversation history as a list of dicts.
        clear_history(): Clears the conversation history.

        add_message(msg): Adds a Message object to the internal message list.
        get_messages(): Returns the stored messages as a list.
        clear_messages(): Clears all stored messages.
    """
    def __init__(self) -> None:
        self.history = []
        self.messages = []

    def add_history(self, msg: dict | list[dict]) -> None:
        if isinstance(msg, dict):
            self.history.append(msg)
        elif isinstance(msg, list) and all(isinstance(m, dict) for m in msg):
            self.history.extend(msg)
        else:
            raise TypeError("msg must be a dict or a list of dicts.")

    def get_history(self) -> list[dict]:
        return self.history

    def clear_history(self) -> None:
        self.history.clear()
    
    def add_message(self, msg: Message) -> None:
        self.messages.append(msg)

    def get_messages(self) -> list[dict]:
        return self.messages

    def clear_messages(self) -> None:
        self.messages.clear()


# Smart Memory that keep memory summarize and safe
class SummarizerMemory(BaseMemory):
    def __init__(self, agent_factory: AgentFactory, memory_limit: int=10) -> None:
        super().__init__()
        self.memory_limit = memory_limit
        self.summarizer = agent_factory.create_agent(
            name="summarizer",
            system_prompt=SMART_MEMORY,
            max_token= 256,
            memory=None
        )
    
    def add_history(self, msg: dict | list[dict]) -> None:
        super().add_history(msg)
        if len(self.get_history()) > self.memory_limit:
            self._summarize_history()
    
    def _summarize_history(self):
        content = "\n".join(f"{m["role"]}: {m["content"]}" for m in self.get_history())
        msg = Message(content=content)
        summarized = self.summarizer.call_message(msg)
        self.clear_history()
        self.add_history({"role": "system", "content": f"(summary) {summarized.content}"})


        