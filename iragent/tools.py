from datetime import datetime

from .agent import AgentFactory
from .message import Message


def get_time_now() -> str:
    """!
    Just return current local time.
    @returns str:
        return current local time as ```string```
    """
    return datetime.now()


def simple_termination(word: str, message: Message) -> bool:
    """!
    This is just a function that check if the termination keyword was in the message, return True or False.

    @param word :
        termination keyword.
    @param message :
        a Message that will be checked if it have termination keyword ```word```
    @returns bool:
        True if ```word``` founded in ```message``` otherwise return ```False```
    """
    if word in message.content:
        return True
    else:
        return False

def agent_test(input: str, prompt: str, cfg: dict) -> str:
    """
    This function create an agent with that system_prompt and pass the input to this and get the output.  
    """
    agent_factory = AgentFactory(cfg)
    agent = agent_factory.create_agent(
        name= "Solo",
        system_prompt = prompt
    )
    msg = Message(content=input)
    return agent.call_message(msg).content
