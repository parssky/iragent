from datetime import datetime
from .message import Message

def get_time_now() -> str:
    """
    Just return current local time.
    """
    return datetime.now()

def simple_termination(word: str, message: Message) -> bool:
    """
    This is just a function that check if the termination keyword was in the message, return True or False.
    """
    if word in message.content:
        return True
    else:
        return False