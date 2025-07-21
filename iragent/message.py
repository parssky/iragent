from typing import Dict

class Message:
    """!
    This is the message class and it's a wrapper class around messages between user and agent
    """
    def __init__(self, 
                 sender: str = None, 
                 reciever: str = None, 
                 content: str = None, 
                 intent: str = None, 
                 metadata: Dict = None) -> None:

        ## This is the one who sends the message.
        self.sender = sender
        ## This is the one who get's the message.
        self.reciever = reciever
        ## This is the cotnent of the message.
        self.content = content
        ## No explanation yet.
        self.intent = intent
        ## It's contain some extra information like what this message replies to.
        self.metadata = metadata or {}
    
    def __repr__(self):
        """!
        in this method we changed the way we want to represent Message object
        """
        return f"Message(from={self.sender}, to={self.reciever}, content={self.content})"
