from typing import Dict

class Message:
    def __init__(self, 
                 sender: str, 
                 reciever: str, 
                 content: str, 
                 intent: str = None, 
                 metadata: Dict = None) -> None:
        self.sender = sender
        self.reciever = reciever
        self.content = content
        self.intent = intent
        self.metadata = metadata or {}
    
    def __repr__(self):
        return f"Message(from={self.sender}, to={self.reciever}, content={self.content})"
