from typing import List, Dict, Any
from .agent import Agent

class SequentialAgents:
    def __init__(self, agents: List[Agent]):
        self.agents = agents
        self.messages = []
    
    def start(self, query: str) -> str:
        """
        [
        {"role": "user", "content": query},
        {"role": "assistant", "content": "first_agent_answer"},
        ]
        """
        self.messages.append(self.build_role_user(query))
        for agent in self.agents:
            last_answer = self.messages[-1]["content"] # Last Message
            ag_chat = [self.build_role_user(last_answer)]
            agent_res = agent.call_messages(ag_chat)
            self.messages.append({"role": agent.name, "content": agent_res})

        return self.messages

    def build_role_user(self, q: str):
        return {"role": "user", "content": q}
    def build_assistant(self, q: str):
        return {"role": "assistant", "content": q}
    
