from typing import List, Dict, Any, Callable
from .agent import Agent
from .message import Message

class SimpleSequentialAgents:
    def __init__(self, agents: List[Agent], init_message: str):
        self.history = []
        # We don't need to know the next agent.
        for i in range(len(agents) - 1):
            agents[i].next_agent = agents[i + 1].name     
        self.agent_manager = AgentManager(
            init_message= init_message,
            agents=agents,
            max_round= len(agents),
            termination_fn= None,
            first_agent=agents[0]
        )
    
    def start(self) -> List[Message]:
        return self.agent_manager.start()
    



class AgentManager:
    def __init__(self, 
                 init_message: str, 
                 agents: List[Agent],
                 first_agent: Agent, 
                 max_round: int = 3, 
                 termination_fn: Callable = None) -> None:
        
        self.termination_fn = termination_fn
        self.max_round = max_round
        self.agents = {agent.name: agent for agent in agents}
        self.init_msg = Message(
            sender="user",
            reciever= first_agent.name,
            content=init_message,
            intent="User request",
            metadata= {}
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
    
