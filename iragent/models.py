from typing import List, Dict, Any, Callable
from .agent import Agent
from .message import Message
from .prompts import AUTO_AGENT_PROMPT

class SimpleSequentialAgents:
    def __init__(self, agents: List[Agent], init_message: str):
        self.history = []
        # We just follow sequencially the agents.
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
    
    
class AutoAgentManager:
    def __init__(self, 
                 init_message: str, 
                 agents: List[Agent],
                 first_agent: Agent, 
                 max_round: int = 3,
                 termination_fn: Callable = None, 
                 termination_word: str = None) -> None:
        self.auto_agent = Agent(
            "agent_manager",
            system_prompt="You are the Auto manager",
            model=first_agent.model,
            base_url=first_agent.base_url,
            api_key=first_agent.api_key,
            temprature= 0.1,
            max_token=32
        )
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
        self.termination_word = termination_word
    
    def start(self) -> Message:
        list_agents_info = "\n".join(
            f"- {agent_name}: {self.agents[agent_name].system_prompt}" for agent_name in self.agents.keys()
        )        
        last_msg = self.init_msg
        for _ in range(self.max_round):
            if last_msg.reciever not in self.agents.keys():
                raise ValueError(f"No agent named {last_msg.reciever}")
            print(f"Routing from {last_msg.sender} -> {last_msg.reciever} \n content: {last_msg.content}")
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
                        content= AUTO_AGENT_PROMPT.format(list_agents_info, last_msg.sender ,last_msg.content)
                    )
                ).content
                if next_agent in self.agents.keys():
                    break
            last_msg.reciever = next_agent
        
        return last_msg

