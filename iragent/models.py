from typing import List, Dict, Any, Callable
from .agent import Agent
from .message import Message
from .prompts import AUTO_AGENT_PROMPT,SUMMARIZER_PROMPT
from googlesearch import search
from .utility import fetch_url, chunker
from tqdm import tqdm

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
            system_prompt="You are the Auto manager.",
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
            f"- [{agent_name}]-> system_prompt :{self.agents[agent_name].system_prompt}" for agent_name in self.agents.keys()
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

class InternetAgent:
    def __init__(self, chunk_size: int,
                  model: str, 
                  base_url: str, 
                  api_key: str, 
                  temperature: float=0.1, 
                  max_token: int=512, 
                  provider: str ="openai") -> None:
        self.chunk_size = chunk_size
        self.summerize_agent = Agent(
            name="Summerize Agent",
            model=model,
            base_url=base_url,
            api_key=api_key,
            system_prompt=SUMMARIZER_PROMPT,
            temprature=temperature,
            max_token=max_token,
            provider= provider
            )
        
    
    def start(self, query: str, num_result) -> str:
        search_results = search(query, advanced=True, num_results=num_result)
        final_result = []
        for result in tqdm(search_results, desc="Searching the websites"):
            # Pass the seach with no title
            if result.title is None:
                continue
            page_text = fetch_url(result.url)
            chunks = chunker(page_text, token_limit=self.chunk_size)
            sum_list = []
            tqdm.write(f"Searching")
            for chunk in tqdm(chunks, desc="Reading"):
                msg = """
                    query: {}
                    context: {}
                    """
                sum_list.append(self.summerize_agent.call_message(Message(content=msg.format(query, chunk))).content)
            final_result.append(
                dict(
                    url= result.url,
                    title= result.title,
                    content= "\n".join(sum_list)
                )
            )
        return final_result
    
AUTO_AGENT_PROMPT= """
You are the Auto Agent Manager in a multi-agent AI system.

Your job is to decide which agent should handle the next step based on the output of the previous agent.

You will be given:
1. A list of agents with their names and descriptions (system prompts)
2. The output message from the last agent

Respond with only the name of the next agent to route the message to.

agents: {}

{} message: {}
"""