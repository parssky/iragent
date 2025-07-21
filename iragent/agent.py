
from openai import OpenAI
from typing import List, Dict, Any, Callable, get_type_hints
import inspect
import json
import re
import requests
from .message import Message

"""
We need to extract docstring of each function too.
"""
class Agent:
    """!
    using this class we'll be able to define an agent.
    """
    def __init__(self,
                name: str,
                model: str, 
                base_url: str, 
                api_key: str, 
                system_prompt: str, 
                temprature: float = 0.1,
                max_token: int=100,
                next_agent: str = None,
                fn: List[Callable] = [],
                provider: str = "openai"
                ):
        
        ## The platform we use for loading the large lanuage models. you should peak ```ollama``` or ```openai``` as provider.
        self.provider = provider
        ## This will be the base url in our agent for communication with llm.
        self.base_url = base_url
        ## Your api-key will set in this variable to create a communication. 
        self.api_key = api_key
        ## Choose the name of the model you want to use.
        self.model = model
        ## set tempreture for generating output from llm. 
        self.temprature = temprature
        ## set max token that will be generated. 
        self.max_token = max_token
        ## set system prompt that will 
        self.system_prompt= system_prompt
        ## set a name for the agent. 
        self.name = name
        ## set a agent as next agent
        self.next_agent = next_agent

        self.function_map = {f.__name__: f for f in fn}
        ## list of tools that available for this agent to use. 
        self.fn = [self.function_to_schema(f) for f in fn]
        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)
    
    def call_message(self, message: Message) -> str:
        msgs = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": message.content}
        ]

        if self.provider == "openai":
            return self._call_openai(msgs=msgs, message=message)
        if self.provider == "ollama":
            return self._call_ollama_v2(msgs=msgs, message=message)
        else:
            raise ValueError(f"Unsupported provider: {self.provider}")

    def _call_ollama(self, msgs: List[Dict], message: Message) -> Message:
        """!
        This function use http call for ollama provider.
        @param msgs:
            this is a list of dictionary
        """
        payload = {
            "model": self.model,
            "messages": msgs,
            "stream": False
        }
        function_payload = {
            "tools": [
                {
                    "type": "function",
                    "function": f
                } for f in self.fn
            ]
        }
        payload.update(function_payload)
 
        try:
            response = requests.post(f"{self.base_url.removesuffix('/v1')}/api/chat", json=payload)
        except Exception as e:
            raise ValueError(f"Error calling Ollama: {str(e)}")

        response.raise_for_status()
        result = response.json()
        msg = result["message"]
        tool_calls = msg.get("tool_calls", [])
        if not tool_calls:
            reply = msg.get("content", "")
            return Message(
                sender=self.name,
                reciever=self.next_agent or message.sender,
                content=reply.strip(),
                metadata={"reply_to": message.metadata.get("message_id")}
            )

        # Handle tool call (assume one for now)
        tool = tool_calls[0]
        fn_name = tool["function"]["name"]
        arguments = tool["function"]["arguments"]

        if fn_name in self.function_map:
            result_str = str(self.function_map[fn_name](**arguments))

            # Add tool call + tool response to messages for a second round
            followup_msgs = msgs + [
                {
                    "role": "assistant",
                    "tool_calls": tool_calls,
                    "content": ""
                },
                {
                    "role": "tool",
                    "tool_call_id": tool.get("id", fn_name),
                    "name": fn_name,
                    "content": result_str
                }
            ]

            followup_payload = {
                "model": self.model,
                "messages": followup_msgs,
                "stream": False
            }

            followup_response = requests.post(
                f"{self.base_url.removesuffix('/v1')}/api/chat", json=followup_payload
            )
            followup_response.raise_for_status()
            followup = followup_response.json()

            final_reply = followup["message"]["content"]
            return Message(
                sender=self.name,
                reciever=self.next_agent or message.sender,
                content=final_reply.strip(),
                metadata={"reply_to": message.metadata.get("message_id")}
            )

        # fallback if function is not found
        return Message(
            sender=self.name,
            reciever=self.next_agent or message.sender,
            content=f"Function `{fn_name}` is not defined.",
            metadata={"reply_to": message.metadata.get("message_id")}
        )      

    def _call_ollama_v2(self, msgs: List[Dict], message: Message) -> Message:
        """
        There is some different when you want to use ollama or openai call. this function work with "role":"tool".
        this function use openai library for comunicate for ollama.
        """
        kwargs = dict(
            model = self.model,
            messages = msgs,
            max_tokens = self.max_token,
            temperature = self.temprature
        )
        
        if self.fn:
            kwargs["tools"] = [
                {
                    "type": "function",
                    "function": f
                } for f in self.fn
            ]
        response = self.client.chat.completions.create(**kwargs)
        
        msg = response.choices[0].message
        # for function call
        if msg.tool_calls:
            fn_name = msg.tool_calls[0].function.name
            arguments = json.loads(msg.tool_calls[0].function.arguments)
            if fn_name in self.function_map:
                result = self.function_map[fn_name](**arguments)
                followup = self.client.chat.completions.create(
                    model=self.model,
                    messages= msgs + [
                        msg,
                        {
                            "role": "tool",
                            "name": fn_name,
                            "content": str(result)
                        }
                    ],
                    max_tokens=self.max_token,
                    temperature=self.temprature
                )
                return Message(
                    sender=self.name,
                    reciever=self.next_agent or message.sender,
                    content=followup.choices[0].message.content.strip(),
                    metadata={"reply_to": message.metadata.get("message_id")}
                )
            
        return Message(
            sender=self.name,
            reciever=self.next_agent or message.sender,
            content=response.choices[0].message.content.strip(),
            metadata={"reply_to": message.metadata.get("message_id")}
        )
    
    def _call_openai(self, msgs: List[Dict], message: Message) -> Message:
        kwargs = dict(
            model = self.model,
            messages = msgs,
            max_tokens = self.max_token,
            temperature = self.temprature
        )
        if self.fn:
            kwargs["functions"] = self.fn
            kwargs["function_call"] = "auto"
        response = self.client.chat.completions.create(**kwargs)
        msg = response.choices[0].message
        # for function call
        if msg.function_call:
            fn_name = msg.function_call.name
            arguments = json.loads(msg.function_call.arguments)
            if fn_name in self.function_map:
                result = self.function_map[fn_name](**arguments)
                followup = self.client.chat.completions.create(
                    model=self.model,
                    messages= msgs + [
                        msg,
                        {
                            "role": "function",
                            "name": fn_name,
                            "content": str(result)
                        }
                    ],
                    max_tokens=self.max_token,
                    temperature=self.temprature
                )
                return Message(
                    sender=self.name,
                    reciever=self.next_agent or message.sender,
                    content=followup.choices[0].message.content.strip(),
                    metadata={"reply_to": message.metadata.get("message_id")}
                )
            
        return Message(
            sender=self.name,
            reciever=self.next_agent or message.sender,
            content=response.choices[0].message.content.strip(),
            metadata={"reply_to": message.metadata.get("message_id")}
        )
        


    def function_to_schema(self,fn: Callable) -> Dict[str, Any]:
        sig = inspect.signature(fn)
        hints = get_type_hints(fn)
        doc_info = self.parse_docstring(fn)
        parameters = {}

        for name, param in sig.parameters.items():
            hint = hints.get(name, str)
            desc = doc_info["param_docs"].get(name, "No description")
            parameters[name] = {
                "type": self.python_type_to_json_type(hint),
                "description": desc
            }
            
        return {
            "name": fn.__name__,
            "description": inspect.getdoc(fn) or "No description provided",
            "parameters": {
                "type": "object",
                "properties": parameters,
                "required": list(parameters.keys())
            }
        }

    def python_type_to_json_type(self, py_type: Any) -> str:
        if py_type in [str]:
            return "string"
        elif py_type in [int]:
            return "integer"
        elif py_type in [float]:
            return "number"
        elif py_type in [bool]:
            return "boolean"
        elif py_type in [list, List]:
            return "array"
        elif py_type in [dict, Dict]:
            return "object"
        else:
            return "string"  # default fallback
    
    def parse_docstring(self, fn: Callable) -> Dict[str, Any]:
        doc = inspect.getdoc(fn) or ""
        lines = doc.strip().splitlines()

        # Extract top-level description (before Args/Parameters/etc.)
        desc_lines = []
        for line in lines:
            if re.match(r"^\s*(Args|Arguments|Parameters)\s*[:：]?", line):
                break
            desc_lines.append(line)
        description = " ".join(desc_lines).strip()

        # Extract parameter descriptions
        param_docs = {}
        param_block = "\n".join(lines)
        matches = re.findall(r"\b(\w+)\s*\(([^)]+)\):\s*(.+)", param_block)
        for name, _type, desc in matches:
            param_docs[name] = desc.strip()

        return {
            "description": description,
            "param_docs": param_docs
        }


class UserAgent:
    def __init__(self) -> None:
        self.name = "user"
