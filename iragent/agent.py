import inspect
import json
import re
import warnings
from typing import Any, Callable, Dict, List, get_type_hints

from openai import OpenAI

from .message import Message

"""
We need to extract docstring of each function too.
"""


class Agent:
    """!
    using this class we'll be able to define an agent.
    """

    def __init__(
        self,
        name: str,
        model: str,
        base_url: str,
        api_key: str,
        system_prompt: str,
        temprature: float = 0.1,
        max_token: int = 100,
        next_agent: str = None,
        fn: List[Callable] = [],
        provider: str = None,
        response_format: str = None,
        memory = None
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
        self.system_prompt = system_prompt
        ## set a name for the agent.
        self.name = name
        ## set a agent as next agent
        self.next_agent = next_agent

        self.function_map = {f.__name__: f for f in fn}
        ## list of tools that available for this agent to use.
        self.fn = [self.function_to_schema(f) for f in fn]
        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)

        # Support Structured output 
        self.response_format = response_format

        # Set Memory
        self.memory = memory() if memory is not None else None

    def call_message(self, message: Message, **kwargs) -> str:
        
        msgs = [{"role": "system", "content": self.system_prompt}]

        # If agent has a history 
        if self.memory:
            history = self.memory.get_history()
            if history:
                msgs.extend(history)       

        user_msg = {"role": "user", "content": message.content}
        msgs.append(user_msg)
           
        # Add to memory if it is first time
        if self.memory:
            self.memory.add_history(user_msg)
        # Provider will be removed in v0.1.8
        # TODO: Remove provider and use just one of them.
        if self.provider:
            warnings.warn(
                "'provider' is deprecated and will be removed in the next release. "
                "You no longer need to define 'provider' — the package will automatically "
                "select the appropriate backend.",
                FutureWarning,  # or DeprecationWarning if you want it hidden by default
                stacklevel=2
            )

            if self.provider == "openai":
                res =  self._call_openai(msgs=msgs, message=message, **kwargs)
            elif self.provider == "ollama":
                res = self._call_ollama_v2(msgs=msgs, message=message)
            else:
                raise ValueError(f"Unsupported provider: {self.provider}")
        
        else:
            res = self._call_ollama_v2(msgs=msgs, message=message)
        
        
        if self.memory:
            self.memory.add_history({"role": "assistant", "content": res.content})
            self.memory.add_message(res)
        return res

    def _call_ollama_v2(self, msgs: List[Dict], message: Message) -> Message:
        """
        There is some different when you want to use ollama or openai call. this function work with "role":"tool".
        this function use openai library for comunicate for ollama.
        """
        kwargs = dict(
            model=self.model,
            messages=msgs,
            max_tokens=self.max_token,
            temperature=self.temprature,
        )
        if self.response_format:
            kwargs["response_format"] = self.response_format
        if self.fn:
            kwargs["tools"] = [{"type": "function", "function": f} for f in self.fn]
        response = self.client.chat.completions.create(**kwargs)

        msg = response.choices[0].message
        
        # for function call
        if msg.tool_calls:
            msgs.append(msg.model_dump())

            for tool_call in msg.tool_calls:
                fn_name = tool_call.function.name
                arguments = json.loads(tool_call.function.arguments)
                fn = self.function_map.get(
                    fn_name, lambda **_: f"Function `{fn_name}` not found."
                )
                try:
                    result = fn(**arguments)
                except Exception as e:
                    result = f"Error executing {fn_name}: {str(e)}"
                msgs.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "name": fn_name,
                    "content": str(result)
                })
            followup = self.client.chat.completions.create(
                model=self.model,
                messages=msgs,
                max_tokens=self.max_token,
                temperature=self.temprature
            )
            follow_msg = followup.choices[0].message
            content = follow_msg.content.strip() if follow_msg.content else ""
            # Structured output handling
            if self.response_format:
                try:
                    content = json.loads(content)
                except json.JSONDecodeError:
                    content = {"error": "Invalid JSON response", "raw": content}  

            return Message(
                sender=self.name,
                reciever=self.next_agent or message.sender,
                content=followup.choices[0].message.content.strip(),
                metadata={"reply_to": message.metadata.get("message_id")},
            )
        # --- Normal assistant reply (no tools) ---
        msgs.append(msg.model_dump())
        content = msg.content.strip() if msg.content else ""
        # Structured output handling
        if self.response_format:
            try:
                content = json.loads(content)
            except json.JSONDecodeError:
                content = {"error": "Invalid JSON response", "raw": content}


        return Message(
            sender=self.name,
            reciever=self.next_agent or message.sender,
            content=content,
            metadata={"reply_to": message.metadata.get("message_id")},
        )

    def _call_openai(self, msgs: List[Dict], message: Message, **kwargs) -> Message:
        args = dict(
            model=self.model,
            messages=msgs,
            max_tokens=self.max_token,
            temperature=self.temprature,
            **kwargs
        )
        if self.response_format:
            args["response_format"] = self.response_format
        if self.fn:
            args["functions"] = self.fn
            args["function_call"] = "auto"
        response = self.client.chat.completions.create(**args)
        msg = response.choices[0].message
        # for function call
        if msg.function_call:
            fn_name = msg.function_call.name
            arguments = json.loads(msg.function_call.arguments)
            if fn_name in self.function_map:
                result = self.function_map[fn_name](**arguments)
                followup = self.client.chat.completions.create(
                    model=self.model,
                    messages=msgs
                    + [
                        msg,
                        {"role": "function", "tool_call_id": msg.tool_calls[0].id, "name": fn_name, "content": str(result)},
                    ],
                    max_tokens=self.max_token,
                    temperature=self.temprature,
                )
                return Message(
                    sender=self.name,
                    reciever=self.next_agent or message.sender,
                    content=followup.choices[0].message.content.strip(),
                    metadata={"reply_to": message.metadata.get("message_id")},
                )
        # Handle response format
        if self.response_format:
            try:
                parsed_content = json.loads(msg.content)
            except json.JSONDecodeError:
                parsed_content = {"error": "Invalid JSON response", "raw": msg.content}
            return Message(
                sender=self.name,
                reciever=self.next_agent or message.sender,
                content=parsed_content,
                metadata={"reply_to": message.metadata.get("message_id")},
            )

        return Message(
            sender=self.name,
            reciever=self.next_agent or message.sender,
            content=response.choices[0].message.content.strip(),
            metadata={"reply_to": message.metadata.get("message_id")},
        )

    def function_to_schema(self, fn: Callable) -> Dict[str, Any]:
        if hasattr(fn, "__self__") and hasattr(fn, "__func__"):
            real_fn = fn.__func__
        else:
            real_fn = fn
        sig = inspect.signature(real_fn)
        hints = get_type_hints(real_fn)
        doc_info = self.parse_docstring(real_fn)
        parameters = {}

        for name, param in sig.parameters.items():
            if name == "self":
                continue
            hint = hints.get(name, str)
            desc = doc_info["param_docs"].get(name, "No description")
            parameters[name] = {
                "type": self.python_type_to_json_type(hint),
                "description": desc,
            }

        return {
            "name": real_fn.__name__,
            "description": inspect.getdoc(real_fn) or "No description provided",
            "parameters": {
                "type": "object",
                "properties": parameters,
                "required": list(parameters.keys()),
            },
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

        return {"description": description, "param_docs": param_docs}


class UserAgent:
    def __init__(self) -> None:
        self.name = "user"

# A way for create simple different agents with same llm and provider
class AgentFactory:
    """
    A factory class for creating Agent instances with shared configuration.

    This class simplifies the process of creating multiple agents by 
    reusing common parameters such as `base_url`, `api_key`, `model`, 
    and `provider`. Additional agent-specific parameters can be passed 
    through the `create_agent` method.

    Attributes:
        base_url (str): The base URL for the agent's API requests.
        api_key (str): The API key used for authentication.
        model (str): The model identifier used by the agent.
        provider (str): The provider name (e.g., 'openai', 'azure', etc.).

    Methods:
        create_agent(name, **kwargs): 
            Creates and returns a new Agent instance using the shared
            configuration and any additional keyword arguments.    
    """
    def __init__(self, base_url: str, api_key: str, model: str, provider: str = None) -> Agent:
        self.base_url = base_url
        self.api_key = api_key
        self.model = model
        self.provider = provider
    
    def create_agent(self, name, **kwargs):
        return Agent(
            name=name,
            base_url=self.base_url,
            api_key=self.api_key,
            model= self.model,
            provider=self.provider,
            **kwargs
        )

