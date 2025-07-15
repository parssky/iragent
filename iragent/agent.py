
from openai import OpenAI
from typing import List, Dict, Any, Callable, get_type_hints
import inspect
import json
import re
from .message import Message

"""
We need to extract docstring of each function too.
"""
class Agent:
    def __init__(self,
                name: str,
                model: str, 
                base_url: str, 
                api_key: str, 
                system_prompt: str, 
                temprature: float = 0.1,
                max_token: int=100,
                next_agent: str = None,
                fn: List[Callable] = []
                ):
        self.base_url = base_url
        self.api_key = api_key
        self.model = model
        self.temprature = temprature
        self.max_token = max_token
        self.system_prompt= system_prompt
        self.name = name
        self.next_agent = next_agent
        self.function_map = {f.__name__: f for f in fn}
        self.fn = [self.function_to_schema(f) for f in fn]
        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)
    
    def call_message(self, message: Message) -> str:
        msgs = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": message.content}
        ]

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
