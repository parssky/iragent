
from openai import OpenAI
from typing import List, Dict, Any

class Agent:
    def __init__(self,
                name: str,
                model: str, 
                base_url: str, 
                api_key: str, 
                system_prompt: str, 
                temprature: float = 0.1,
                max_token: int=100,
                fn: List[Dict[str, Any]] = []
                ):
        self.base_url = base_url
        self.api_key = api_key
        self.model = model
        self.temprature = temprature
        self.max_token = max_token
        self.system_prompt= system_prompt
        self.name = name
        self.fn = fn
        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)
    
    def call_messages(self, messages: List[Dict[str, str]]) -> str:
        msgs = [
            {"role": "system", "content": self.system_prompt}
        ]
        msgs.extend(messages)
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
        return response.choices[0].message.content.strip()
    
    def call_message(self, query: str) -> str:
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": query}
        ]
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=self.max_token,
            temperature=self.temprature
        )
        return response.choices[0].message.content.strip()


