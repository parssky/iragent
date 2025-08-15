# iragent
<!-- README.md -->

<p align="center">
  <img src="https://raw.githubusercontent.com/parssky/iragent/main/docs/banner.svg" alt="iragent – a simple multi‑agent framework" width="90%" />
</p>

<p align="center">
  <a href="https://pypi.org/project/iragent"><img alt="PyPI" src="https://img.shields.io/pypi/v/iragent"></a>
  <img alt="Python" src="https://img.shields.io/badge/python-3.10%2B-blue">
  <img alt="License" src="https://img.shields.io/badge/License-MIT-yellow.svg">
  <img alt="CI" src="https://github.com/parssky/iragent/actions/workflows/build.yml/badge.svg">
  <a href="https://pepy.tech/projects/iragent"><img src="https://static.pepy.tech/badge/iragent" alt="PyPI Downloads"></a>
</p>

> **iragent** is a simple framework for building OpenAI‑Like, tool‑using software agents.  
> It sits halfway between a prompt‑engineering playground and a full orchestration layer—perfect for *experiments*, *research helpers* and *production micro‑agents*.

---

## ✨ Key features

| Feature                                      | Why it matters                                                                                                                                                                                                                                                                                       |
| -------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Composable `Agent` model**                 | Chain or orchestrate agents via `SimpleSequentialAgents`, `AgentManager`, and `AutoAgentManager` for flexible workflows                                                                                                                                                                             |
| **Auto-routing agent**                       | `AutoAgentManager` uses a language model to dynamically decide the next agent in the loop                                                                                                                                                                                                           |
| **Web-augmented agent**                      | `InternetAgent` uses `googlesearch`, `requests`, and summarizing agents to fetch and condense live web data                                                                                                                                                                                          |
| **Parallel summarization**                   | `fast_start` method uses `ThreadPoolExecutor` to speed up web content processing                                                                                                                                                                                                                     |
| **Prompt-driven summaries**                   | Summarization is driven by customizable system prompts and token-limited chunking for accurate context                                                                                                                                                                                               |
| **Simple, Pythonic design**                   | Agents are lightweight Python classes with callable message interfaces—no metaclasses or hidden magic                                                                                                                                                                                                |
| **Memory, BaseMemory**                        | `BaseMemory` provides foundational memory management for agents, storing conversation history and message objects. It supports adding, retrieving, and clearing memory, offering a flexible design for session-based context, interaction history, or task-specific memory across multiple agent invocations. Ideal for scenarios where the agent needs to recall past interactions for continuity. |
| **SummarizerMemory with summarizer agent**    | `SummarizerMemory` extends `BaseMemory` by integrating a summarizing `Agent` that automatically condenses long histories when memory limits are exceeded. This enables agents to maintain compact, relevant context over time, ensuring efficiency without losing key information.                      |
| **SmartAgentBuilder for automated agent creation** | `SmartAgentBuilder` automates breaking down a high-level task into structured subtasks, then creates specialized agents for each subtask using a sequential pipeline. It ensures that each agent is precisely configured with a strict role, and outputs an `AutoAgentManager` to run them in coordination. |

---

## SimpleAgenticRAG + KnowledgeGraphBuilder

**SimpleAgenticRAG** combines a FAISS-powered retriever (`KnowledgeGraphBuilder`) with agent-based orchestration for question answering.  
It follows a **Retriever → Generator** flow: search relevant chunks, then generate an answer with your LLM.

### Example (Local LLM)
```python
from sentence_transformers import SentenceTransformer
from iragent.models import KnowledgeGraphBuilder
from iragent.agent import AgentFactory
from iragent.models import SimpleAgenticRAG

base_url= "http://127.0.0.1:1234/v1" # use your own base_url from api provider or local provider like ollama.
api_key = "no-key" # use your own api_key.
provider = "ollama" # openai for openai like provider (vLLM or openrouter) and ollama for local use.
model = "qwen3-4b-instruct-2507"

emb_model = SentenceTransformer("all-MiniLM-L6-v2")
kg = KnowledgeGraphBuilder(embedding_model=emb_model, index_dir="./text-store/")
texts = ["FAISS (Facebook AI Similarity Search) is a library for efficient similarity search and clustering of dense vectors.",
        "It is written in C++ with bindings for Python, and is widely used for large-scale nearest-neighbor search.", 
        "FAISS supports both exact search and approximate search algorithms, making it flexible for different speed/accuracy needs.", 
        "The library was developed by Facebook’s AI Research (FAIR) team."]
kg.build_index_from_texts(texts)
agent_factory = AgentFactory(
    base_url=base_url,
    api_key=api_key,
    model=model,
    provider=provider
)

rag = SimpleAgenticRAG(
    kg = kg,
    agent_factory= agent_factory
)
answer = rag.ask("Who developed FAISS?")
```
See examples/SimpleAgenticRAG for more usage.

## 🚀 Installation

```bash
# Requires Python 3.10+
pip install iragent
# Or directly from GitHub
pip install git+https://github.com/parssky/iragent.git
```

## ⚡ Quick start
```python
from iragent.tools import get_time_now, simple_termination

factory = AgentFactory(base_url,api_key, model, provider)

agent1 = factory.create_agent(name="time_reader",
                            system_prompt="You are that one who can read time. there is a fucntion named get_time_now(), you can call it whether user ask about time or date.",
                            fn=[get_time_now]
                            )
agent2 = factory.create_agent(name="date_exctractor", 
                              system_prompt= "You are that one who extract time from date. only return time.")
agent3 = factory.create_agent(name="date_converter", 
                              system_prompt= "You are that one who write the time in Persian. when you wrote time, then in new line write [#finish#]")

manager = AutoAgentManager(
    init_message="what time is it?",
    agents= [agent1,agent2,agent3],
    first_agent=agent1,
    max_round=5,
    termination_fn=simple_termination,
    termination_word="[#finish#]"
)

res = manager.start()
res.content
```

## More docs

visit below url:
https://parssky.github.io/iragent/namespacemembers.html

## 📚 More Usage Examples

Explore practical examples and use cases in the [example directory](https://github.com/parssky/iragent/tree/main/example).


## Development
```bash
git clone https://github.com/parssky/iragent.git
cd iragent
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"          # ruff, pytest, etc.
```

## 🤝 Contributing
Pull requests are welcome! Please open an issue first if you plan large‑scale changes.
1- Fork → create feature branch

2- Write tests & follow ruff style (ruff check . --fix)

3- Submit PR; GitHub Actions will run lint & tests.

## 📄 License

This project is released under the MIT License.
