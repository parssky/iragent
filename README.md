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
</p>

> **iragent** is a simple framework for building OpenAI‑Like, tool‑using software agents.  
> It sits halfway between a prompt‑engineering playground and a full orchestration layer—perfect for *experiments*, *research helpers* and *production micro‑agents*.

---

## ✨ Key features

| Feature                      | Why it matters                                                                                                          |
| ---------------------------- | ----------------------------------------------------------------------------------------------------------------------- |
| **Composable `Agent` model** | Chain or orchestrate agents via `SimpleSequentialAgents`, `AgentManager`, and `AutoAgentManager` for flexible workflows |
| **Auto-routing agent**       | `AutoAgentManager` uses a language model to dynamically decide the next agent in the loop                               |
| **Web-augmented agent**      | `InternetAgent` uses `googlesearch`, `requests`, and summarizing agents to fetch and condense live web data             |
| **Parallel summarization**   | `fast_start` method uses `ThreadPoolExecutor` to speed up web content processing                                        |
| **Prompt-driven summaries**  | Summarization is driven by customizable system prompts and token-limited chunking for accurate context                  |
| **Simple, Pythonic design**  | Agents are lightweight Python classes with callable message interfaces—no metaclasses or hidden magic                   |

---

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
