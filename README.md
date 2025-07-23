# iragent
<!-- README.md -->

<p align="center">
  <img src="https://raw.githubusercontent.com/parssky/iragent/main/docs/banner.svg" alt="iragent – a simple multi‑agent framework" width="70%" />
</p>

<p align="center">
  <a href="https://pypi.org/project/iragent"><img alt="PyPI" src="https://img.shields.io/pypi/v/iragent"></a>
  <img alt="Python" src="https://img.shields.io/pypi/pyversions/iragent">
  <img alt="License" src="https://img.shields.io/badge/License-MIT-yellow.svg">
  <img alt="CI" src="https://github.com/parssky/iragent/actions/workflows/ci.yml/badge.svg">
</p>

> **iragent** is a simple framework for building OpenAI‑Like, tool‑using software agents.  
> It sits halfway between a prompt‑engineering playground and a full orchestration layer—perfect for *experiments*, *research helpers* and *production micro‑agents*.

---

## ✨ Key features

| Feature | Why it matters |
|---------|----------------|
| **Composable `BaseAgent`** | Plug‑and‑play tools, memories and decision strategies |
| **Built‑in web toolbox**   | `WebSearchTool`, `WebScrapeTool`, `BrowserTool` wrap `requests`, *Google Search* and *BeautifulSoup4* utilities |
| **Automatic reasoning loop** | ReAct‑style plan‑‑>act‑‑>observe‑‑>reflect loop, powered by the OpenAI Chat API |
| **Vector & episodic memory** | Simple in‑RAM store + optional embedding‑based retriever |
| **Pythonic API** | Agents are *callables*; tools are plain dataclasses; no metaprogramming magic |
| **Lightweight deps** | Pure‑Python, ~8 MiB install; only `openai`, `requests`, `googlesearch‑python`, `bs4`, `lxml`, `nltk`:contentReference[oaicite:1]{index=1} |

---

## 🚀 Installation

```bash
# Requires Python 3.10+
pip install iragent
# Or directly from GitHub
pip install git+https://github.com/parssky/iragent.git
```

Set your OpenAI key once:
```bash
export OPENAI_API_KEY="sk‑..."
```

## ⚡ Quick start
```python
from iragent import agent

researcher = Agent(
    name="Researcher‑GPT",
    system_prompt="You are an expert researcher.",
    tools=[search, scrape],
    base_url= ""
    api_key= ""
    model= ""
    provider= "openai" # or ollama for local use
)

answer = agent.start("Who won the Nobel Prize in Physics in 2024 and why?")
print(answer)
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
