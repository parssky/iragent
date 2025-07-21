import requests
from bs4 import BeautifulSoup
from typing import List
from nltk.tokenize import sent_tokenize, word_tokenize

def fetch_url(url: str, parser: str = "lxml") -> str:
    """
    This uses lxml libary for Parsing the web page.
    """
    resp = requests.get(url, timeout=15)
    resp.raise_for_status()
    page = BeautifulSoup(resp.text, parser)
    text = page.get_text(separator="\n", strip=True)
    return text

def chunker(text: str, token_limit: int = 512) -> List[str]:
    """
    Chunks text into pieces, each not exceeding `token_limit` words (as a rough proxy for tokens).
    Sentences are kept intact.
    """
    chunks = []
    sentences = sent_tokenize(text)
    current_chunk = []
    current_len = 0
    for sent in sentences:
        sent_len = len(word_tokenize(sent))
        # If adding this sentence would exceed the limit, start a new chunk
        if current_len + sent_len > token_limit:
            if current_chunk: 
                chunks.append(' '.join(current_chunk))
            current_chunk = [sent]
            current_len = sent_len
        else:
            current_chunk.append(sent)
            current_len += sent_len
    if current_chunk:
        chunks.append(' '.join(current_chunk))

    return chunks

