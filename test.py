#!/usr/bin/env python
import argparse
import readline
import os
import io
import re
from urllib.parse import urlparse
from dotenv import load_dotenv

# ─────────────────────────────────────────────────────────────────────────────
# Your doctested helper functions
# ─────────────────────────────────────────────────────────────────────────────

def llm(messages, temperature=1):
    """
    >>> llm([
    ...     {'role':'system','content':'You are a helpful assistant.'},
    ...     {'role':'user','content':'What is the capital of France?'}
    ... ], temperature=0)
    'The capital of France is Paris!'
    """
    import groq
    client = groq.Groq()
    resp = client.chat.completions.create(
        messages=messages,
        model="llama3-8b-8192",
        temperature=temperature,
    )
    return resp.choices[0].message.content

def chunk_text_by_words(text: str, max_words: int=5, overlap: int=2) -> list[str]:
    """
    Splits text into overlapping chunks by word count.

    >>> text = "The quick brown fox jumps over the lazy dog. It was a sunny day."
    >>> chunks = chunk_text_by_words(text, max_words=5, overlap=2)
    >>> len(chunks)
    3
    >>> chunks[0]
    'The quick brown fox jumps'
    """
    words = text.split()
    if not (0 <= overlap < max_words):
        raise ValueError("overlap must be non-negative and less than max_words")
    if len(words) <= max_words:
        return [" ".join(words)]
    chunks, step, i = [], max_words - overlap, 0
    while i < len(words):
        chunks.append(" ".join(words[i:i+max_words]))
        i += step
    return chunks

import re
def score_chunk(chunk: str, query: str) -> float:
    """
    >>> round(score_chunk("The quick brown fox","quick fox jumps"),2)
    0.33
    """
    a = set(re.findall(r"\b\w+\b", chunk.lower()))
    b = set(re.findall(r"\b\w+\b", query.lower()))
    if not a or not b:
        return 0.0
    intersect = a & b
    union = a | b
    return len(intersect) / len(union)

def find_relevant_chunks(text: str, query: str, num_chunks: int=5) -> list[str]:
    """
    >>> doc="a b c d e f g h i j"
    >>> find_relevant_chunks(doc,"g h",num_chunks=2)
    ['g h i j', 'd e f g h']
    """
    chunks = chunk_text_by_words(text, max_words=5, overlap=2)
    scored = [(score_chunk(c, query), idx, c) for idx,c in enumerate(chunks)]
    scored.sort(key=lambda t: (-t[0], t[1]))
    return [c for s,_,c in scored if s>0][:num_chunks]

def load_text(filepath_or_url: str) -> str:
    """
    >>> import tempfile, os
    >>> with tempfile.NamedTemporaryFile(delete=False, mode='w', suffix='.txt') as f:
    ...     _ = f.write('Hello')
    >>> load_text(f.name)
    'Hello'
    >>> os.unlink(f.name)
    """
    parsed = urlparse(filepath_or_url)
    is_url = parsed.scheme in ('http','https')
    if is_url:
        import requests
        r = requests.get(filepath_or_url); r.raise_for_status()
        raw = r.content
        ext = os.path.splitext(parsed.path)[1].lower() or '.html'
    else:
        ext = os.path.splitext(filepath_or_url)[1].lower()
        with open(filepath_or_url,'rb') as f:
            raw = f.read()
    if ext=='.txt':
        try: return raw.decode('utf-8')
        except: return raw.decode('latin-1')
    if ext in ('.html',' .htm'):
        from bs4 import BeautifulSoup
        return BeautifulSoup(raw,'html.parser').get_text(' ',strip=True)
    if ext=='.pdf':
        from PyPDF2 import PdfReader
        reader = PdfReader(io.BytesIO(raw))
        return "\n".join(p.extract_text() or '' for p in reader.pages).strip()
    raise ValueError(f"Unsupported file extension: {ext}")

# ─────────────────────────────────────────────────────────────────────────────
# main(): loads a document, summarizes, then enters a retrieval-augmented loop
# ─────────────────────────────────────────────────────────────────────────────

def main():
    load_dotenv()

    p = argparse.ArgumentParser(description="DocChat: ask questions of a document")
    p.add_argument("source", help="Path or URL of the document")
    args = p.parse_args()

    # 1) load text
    doc = load_text(args.source)

    # 2) summarize
    summary = llm([
        {"role":"system","content":"You are a document summarizer."},
        {"role":"user","content":
            "Please summarize the following in one or two sentences:\n\n" + doc}
    ], temperature=0.3)

    # 3) set up history
    system_msg = (
        "You are a helpful assistant that answers questions about the provided document. "
        "Use the summary and relevant excerpts."
    )
    messages = [{"role":"system","content":system_msg}]

    print("\nDocument loaded. Summary:\n", summary, "\n")
    print("Ask questions (Ctrl-C to quit)\n")

    # 4) chat loop
    while True:
        q = input("docchat> ").strip()
        if not q:
            continue

        # 5) retrieve
        top5 = find_relevant_chunks(doc, q, num_chunks=5)
        context = "\n\n".join(top5)

        # 6) build prompt
        prompt = (
            f"Summary:\n{summary}\n\n"
            f"Relevant excerpts:\n{context}\n\n"
            f"Question: {q}\n"
            "Answer clearly and concisely."
        )

        messages.append({"role":"user","content":prompt})
        ans = llm(messages)
        messages.append({"role":"assistant","content":ans})

        print("\n" + ans + "\n")

# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # 1) run all doctests
    import doctest
    doctest.testmod(verbose=True)
    # 2) then start the document‐chat loop
    main()
