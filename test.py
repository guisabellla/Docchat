#!/usr/bin/env python
import argparse, readline, os, io, re
from urllib.parse import urlparse
from dotenv import load_dotenv

# ─────────────────────────────────────────────────────────────────────────────
# 1) Your LLm wrapper and helper functions (all doctested)
# ─────────────────────────────────────────────────────────────────────────────

def llm(messages, temperature=1):
    """
    >>> llm([
    ...   {'role':'system','content':'You are a helpful assistant.'},
    ...   {'role':'user','content':'What is the capital of France?'}
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
    >>> text = "The quick brown fox jumps over the lazy dog."
    >>> chunk_text_by_words(text, max_words=5, overlap=2)
    ['The quick brown fox jumps',
     'fox jumps over the lazy',
     'over the lazy dog.']
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

def score_chunk(chunk: str, query: str) -> float:
    """
    >>> round(score_chunk("The quick brown fox","quick fox jumps"),2)
    0.33
    """
    a = set(re.findall(r"\b\w+\b", chunk.lower()))
    b = set(re.findall(r"\b\w+\b", query.lower()))
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)

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

    if ext == '.txt':
        try: return raw.decode('utf-8')
        except: return raw.decode('latin-1')
    if ext in ('.html','.htm'):
        from bs4 import BeautifulSoup
        return BeautifulSoup(raw,'html.parser').get_text(' ',strip=True)
    if ext == '.pdf':
        from PyPDF2 import PdfReader
        reader = PdfReader(io.BytesIO(raw))
        return "\n".join(p.extract_text() or '' for p in reader.pages).strip()

    raise ValueError(f"Unsupported file extension: {ext}")

# ─────────────────────────────────────────────────────────────────────────────
# 2) main(): CLI + summary + RAG loop
# ─────────────────────────────────────────────────────────────────────────────

def main():
    load_dotenv()

    parser = argparse.ArgumentParser(
        description="DocChat: ask questions of a document"
    )
    parser.add_argument(
        "source",
        help="Local file path or HTTP/HTTPS URL of the document"
    )
    args = parser.parse_args()

    # 1) load entire document
    doc = load_text(args.source)

    # 2) ask LLM to summarize it in one or two sentences
    summary = llm([
        {"role":"system", "content":"You are a document summarizer."},
        {"role":"user",   "content":
            "请用一句话概括以下内容：\n\n" + doc}
    ], temperature=0.3)

    # 3) init conversation history
    system_msg = (
        "你是一位文档问答助手，回答时请使用文档摘要和相关摘录作为依据。"
    )
    messages = [{"role":"system","content":system_msg}]

    print("\n文档已加载，摘要如下：\n", summary, "\n")
    print("可开始提问 (Ctrl-C 退出)\n")

    # 4) 无限聊天循环
    while True:
        user_q = input("docchat> ").strip()
        if not user_q:
            continue

        # 5) retrieve relevant chunks
        top5 = find_relevant_chunks(doc, user_q, num_chunks=5)
        context = "\n\n".join(top5)

        # 6) build a grounded prompt
        prompt = (
            f"文档摘要：\n{summary}\n\n"
            f"相关摘录：\n{context}\n\n"
            f"问题：{user_q}\n"
            "请用清晰、简洁的中文回答，必要时引用摘录。"
        )

        # 7) append user, call model, append assistant
        messages.append({"role":"user",    "content":prompt})
        answer = llm(messages)
        messages.append({"role":"assistant","content":answer})

        # 8) display the answer
        print("\n" + answer + "\n")

# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # 1) first run all doctests
    import doctest
    doctest.testmod(verbose=True)
    # 2) then launch the document‐chat
    main()
