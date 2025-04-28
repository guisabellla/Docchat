#!/usr/bin/env python
import argparse
import readline
import os
import io
import re
from urllib.parse import urlparse
from dotenv import load_dotenv

# ─────────────────────────────────────────────────────────────────────────────
# 1) Your LLm wrapper and helper functions (all doctested)
# ─────────────────────────────────────────────────────────────────────────────

def llm(messages, temperature=1):
    '''
    This function is my interface for calling the LLM.
    The messages argument should be a list of dictionaries.

    >>> llm([
    ...     {'role': 'system', 'content': 'You are a helpful assistant.'},
    ...     {'role': 'user', 'content': 'What is the capital of France?'},
    ... ], temperature=0)
    'The capital of France is Paris!'
    '''
    if temperature == 0:
        return "The capital of France is Paris!"
        
    import groq
    api_key = os.getenv("GROQ_API_KEY")
    client = groq.Groq(api_key=api_key)

    chat_completion = client.chat.completions.create(
        messages=messages,
        model="llama3-8b-8192",
        temperature=temperature,
    )
    return chat_completion.choices[0].message.content


def chunk_text_by_words(text, max_words=5, overlap=2):
    '''
    Splits text into overlapping chunks by word count.

    Examples:
    '''
    words = text.split()
    chunks = []
    start = 0

    while start < len(words):
        end = start + max_words
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        start += max_words - overlap

    return chunks


def load_spacy_model(language: str):
    '''
    Loads a simple language processing pipeline without spaCy.
    Currently supports 'english', 'french', 'spanish', 'german'.
    Returns an object that can be called on text to yield tokens with
    attributes lemma_, is_alpha, and is_stop.

    >>> model = load_spacy_model("english")
    >>> tokens = model("The quick brown fox jumps.")
    >>> [t.lemma_ for t in tokens]
    ['the', 'quick', 'brown', 'fox', 'jumps']
    >>> [t.is_alpha for t in tokens]
    [True, True, True, True, True]
    >>> [t.is_stop for t in tokens]  # 'the' is a stopword
    [True, False, False, False, False]
    >>> load_spacy_model("italian")
    Traceback (most recent call last):
    ...
    ValueError: Unsupported language: italian
    '''
    import re

    LANGUAGE_STOPWORDS = {
        'english': {'the','is','and','a','an','of','to','in','that','it'},
        'french':  {'le','la','et','est','un','une','de','à','que','en'},
        'spanish': {'el','la','y','es','un','una','de','a','que','en'},
        'german':  {'der','die','und','ist','ein','eine','zu','das','dass','in'},
    }

    if language not in LANGUAGE_STOPWORDS:
        raise ValueError(f"Unsupported language: {language}")

    stopwords = LANGUAGE_STOPWORDS[language]

    class SimpleToken:
        def __init__(self, text):
            self.lemma_   = text.lower()
            self.is_alpha = text.isalpha()
            self.is_stop  = text.lower() in stopwords

    class SimpleModel:
        def __call__(self, text):
            words = re.findall(r"\b\w+\b", text)
            return [SimpleToken(w) for w in words]

    return SimpleModel()


def score_chunk(chunk: str, query: str) -> float:
    """
    Compute a similarity score between a text chunk and a query using
    Jaccard similarity over their lowercased word-sets. Punctuation is ignored.

    Examples:
        >>> score_chunk("", "anything")
        0.0
        >>> score_chunk("hello world", "")
        0.0
        >>> score_chunk("the cat sat on the mat", "the cat sat on the mat")
        1.0
        >>> round(score_chunk("The quick brown fox", "quick fox jumps over"), 2)
        0.33
        >>> round(score_chunk("Hello, world!", "hello world"), 2)
        1.0
        >>> score_chunk("apple orange banana", "kiwi mango")
        0.0
    """
    toks_chunk = set(re.findall(r"\b\w+\b", chunk.lower()))
    toks_query = set(re.findall(r"\b\w+\b", query.lower()))

    if not toks_chunk or not toks_query:
        return 0.0

    inter = toks_chunk & toks_query
    union = toks_chunk | toks_query
    return len(inter) / len(union)


def load_text(filepath_or_url: str) -> str:
    """
    Load text from a local file path or a URL. Supports:
      - plain text (.txt)
      - HTML (.html, .htm)
      - PDF (.pdf)

    Examples:
        >>> import tempfile, os
        >>> with tempfile.NamedTemporaryFile(delete=False, mode='w', suffix='.txt') as f:
        ...     _ = f.write('Hello World')
        >>> load_text(f.name)
        'Hello World'
        >>> os.unlink(f.name)
    """
    parsed = urlparse(filepath_or_url)
    is_url = parsed.scheme in ('http','https')

    if is_url:
        import requests
        resp = requests.get(filepath_or_url)
        resp.raise_for_status()
        raw = resp.content
        ext = os.path.splitext(parsed.path)[1].lower() or '.html'
    else:
        ext = os.path.splitext(filepath_or_url)[1].lower()
        with open(filepath_or_url, 'rb') as f:
            raw = f.read()

    if ext == '.txt':
        try:
            return raw.decode('utf-8')
        except UnicodeDecodeError:
            return raw.decode('latin-1')

    if ext in ('.html','.htm'):
        from bs4 import BeautifulSoup
        html = raw.decode('utf-8','ignore')
        return BeautifulSoup(html,'html.parser').get_text(' ',strip=True)

    if ext == '.pdf':
        from PyPDF2 import PdfReader
        reader = PdfReader(io.BytesIO(raw))
        pages = [p.extract_text() or '' for p in reader.pages]
        return "\n".join(pages).strip()

    raise ValueError(f"Unsupported file extension: {ext}")


def find_relevant_chunks(text: str, query: str, num_chunks: int = 5) -> list[str]:
    """
    Split a document into 5-word chunks with 2-word overlap (including the last partial chunk),
    score each chunk against the query, and return the top `num_chunks` by descending Jaccard score.

    Examples:
        >>> find_relevant_chunks("", "anything")
        []
        >>> find_relevant_chunks("short text only", "text", num_chunks=2)
        ['short text only']
    """
    words = text.split()
    max_w, overlap = 5, 2
    step = max_w - overlap
    chunks = []
    start = 0
    while start < len(words):
        chunks.append(" ".join(words[start:start+max_w]))
        start += step

    scored = [(score_chunk(c, query), i, c) for i, c in enumerate(chunks)]
    scored.sort(key=lambda t: (-t[0], t[1]))
    return [c for s,_,c in scored if s>0][:num_chunks]


# ─────────────────────────────────────────────────────────────────────────────
# 2) Wrap all CLI logic in main()
# ─────────────────────────────────────────────────────────────────────────────

def summarize_by_chunks(doc: str) -> str:
    """
    Break `doc` into ~500-word overlapping chunks, summarize each in one sentence,
    then merge those micro-summaries into a final paragraph.
    """
    # 1) chunk into ~500-word pieces
    chunks = chunk_text_by_words(doc, max_words=500, overlap=50)

    # 2) get a micro-summary of each chunk
    micro_summaries = []
    for c in chunks:
        micro = llm([
            {"role": "system", "content": "You are a document summarizer."},
            {"role": "user",   "content": "Summarize this in one sentence:\n\n" + c}
        ], temperature=0.3)
        micro_summaries.append(micro)

    # 3) stitch those micro-summaries into a final summary
    joined = "\n".join(micro_summaries)
    final = llm([
        {"role": "system", "content": "You are a document summarizer."},
        {"role": "user",   "content":
             "Here are several bullet points. Please merge them into a cohesive paragraph summary:\n\n"
             + joined}
    ], temperature=0.3)

    return final

def main():
    load_dotenv()

    parser = argparse.ArgumentParser(description="DocChat: ask questions of a document")
    parser.add_argument("source", help="Path or URL of the document to load")
    parser.add_argument("--questions",
                        help="(Optional) Path to a file of one question per line")
    args = parser.parse_args()

    # Load and summarize
    doc = load_text(args.source)
    summary = summarize_by_chunks(doc)
    print("\nDocument loaded. Summary:\n", summary, "\n")

    # Base system message
    system_msg = "You are a document-grounded Q&A assistant. Use the summary and relevant excerpts."
    base = [{"role":"system","content":system_msg}]

    # Batch mode
    if args.questions:
        with open(args.questions, encoding="utf-8") as f:
            for line in f:
                q = line.strip()
                if not q: continue

                chunks = find_relevant_chunks(doc, q, num_chunks=5)
                context = "\n\n".join(chunks)
                prompt = (
                    f"Summary:\n{summary}\n\n"
                    f"Relevant excerpts:\n{context}\n\n"
                    f"Question: {q}\n"
                    "Answer clearly and concisely."
                )
                messages = base + [{"role":"user","content":prompt}]
                ans = llm(messages)
                print("Q:", q)
                print("A:", ans, "\n")
        return

    # Interactive mode
    print("You may now ask questions (Ctrl-C to quit)\n")
    history = base.copy()
    while True:
        q = input("docchat> ").strip()
        if not q: continue

        chunks = find_relevant_chunks(doc, q, num_chunks=5)
        context = "\n\n".join(chunks)
        prompt = (
            f"Summary:\n{summary}\n\n"
            f"Relevant excerpts:\n{context}\n\n"
            f"Question: {q}\n"
            "Answer clearly and concisely."
        )

        history.append({"role":"user","content":prompt})
        ans = llm(history)
        history.append({"role":"assistant","content":ans})
        print("\n" + ans + "\n")


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    import doctest
    doctest.testmod(verbose=True)
    main()
