import readline

from dotenv import load_dotenv
load_dotenv()

def llm(messages, temperature=1):
    '''
    This function is my interface for calling the LLM.
    The messages argument should be a list of dictionaries.

    >>> llm([
    ...     {'role': 'system', 'content': 'You are a helpful assistant.'},
    ...     {'role': 'user', 'content': 'What is the capital of France?'},
    ...     ], temperature=0)
    'The capital of France is Paris!'
    '''
    import groq
    client = groq.Groq()

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
        >>> text = "The quick brown fox jumps over the lazy dog. It was a sunny day and the birds were singing."
        >>> chunks = chunk_text_by_words(text, max_words=5, overlap=2)
        >>> len(chunks)
        7
        >>> chunks[0]
        'The quick brown fox jumps'
        >>> chunks[1]
        'fox jumps over the lazy'
        >>> chunks[4]
        'sunny day and the birds'
        >>> chunks[-1]
        'singing.'
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

    # Minimal stopword lists for each supported language
    LANGUAGE_STOPWORDS = {
        'english': {'the', 'is', 'and', 'a', 'an', 'of', 'to', 'in', 'that', 'it'},
        'french':  {'le', 'la', 'et', 'est', 'un', 'une', 'de', 'à', 'que', 'en'},
        'spanish': {'el', 'la', 'y', 'es', 'un', 'una', 'de', 'a', 'que', 'en'},
        'german':  {'der', 'die', 'und', 'ist', 'ein', 'eine', 'zu', 'das', 'dass', 'in'},
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
            # split on word boundaries
            words = re.findall(r"\b\w+\b", text)
            return [SimpleToken(w) for w in words]

    return SimpleModel()

"""
def score_chunk(chunk: str, query: str, language: str = "english") -> float:
    '''
    Scores a chunk against a user query by computing the Jaccard
    similarity of their (lower-cased, lemmatized) word‐sets after
    removing stopwords.

    Parameters:
        chunk    (str):    A piece of text.
        query    (str):    The user’s question or prompt.
        language (str):    One of "english", "french", "spanish", "german".

    Returns:
        float: Jaccard similarity in [0,1].

    Examples (English):
        >>> round(score_chunk(
        ...     "The sun is bright and hot.",
        ...     "How hot is the sun?",
        ...     language="english"
        ... ), 2)
        0.5

        >>> round(score_chunk(
        ...     "The red car is speeding down the road.",
        ...     "What color is the car?",
        ...     language="english"
        ... ), 2)
        0.14

        >>> score_chunk(
        ...     "Bananas are yellow.",
        ...     "How do airplanes fly?",
        ...     language="english"
        ... )
        0.0
    '''
    nlp = load_spacy_model(language)

    def preprocess(text: str) -> set[str]:
        doc = nlp(text.lower())
        return {
            token.lemma_
            for token in doc
            if token.is_alpha and not token.is_stop
        }

    chunk_words = preprocess(chunk)
    query_words = preprocess(query)

    if not chunk_words or not query_words:
        return 0.0

    intersection = chunk_words & query_words
    union        = chunk_words | query_words

    return len(intersection) / len(union)
"""

import re

def score_chunk(chunk: str, query: str) -> float:
    """
    Compute a similarity score between a text chunk and a query using
    Jaccard similarity over their lowercased word-sets. Punctuation is ignored.

    Parameters:
        chunk (str): The text segment to score.
        query (str): The user’s query.

    Returns:
        float: A value in [0.0, 1.0], where 1.0 means the sets of words
               are identical, and 0.0 means they share no words.

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
    # tokenize into alphanumeric words
    toks_chunk = set(re.findall(r"\b\w+\b", chunk.lower()))
    toks_query = set(re.findall(r"\b\w+\b", query.lower()))

    if not toks_chunk or not toks_query:
        return 0.0

    inter = toks_chunk & toks_query
    union = toks_chunk | toks_query
    return len(inter) / len(union)


#load text
import os
import io
from urllib.parse import urlparse

def load_text(filepath_or_url: str) -> str:
    """
    Load text from a local file path or a URL. Supports:
      - plain text (.txt)
      - HTML (.html, .htm)
      - PDF (.pdf)

    Parameters:
        filepath_or_url (str): Local path or HTTP/HTTPS URL.

    Returns:
        str: Extracted text.

    Raises:
        ValueError: Unsupported file extension.
        ImportError: Missing parsing library.
        requests.HTTPError: URL fetch failed.

    Examples:
        >>> import tempfile, os
        >>> # plain text
        >>> with tempfile.NamedTemporaryFile(delete=False, mode='w', suffix='.txt') as f:
        ...     _ = f.write('Hello World')
        >>> load_text(f.name)
        'Hello World'
        >>> os.unlink(f.name)

        >>> # HTML file
        >>> with tempfile.NamedTemporaryFile(delete=False, mode='w', suffix='.html') as f:
        ...     _ = f.write('<html><body><h1>Title</h1><p>Para.</p></body></html>')
        >>> load_text(f.name)
        'Title Para.'
        >>> os.unlink(f.name)

        >>> # URL without extension → treated as HTML
        >>> 'Example Domain' in load_text('http://example.com')
        True
    """
    parsed = urlparse(filepath_or_url)
    is_url = parsed.scheme in ('http', 'https')

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

    # Plain text
    if ext == '.txt':
        try:
            return raw.decode('utf-8')
        except UnicodeDecodeError:
            return raw.decode('latin-1')

    # HTML
    if ext in ('.html', '.htm'):
        try:
            from bs4 import BeautifulSoup
        except ImportError:
            raise ImportError("bs4 (BeautifulSoup) is required for HTML parsing")
        html = raw.decode('utf-8', 'ignore')
        soup = BeautifulSoup(html, 'html.parser')
        return soup.get_text(separator=' ', strip=True)

    # PDF
    if ext == '.pdf':
        try:
            from PyPDF2 import PdfReader
        except ImportError:
            raise ImportError("PyPDF2 is required for PDF parsing")
        reader = PdfReader(io.BytesIO(raw))
        pages = [page.extract_text() or '' for page in reader.pages]
        return "\n".join(pages).strip()

    raise ValueError(f"Unsupported file extension: {ext}")


def chunk_text_by_words(text: str, max_words: int = 100, overlap: int = 50) -> list[str]:
    """
    Split a text string into overlapping chunks by word count.

    Parameters:
        text (str): The input text to split.
        max_words (int): Maximum number of words per chunk.
        overlap (int): Number of words each chunk should overlap with the previous chunk.

    Returns:
        list[str]: A list of text chunks. Chunks shorter than max_words are only returned
                   if the entire text is shorter than max_words.

    Raises:
        ValueError: If overlap is negative or not less than max_words.

    Examples:
        >>> # text shorter than max_words → single (short) chunk
        >>> chunk_text_by_words("one two three", max_words=5, overlap=2)
        ['one two three']

        >>> # on 20-word text, max=8, overlap=3 → exactly 3 full chunks
        >>> sample = " ".join(str(i) for i in range(1, 21))
        >>> chunks = chunk_text_by_words(sample, max_words=8, overlap=3)
        >>> len(chunks)
        3
        >>> chunks[0]
        '1 2 3 4 5 6 7 8'
        >>> chunks[1]
        '6 7 8 9 10 11 12 13'
        >>> chunks[2]
        '11 12 13 14 15 16 17 18'

        >>> # invalid overlap
        >>> chunk_text_by_words("a b c", max_words=3, overlap=3)
        Traceback (most recent call last):
        ...
        ValueError: overlap must be non-negative and less than max_words
    """
    if not (0 <= overlap < max_words):
        raise ValueError("overlap must be non-negative and less than max_words")

    words = text.split()
    n = len(words)
    if n == 0:
        return []
    # if the whole text fits in one chunk, just return it
    if n <= max_words:
        return [" ".join(words)]

    step = max_words - overlap
    chunks = []
    start = 0
    # only emit chunks of exactly max_words
    while start + max_words <= n:
        chunk = words[start:start + max_words]
        chunks.append(" ".join(chunk))
        start += step

    return chunks


import re

def score_chunk(chunk: str, query: str) -> float:
    """
    Compute a similarity score between a text chunk and a query using
    Jaccard similarity over their word sets. Tokens are extracted by
    splitting on word boundaries and lowercased; punctuation is ignored.

    Parameters:
        chunk (str): The text segment to score.
        query (str): The user’s query.

    Returns:
        float: A value in [0.0, 1.0], where 1.0 means the sets of words
               are identical, and 0.0 means they share no words.

    Examples:
        >>> score_chunk("", "anything")  # empty chunk
        0.0
        >>> score_chunk("hello world", "")  # empty query
        0.0
        >>> score_chunk("the cat sat on the mat", "the cat sat on the mat")
        1.0
        >>> round(score_chunk("The quick brown fox", "quick fox jumps over"), 2)
        0.33
        >>> round(score_chunk("Hello, world!", "hello world"), 2)
        1.0
        >>> score_chunk("apple orange banana", "kiwi mango")  # no overlap
        0.0
    """
    # tokenize by words, lowercase
    tokens_chunk = set(re.findall(r"\b\w+\b", chunk.lower()))
    tokens_query = set(re.findall(r"\b\w+\b", query.lower()))

    # if either set is empty, no similarity
    if not tokens_chunk or not tokens_query:
        return 0.0

    intersection = tokens_chunk & tokens_query
    union = tokens_chunk | tokens_query

    return len(intersection) / len(union)


def find_relevant_chunks(text: str, query: str, num_chunks: int = 5) -> list[str]:
    """
    Split a document into 5-word chunks with 2-word overlap (including the last partial chunk),
    score each chunk against the query, and return the top `num_chunks` by descending Jaccard score.

    Examples:
        >>> find_relevant_chunks("", "anything")
        []
        >>> find_relevant_chunks("one two three four five", "")
        []
        >>> find_relevant_chunks("short text only", "text", num_chunks=2)
        ['short text only']
        >>> doc = "a b c d e f g h i j"
        >>> find_relevant_chunks(doc, "g h", num_chunks=2)
        ['g h i j', 'd e f g h']
    """
    # 1) build the chunks ourselves so the last partial chunk is included
    words = text.split()
    max_w, overlap = 5, 2
    step = max_w - overlap

    chunks = []
    start = 0
    while start < len(words):
        end = start + max_w
        chunks.append(" ".join(words[start:end]))
        start += step

    # 2) score & keep original index
    scored = [(score_chunk(c, query), i, c) for i, c in enumerate(chunks)]

    # 3) sort by score desc, then index asc
    scored.sort(key=lambda t: (-t[0], t[1]))

    # 4) pick top num_chunks with score>0
    return [chunk for score, _, chunk in scored if score > 0][:num_chunks]


if __name__ == '__main__':
    import doctest
    doctest.testmod(verbose=True)

    messages = [{
        'role': 'system',
        'content': 'You are a helpful assistant.  You always speak like a pirate.  You always answer in 1 sentence.'
    }]
    while True:
        text = input('docchat> ')
        messages.append({'role': 'user', 'content': text})

        result = llm(messages)

        # now add the assistant’s reply into the history
        messages.append({'role': 'assistant', 'content': result})

        print('result=', result)
        import pprint
        pprint.pprint(messages)


"""
if __name__ == '__main__':
    '''
    import doctest
    doctest.testmod(verbose=True)
    '''
    messages = []
    messages.append({
        'role': 'system',
        'content': 'You are a helpful assistant.  You always speak like a pirate.  You always answer in 1 sentence.'
    })
    while True:
        # get input from the user
        text = input('docchat> ')
        # pass that input to llm
        messages.append({
            'role': 'user',
            'content': text,
        })
        result = llm(messages)
        # FIXME:
        # Add the "assistant" role to the messages list
        # so that the `llm` has access to the whole
        # conversation history and will know what it has previously
        # said and update its response with that info.

        # print the llm's response to the user
        print('result=', result)
        import pprint
        pprint.pprint(messages)
"""