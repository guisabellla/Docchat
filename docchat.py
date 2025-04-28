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
    """
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
    """
    words = text.split()
    chunks = []
    start = 0

    while start < len(words):
        end = start + max_words
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        start += max_words - overlap

    return chunks

"""
import spacy

def load_spacy_model(language: str):
    
    #Loads a spaCy model for the specified language.
    
    LANGUAGE_MODELS = {
        'french': 'fr_core_news_sm',
        'german': 'de_core_news_sm',
        'spanish': 'es_core_news_sm',
        'english': 'en_core_web_sm',
    }

    if language not in LANGUAGE_MODELS:
        raise ValueError(f"Unsupported language: {language}")

    return spacy.load(LANGUAGE_MODELS[language])
"""

#my own version without spacy
def load_tokenizer(language: str):
    """
    Return a basic whitespace + punctuation tokenizer for the given language.
    >>> tok = load_tokenizer("english")
    >>> tok("Hello, world! This is 2025.")
    ['hello', 'world', 'this', 'is', '2025']
    >>> load_tokenizer("klingon")
    Traceback (most recent call last):
      ...
    ValueError: Unsupported language: klingon
    """
    SUPPORTED = {'english', 'french', 'spanish', 'german'}
    if language not in SUPPORTED:
        raise ValueError(f"Unsupported language: {language}")

    pattern = re.compile(r"\b\w+\b", re.UNICODE)
    def tokenize(text: str) -> list[str]:
        return [tok.lower() for tok in pattern.findall(text)]
    return tokenize

_STOPWORDS = {
    'english': {'the','and','is','in','it','of','to','a'},
    'french':  {'le','et','la','les','de','du','un','une'},
    'spanish': {'el','y','la','los','de','un','una','que'},
    'german':  {'der','die','und','das','in','zu','ein','eine'},
}

"""
def score_chunk(chunk: str, query: str, language: str = "french") -> float:
    
    #Scores a chunk against a user query using Jaccard similarity of lemmatized word sets
    #with stopword removal, using spaCy for multilingual support.

    Examples (French):
        >>> round(score_chunk("Le soleil est brillant et chaud.", "Quelle est la température du soleil ?", language="french"), 2)
        0.33
        >>> round(score_chunk("La voiture rouge roule rapidement.", "Quelle est la couleur de la voiture ?", language="french"), 2)
        0.25
        >>> score_chunk("Les bananes sont jaunes.", "Comment fonctionnent les avions ?", language="french")
        0.0

    Examples (Spanish):
        >>> round(score_chunk("El sol es brillante y caliente.", "¿Qué temperatura tiene el sol?", language="spanish"), 2)
        0.33
        >>> round(score_chunk("El coche rojo va muy rápido.", "¿De qué color es el coche?", language="spanish"), 2)
        0.25
        >>> score_chunk("Los plátanos son amarillos.", "¿Cómo vuelan los aviones?", language="spanish")
        0.0

    Examples (English):
        >>> round(score_chunk("The sun is bright and hot.", "How hot is the sun?", language="english"), 2)
        0.5
        >>> round(score_chunk("The red car is speeding down the road.", "What color is the car?", language="english"), 2)
        0.25
        >>> score_chunk("Bananas are yellow.", "How do airplanes fly?", language="english")
        0.0
    
    nlp = load_spacy_model(language)
"""

def score_chunk(chunk: str, query: str, language: str = "english") -> float:
    """
    Scores a chunk against a user query using Jaccard similarity
    over token sets (with simple stopword removal).
    Examples (French):
        >>> round(score_chunk("Le soleil est brillant et chaud.", "Quelle est la température du soleil ?", language="french"), 2)
        0.33
        >>> round(score_chunk("La voiture rouge roule rapidement.", "Quelle est la couleur de la voiture ?", language="french"), 2)
        0.25
        >>> score_chunk("Les bananes sont jaunes.", "Comment fonctionnent les avions ?", language="french")
        0.0

    Examples (Spanish):
        >>> round(score_chunk("El sol es brillante y caliente.", "¿Qué temperatura tiene el sol?", language="spanish"), 2)
        0.33
        >>> round(score_chunk("El coche rojo va muy rápido.", "¿De qué color es el coche?", language="spanish"), 2)
        0.25
        >>> score_chunk("Los plátanos son amarillos.", "¿Cómo vuelan los aviones?", language="spanish")
        0.0

    Examples (English):
        >>> round(score_chunk("The sun is bright and hot.", "How hot is the sun?", language="english"), 2)
        0.5
        >>> round(score_chunk("The red car is speeding down the road.", "What color is the car?", language="english"), 2)
        0.25
        >>> score_chunk("Bananas are yellow.", "How do airplanes fly?", language="english")
        0.0
    """

    tokenizer = load_tokenizer(language)
    stopwords = _STOPWORDS[language]

    def preprocess(text: str) -> set[str]:
        return {
            w for w in tokenizer(text)
            if w.isalpha() and w not in stopwords
        }

    chunk_words = preprocess(chunk)
    query_words = preprocess(query)

    if not chunk_words or not query_words:
        return 0.0

    intersection = chunk_words & query_words
    union = chunk_words | query_words
    return len(intersection) / len(union)

    def preprocess(text):
        doc = nlp(text.lower())
        return set(
            token.lemma_ for token in doc
            if token.is_alpha and not token.is_stop
        )

    chunk_words = preprocess(chunk)
    query_words = preprocess(query)

    if not chunk_words or not query_words:
        return 0.0

    intersection = chunk_words & query_words
    union = chunk_words | query_words

    return len(intersection) / len(union)


import sys
import doctest

def _run_doctests():
    """Run all the >>> examples in this module and exit with OK/fail."""
    failures, tests = doctest.testmod(verbose=True)
    if failures:
        sys.exit(f"{failures} doctest failures out of {tests} tests")
    else:
        print(f" All {tests} doctests passed!")
        sys.exit(0)

if __name__ == "__main__":
    if "--test" in sys.argv:
        _run_doctests()
    else:
        main()

def main():
    messages = [{
        'role': 'system',
        'content': 'You are a helpful assistant. ...'
    }]
    while True:
        text = input('docchat> ')
        messages.append({'role': 'user', 'content': text})
        result = llm(messages)
        messages.append({'role': 'assistant', 'content': result})
        print(result)

'''
if __name__ == '__main__':
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
'''