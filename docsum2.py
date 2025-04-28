import os
from dotenv import load_dotenv
load_dotenv()

from groq import Groq
import groq

client = Groq(
    api_key=os.environ.get("GROQ_API_KEY"),  
)

def llm(text):
    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                #content is propmt
                "content":text,
            }
        ],
        model="meta-llama/llama-4-scout-17b-16e-instruct",
    )
    return chat_completion.choices[0].message.content


def split_text(text, max_chunk_size=1000):
    accumulator = []
    while len(text) > 0:
        accumulator.append(text[:max_chunk_size])
        text = text[max_chunk_size:]
    return accumulator

def summarize_text(text):
    prompt = f'''
    Summarize the following text in 1-3 sentences.

    {text}
    '''
    try:
        output = llm(prompt)
        return output.split('\n')[-1]
    except groq.APIStatusError:
        chunks = split_text(text, 10000)
        print('len(chunks)=', len(chunks))
        accumulator = []
        for i, chunk in enumerate(chunks):
            print('i=', i)
            summary = summarize_text(chunk)
            accumulator.append(summary)
        summarized_text = ' '.join(accumulator)
        summarized_text = summarize_text(summarized_text)
        # print('summarized_text=', summarized_text)
        return summarized_text

def summarize_image_from_url(image_url):
    """
    Uses Groq to summarize an image by passing a message that includes both text and an image URL.
    """
    message = [
        {
            "type": "text",
            "text": "What's in this image?"
        },
        {
            "type": "image_url",
            "image_url": {
                "url": image_url
            }
        }
    ]
    completion = client.chat.completions.create(
        model="meta-llama/llama-4-scout-17b-16e-instruct",
        messages=[
            {
                "role": "user",
                "content": message
            }
        ],
        temperature=1,
        max_completion_tokens=1024,
        top_p=1,
        stream=False,
        stop=None,
    )
    return completion.choices[0].message.content

def extract_text_from_pdf(pdf_path):
    """
    Extracts text from a PDF file using PyPDF2.
    """
    try:
        from PyPDF2 import PdfReader
    except ImportError:
        print("PyPDF2 is required to process PDF files. Install it via 'pip install PyPDF2'.")
        exit(1)
    try:
        reader = PdfReader(pdf_path)
    except Exception as e:
        print("Error opening PDF:", e)
        exit(1)
    
    text = ""
    for page_num, page in enumerate(reader.pages):
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"
        else:
            print(f"Warning: No text extracted from page {page_num}")
    return text

#default groq model has context window size for 8192 tokens/around 4000 words
#need to strip uncessary contents of html

'''
import fulltext
fulltext.get(args.filename, None)
print(summarize_text(text))
'''

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(
        prog='docsum',
        description='summarize the input document',
    )
    parser.add_argument('source',help='Path to the file or URL to summarize')
    args = parser.parse_args()

    image_extensions = ('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff')

    if args.source.lower().startswith('https://'):
        if args.source.lower().endswith(image_extensions):
            print("Processing image URL:", args.source)
            summary = summarize_image_from_url(args.source)
            print("Summary of the image:\n", summary)
        else:
            import requests
            try:
                response = requests.get(args.source)
                encoding = response.encoding if response.encoding else 'utf-8'
                html = response.content.decode(encoding, errors='replace')
            except Exception as e:
                print("Error fetching URL:",e)
                exit(1)
            '''
            print('filename=',args.filename)
            with open(args.filename,'r', encoding='utf-8') as fin:
            text = fin.read()
            print(summarize_text(text))
            '''
            # one way to solve the problem of too much text for the context window
            # is to remove the "unnecessary" text;
            # for html files, that is the html tags
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(html, features="lxml")
            text = soup.get_text()
            print("Summary of the webpage:\n")
            print(summarize_text(text))
    else:
        file_lower = args.source.lower()
        if file_lower.endswith(image_extensions):
            print("Local image summarization is not supported. Please provide an image URL.")
            exit(1)
        elif file_lower.endswith('.pdf'):
            print("Processing PDF file:", args.source)
            text = extract_text_from_pdf(args.source)
            print("Summary of the PDF document:\n")
            print(summarize_text(text))
        else:
            print("Processing file:", agrs.source)
            try:
                with open(args.source,'r',encoding='utf-8',errors='replace')as fin:
                    text = fin.read()
            except Exception as e:
                print("Error reading file:", e)
                exit(1)
            if args.source.lower().endswith('.html')or args.source.lower().endswith('.htm'):
                from bs4 import BeautifulSoup
                soup = BeautifulSoup(text, features="lxml")
                text = soup.get_text()
            print("Summary of the document:\n")
            print(summarize_text(text))
    """        
    with open(args.filename, 'r') as fin:
        html = fin.read()
        soup = BeautifulSoup(html, features="lxml")
        text = soup.text
        #print('text=', text)
        print(summarize_text(text))
    """

#print(chat_completion.choices[0].message.content)