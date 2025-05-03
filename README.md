[![tests](https://github.com/guisabellla/Docchat/actions/workflows/tests.yml/badge.svg)](https://github.com/guisabellla/Docchat/actions/workflows/tests.yml)

# Docchat

This program allows me to interact with LLM llama3-8b-8192 on Groq and ask it to analyze documents. 
The program is good at summarizing texts for documents and main argument of a text.
![Gif of my project running](Animation.gif)

The format of the command is:
```
python docchat.py path/to/document.txt < questions.txt
```
This command supposedly works for barch mode using input redirection.

However, mine is a windows powershell, so instead, this command works on windows computer:
```
Get-Content questions.txt | python docchat.py path/to/document.txt
```


### Try:
When I enter the command in python terminal:
```
Get-Content questions.txt | python docchat.py docs/news2.txt
```

I get the result:
```
docchat>
Here is a summary of the document:

The National Institutes of Health (NIH) is facing significant budget cuts, layoffs, and supply shortages, which have compromised its ability to conduct research and care for patients. This has resulted in the firing of 1,300 employees and cancellation of over $2 billion in research grants, potentially impacting medical research and treatment options for generations to come. The cuts have sparked concerns among scientists and patients, with some considering leaving the US to continue their work, and have left patients with rare diseases without hope for potential treatments.

docchat>
The author's main argument is that the proposed 40% budget cut to the National Institutes of Health (NIH) will have severe and long-lasting consequences for medical research and patient care, jeopardizing the health of Americans and potentially leading to devastating outcomes for patients with rare diseases.

docchat>
Some key-words that frequently appear in the article are:

* NIH (7 times)
* Budget (3 times)
* Patients (3 times)
* Grants (2 times)

These key-words relate to the main topics of the article, including the National Institutes of Health, its budget and budget cuts, the impact on research and patient care, and the potential consequences for the future 
of medical science.

```
This docchat program is good at capturing key-words and analyzing related topics and themes, yet its summarization could be unstable in lens, as sometimes it has long summarization and sometimes summarization is one-sentence short. 

### Try with Spanish Texts

The `news_spanish.txt` file in the `docs` folder is a document written in Spanish. The LLM could successfully read the Spanish in the document and answer the questions in English, as the questions are also asked in English. 

My command is:

```
Get-Content questions.txt | python docchat.py docs/news_spanish.txt
```
The questions in the `questions.txt` is the same as last try. I did not add any new command like asking for translation.
I get the result:
```
docchat>
The situation in Ukraine's war with Russia remains tense, with conflicting reports about the recapture of the Kursk border region.

docchat>
There is no clear authorial argument present in the summary and excerpts. The text provides a factual summary of the current situation in Ukraine's war with Russia, reporting on conflicting claims and statements from various leaders.

docchat>
The key-words that frequently appear in the article are "Russia", "Ukraine", " Putin", "Zelensky", "ceasefire", and "sanctions".

These key-words appear the following number of times:

* Russia: 3 times
* Ukraine: 3 times
* Putin: 2 times
* Zelensky: 2 times
* ceasefire: 1 time
* sanctions: 1 time

docchat>
Goodbye!
```
The LLM could successfully summarize the text and provide the author's main argument, though I am not sure how it provides key-words in English when reading a Spanish text, and sometimes it provides words that only appeared 1 time as key-words, which could be concerning. 


### Note
1. The case badge is now working.
2. There is a try of asking question in English while the news content in a document is in Spanish on the README file.
3. I have changed my code from needing the `--questions` flag in my command to the new version that remove extra code and supposedly could accept a `python docchat.py news.txt < questions.txt`, but because my windows powershell does not support the `<` redirection operator, I changed the command to `Get-Content questions.txt | python docchat.py path/to/document.txt`, which works on windows powershell. 
