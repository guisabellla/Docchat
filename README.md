[![tests](https://github.com/guisabellla/Docchat/actions/workflows/tests.yml/badge.svg)](https://github.com/guisabellla/Docchat/actions/workflows/tests.yml)

# Docchat

This program allows me to interact with LLM llama3-8b-8192 on Groq and ask it to analyze documents. 
The program is good at summarizing texts for documents and main argument of a text.
![Gif of my project running](Animation.gif)

The format of the command is:
```
python docchat.py [-h] --questions questions.txt
```
The questions are stored in the `questions.txt`.

### Try:
When I enter the command in python terminal:
```
python docchat.py news2.txt --questions questions.txt
```

I get the result:
```
Document loaded. Summary:
 Here is a cohesive paragraph summary:

The National Institutes of Health (NIH), the world's largest funder of biomedical research, is facing significant challenges that may have far-reaching consequences for the health of Americans. The agency has fired over 1,300 employees and cancelled over $2 billion in research grants, according to Dr. Francis Collins, who led the agency for 12 years. The NIH is also grappling with a proposed 40% budget cut, 
staff firings, and supply shortages, which have led to low morale and concerns about the agency's ability to conduct research and care for patients. These changes have sparked worries among scientists and patients, with some considering leaving the US to continue their research and others concerned about the impact on life-saving trials and research. As a result, patients, including those seeking treatment for rare diseases, are expressing concern and disappointment, leaving them without hope for potential treatments.

Q: Summarize the document.
A: The National Institutes of Health (NIH), the world's largest funder of biomedical research, is facing significant challenges, including staff firings, research grant cancellations, and a proposed 40% budget cut. This has led to low morale, supply shortages, and concerns 
about the agency's ability to conduct research and care for patients, sparking worries among scientists and patients, particularly those 
with rare diseases seeking treatment.

Q: What is the author's main argument?
A: The author's main argument is that the National Institutes of Health (NIH) is facing significant challenges, including budget cuts, staff firings, and supply shortages, which have compromised its ability to fund research and care for patients, ultimately putting the health of Americans at risk.
Q: What are some key-words that frequently appear in the article? How many times do the key-words appear?
A: The key-words that frequently appear in the article are:

* "NIH" (5 times)
* "research" (3 times)
* "budget" (2 times)
* "employees" (1 time)
* "grants" (1 time)

Note: The frequency count only includes the excerpts provided and does not account for the full article.
```
This is particularly a good result on summarization and restating the author's main argument. However, the program has poorer performance on providing frequently appeared key-words, as sometimes it gives words that only appears one time in the document as frequently appeared key-words.

### Try with Spanish Texts

The `news_spanish.txt` file in the `docs` folder is a document written in Spanish. The LLM could successfully read the Spanish in the document and answer the questions in English, as the questions are also asked in English. 

My command is:

```
python docchat.py docs/news_spanish.txt --questions questions.txt
```
The questions in the `questions.txt` is the same as last try. I did not add any new command like asking for translation.
I get the result:
```
Document loaded. Summary:
 Here is a cohesive paragraph summary:

The ongoing war between Russia and Ukraine remains a contentious issue, with both sides trading claims and accusations. 
Russia claims to have recaptured the border region of Kursk, but Ukraine disputes this, saying that fighting continues. 
Meanwhile, North Korean soldiers are reportedly fighting in Russia, while Ukraine's President Zelensky reiterates calls 
for a "total and unconditional" ceasefire. The US Secretary of State Marco Rubio suggests that the next week will be crucial in determining whether the US will mediate a peace agreement between the two countries. Former US President Donald 
Trump has also weighed in, suggesting that Russian President Vladimir Putin may not be serious about ending the war and 
instead is just "giving pokes" by launching attacks on civilian areas. Trump proposed alternative measures such as "Banca" or "Secondary Sanctions" to address the situation, but it remains unclear whether these measures will be effective in bringing an end to the conflict.

Q: Summarize the document.
A: Here is a summary of the document:

The ongoing war between Russia and Ukraine remains unresolved, with both sides disputing claims and accusations. Russia 
allegedly recaptured the Kursk border region, but Ukraine denies this, and North Korean soldiers are reportedly fighting in Russia. The US Secretary of State suggests the next week will be crucial in determining mediation, while former President Trump thinks Putin may not be serious about ending the war.

Q: What is the author's main argument?
A: The author's main argument is not explicitly stated, as this text appears to be a news summary providing an update on the ongoing conflict between Russia and Ukraine, rather than making an argumentative claim. The text mainly presents a 
factual summary of the situation and quotes from various individuals, without taking a clear stance or making a persuasive argument.

Q: What are some key-words that frequently appear in the article? How many times do the key-words appear?
A: The key-words that frequently appear in the article are:

* Russia (7 times)
* Ukraine (6 times)
* War (4 times)
* Putin (2 times)
* Zelensky (2 times)
* Trump (2 times)
* Ceasefire (1 time)
* Sanctions (1 time)

These key-words appear a total of 25 times in the article.
```
Again, the LLM could successfully summarize the text and provide the author's main argument, though I am not sure how it provides key-words in English when reading a Spanish text. 


### Note
I thought the "1-2 sentence explanation of what you believe your grade should be" should be added to the readme file. When I realize I should upload this part to sakai, I already turned the work in... Here is my 1-2 sentence explanation:

I am not sure how to exactly deduct points but my test-cases badge does not pass the tests. I earn 2 extra credits for enabling asking questions in English when the document is non-english, I successfully tested this part with `news_spanish.txt` file. 
