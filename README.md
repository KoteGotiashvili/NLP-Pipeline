# NLP Fundamentals

Welcome to the NLP Fundamentals repository, where we explore various fundamental concepts and techniques in Natural Language Processing (NLP). Below are the main topics covered:

## Table of Contents

1. [NLP Use Cases](#nlp-use-cases)
2. [NLP Pipelines](#nlp-pipelines)
3. [Spacy vs NLTK](#spacy-vs-nltk)
4. [Text Cleaning](#text-cleaning)
4. [Tokenization in Spacy](#tokenization-in-spacy)
5. [Language Processing Pipeline in Spacy](#language-processing-pipeline-in-spacy)
6. [Stemming and Lemmatization](#stemming-and-lemmatization)
7. [Part Of Speech (POS) Tagging](#part-of-speech-pos-tagging)
8. [Named Entity Recognition (NER)](#named-entity-recognition-ner)
9. [Text Representation](#text-representation)
   - Label One-Hot Encoding
   - Bag of Words
   - Stop Words
   - Text Representation using Bag of N-grams
   - Using TF-IDF (Term Frequency-Inverse Document Frequency)
   - Using Word Embeddings
   - Word Vectors in Spacy
   - Classification using Word Vectors in Spacy
   - Word Vectors in Gensim

## NLP Use Cases
![alt text](images/nlpusecases.png)

I will discuss few of them and add other use cases too:

**Text Classification:**  Sorting text into predefined categories
like spam detection in emails or sentiment analysis in customer 
reviews.

**Information Extraction/Gathering:** Identifying and extracting 
specific pieces of information from text,
such as names, dates, or events from news articles.

**Chat Bots:** AI-powered programs that simulate human 
conversation, used for customer 
support, information retrieval, or entertainment.

**Text Summarization:** Condensing large amounts of text into 
shorter, coherent summaries, 
often used for news articles or research papers.

**Machine Translation:** Automatically translating text from one language to another.

As you can see NLP is quite cool and useful skill to learn (:

## NLP Pipelines

Building NLP means perform various steps starting from data collection, cleaning 
end to deployment, monitoring, all of those steps are called NLP pipeline:
![alt text](images/nlppipeline.png)

**Data Acquisition:** This is the first step in the pipeline, and it involves collecting text data from a variety of sources, such as the web, social media, or customer reviews.

**Text Cleaning:** Once the data has been collected, it needs to be cleaned. This step involves removing irrelevant information, such as HTML tags, punctuation, and stop words. Stop words are words that are common in a language but do not provide much meaning, such as "the", "a", and "is".

**Pre-processing:** After the text has been cleaned, it is pre-processed. This step involves breaking the text down into smaller units, such as words or phrases (tokenization), and converting the words to their base form (lemmatization or stemming).

**Feature Engineering:** In this step, features are extracted from the text data. Features are numerical representations of the text that can be used by machine learning models. There are many different feature engineering techniques, such as bag-of-words and TF-IDF.

**Modeling:** Once the features have been extracted, they are used to train a machine learning model. The model can then be used to perform a variety of NLP tasks, such as sentiment analysis, text classification, or machine translation.

**Evaluation:** After the model has been trained, it is evaluated to assess its performance. This step involves testing the model on a set of unseen data and measuring how well it performs.

**Deployment:** If the model performs well on the evaluation task, it can be deployed into production. This means that the model can be used to make predictions on new data.

**Monitoring and Model Updating:** Once the model is deployed, it is important to monitor its performance over time. If the model's performance starts to degrade, it may need to be updated with new data or retrained.

## Spacy vs NLTK
**Spacy:** Object-oriented approach. Spacy provides most efficient algorithms for particular tasks, 
don't gives you a lot of choice but the best one.

**NLTK:** String processing library. NLTK give you broader choice of algorithms,
it is more customizable, good for researchers.

**Overally both library have their use, and both is the good, but I'm going to use Spacy.**
*

## Text Cleaning
There are few important steps for text cleaning:

**Lowercasing**

**Remove HTML tags**

**Remove URLS**

**Remove Punctuations**

**Spelling Correction**

**Handling Emojis**

And so on, It's basically depends on data.




## Tokenization in Spacy

**Tokenization** is process of splitting text into meaningful segments, parts.

**Sentence Tokenization:** Split sentences 

**Word Tokenization:** Split words

Both Sentence and Word Tokenizer code implementation in Spacy: [Tokenization](Tokenization.py)


### Token Attributes 
 Token have a lot of different attributes, For instance

**LIKE_NUM:** which checks whether token in numerical value.

**LIKE_EMAIL:** Token text look like an email address. 

**LIKE_URL:** 	Token text looks like a URL. 

And there are a lot of different ones like that which helps to understand token context.


## Language Processing Pipeline in Spacy

There is code:  [Language Processing Pipeline](LanguageProcessingPipeline.py)

**explanation**

This code demonstrates basic NLP processing with SpaCy. 
It first tokenizes the input text, which breaks it down into individual
words and punctuation. For each token, the script prints the token text, its
part-of-speech tag, and its lemma (base form). It then identifies named entities 
(like people or organizations) in the text, printing each entity's text, label,
and a description of the label. This process shows how to extract and analyze 
text.

## Stemming and Lemmatization
**Stemming:** Use fixed rules, for example talking -> talk, For that you can simply use regex and get base words.

**Lemmatization:** Whenever you need to use knowledge of language, or linguistic knowledge, you need lemmatization, base word is called lemma.

**spaCy does not support stemming, for that reason I'm going to use NLTK**

Let's compare Stemming and Lemmatization results:

### Stemming results:
![alt text](images/stem.png)

### Lemmatization results:
![alt text](images/lemma.png)

As you can see as a machine when you have linguistic knowledge you make better job ()



## Part Of Speech (POS) Tagging

**SOON 🔜** Introduction to POS tagging and its importance in NLP tasks.

## Named Entity Recognition (NER)

**SOON 🔜** Overview of Named Entity Recognition (NER) and its applications.

## Text Representation

**SOON 🔜** Various techniques for representing text data in NLP tasks.

### Label One-Hot Encoding

**SOON 🔜** Explanation and implementation of label one-hot encoding for classification tasks.

### Bag of Words

**SOON 🔜** Introduction to the Bag of Words model for text representation.

### Stop Words

**SOON 🔜** Understanding stop words and their role in NLP preprocessing.

### Text Representation using Bag of N-grams

**SOON 🔜** Using Bag of N-grams for capturing n-gram information in text.

### Using TF-IDF (Term Frequency-Inverse Document Frequency)

**SOON 🔜** Explanation and implementation of TF-IDF for text representation.

### Using Word Embeddings

**SOON 🔜** Introduction to word embeddings and their applications in NLP.

### Word Vectors in Spacy

**SOON 🔜** Understanding word vectors and their implementation using Spacy.

### Classification using Word Vectors in Spacy

**SOON 🔜** How to perform classification tasks using word vectors in Spacy.

### Word Vectors in Gensim

**SOON 🔜** Introduction to word vectors using the Gensim library in Python.


This repository aims to be a thorough, hands-on guide to Natural Language Processing. Happy learning and coding!