import spacy
from nltk.stem import PorterStemmer

stemmer = PorterStemmer()

words = ["Running", "Bigger", "Wolves", "Sang", "Children", "Better", "Cacti", "Mice", "Went", "Happier"]

def word_stemmer(words):
    """
    This function takes a list of words and applies the Porter Stemmer to each word.

    :param words: A list of words to be stemmed.
    :return: A list of stemmed words.
    """
    for word in words:
        print(f"{word}: {stemmer.stem(word)}")


word_stemmer(words) # does not work well, because it uses fixed words, using stemmer is just efficient it is faster, does not need language knowledge

# Lemmatization using spaCy
words ="Running Bigger Wolves Sang Children Better Cacti Mice Went Happier"

print()
nlp = spacy.load("en_core_web_sm")

def lemmatization(words):
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(words)
    for token in doc:
        print(f"{token}: {token.lemma_}")

lemmatization(words) # works way better


