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
doc = nlp(words)


def lemmatization(words):
    for token in doc:
        print(f"{token}: {token.lemma_}")

lemmatization(words) # works way better


# Make custom word for lemmatization
print()

def custom_word(slang, string):

    """
    pass one slang get corrected lemma
    :param slang: one slang word: str
    :param string: text to be lemmatized: str
    :return: None, void
    """
    # Get the attribute ruler
    ar = nlp.get_pipe('attribute_ruler')

    # Add custom rule
    ar.add([[{"TEXT":f"{slang}"}]], {"LEMMA": "Friend"})

    # Process the document
    doc = nlp(string)

    # Iterate over tokens and print their text and lemma
    for token in doc:
        print(token.text, " | ", token.lemma_)


custom_word("homie","Hey my homie, how are you doing today")





