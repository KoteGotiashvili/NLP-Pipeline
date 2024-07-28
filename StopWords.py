import spacy
from spacy.lang.en.stop_words import STOP_WORDS

# print all stop words
print(STOP_WORDS)

nlp = spacy.load("en_core_web_sm")

quote_from_sutskever ="It may be that today's large neural networks are slightly conscious."
doc = nlp(quote_from_sutskever)

#pritn stop words
for token in doc:
    if token.is_stop:
        pass
        #print(token.text)

def preprocess(text):
    doc = nlp(text)

    no_stop_words = [token.text for token in doc if not token.is_stop and not token.is_punct]
    return " ".join(no_stop_words)

def print_percentage_stopwords(before, after):
    before_len = len(before)
    after_len = before_len - len(after)
    return (after_len/before_len) * 100

print("before")
print(quote_from_sutskever)
print("after")
print(preprocess(quote_from_sutskever))
print("percentage of stop words")
print(print_percentage_stopwords(quote_from_sutskever, preprocess(quote_from_sutskever)))