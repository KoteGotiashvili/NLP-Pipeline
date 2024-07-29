from sklearn.feature_extraction.text import CountVectorizer
import spacy
v = CountVectorizer(ngram_range=(1,2))
# v.fit(["doesn’t bet against deep learning. Somehow, every time you run into an obstacle, within six months or a year researchers find a way around it"])

# get splited text with count, just fin how it works
#print(v.vocabulary_)

words = [
    "Without Ilya Sutskever OpenAI will not be there today, he is genius",
    "Andrej Karpathy is genius behind the Tesla CV",
    "Without Nvidia, there will be no AI boom"
]

# preprocess text, remove stop words, do lemmatization, etc
nlp = spacy.load("en_core_web_sm")
def preprocess(text):
    doc = nlp(text)

    clean_text = []

    for token in doc:
        if token.is_stop or token.is_punct:
            continue
        clean_text.append(token.lemma_)
    return " ".join(clean_text)

words_processed = [ preprocess(text) for text in words]
# print(words_processed)
v.fit(words_processed)


# test how it turns text int ovectors
# takes all unique words from dataset and adds to column, and based on that predicts unseen data, For large datasets basically it will eat memory (:
# transformed = v.transform(["OpenAI is not open but paid (:"])
# print(transformed.toarray())
# print(transformed.shape)

