import spacy
from spacy.tokens import Span
nlp = spacy.load('en_core_web_sm')

def get_ents(text):
    doc = nlp(text)
    for ent in doc.ents:
        print(ent.text, " |", ent.label_," |", spacy.explain(ent.label_))

# get ents from text
#get_ents("Ilya Sutskever, co-founder of OpenAI, has published over 50 research papers and holds multiple AI patents.")




