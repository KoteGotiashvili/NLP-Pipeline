import spacy
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder


nlp = spacy.load("en_core_web_lg")

#doc = nlp("workout running push-ups sleeping playing gamer AI ML ")

#for token in doc:
#    print(token.text, "Vector: ", token.has_vector, "OOV: ", token.is_oov)

#print("print vector and shape")
#print(doc[0].vector, doc[0].vector.shape)

# def check_similarity(doc, base_token):
#     for token in doc:
#         print(f"{token.text} | {base_token.text}: Get similarity: ", token.similarity(base_token))\
#
# base_token = nlp("program")
#
# doc = nlp("workout running pushups sleep playing gamer AI ML food ")
#
# check_similarity(doc, base_token)

## Let's clasify news dataset using WordEmbedding
df = pd.read_json('data/news_dataset.json', lines=True)
df = df.drop(columns=['authors', 'date', 'link', "short_description"])

# Initialize the LabelEncoder
label_encoder = LabelEncoder()

df['label'] = label_encoder.fit_transform(df['category'])

df = df.drop(columns=['category'])




