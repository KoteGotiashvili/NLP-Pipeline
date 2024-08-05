import spacy
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report



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
# print(df['category'].value_counts())

# Determine the number of samples for each category
min_samples = 100

# Sample from each category to get exactly min_samples
df_balanced = df.groupby('category').apply(lambda x: x.sample(min(len(x), min_samples))).reset_index(drop=True)


df_balanced = df_balanced.drop(columns=['category'])

df_balanced['vector'] = df_balanced['headline'].apply(lambda x: nlp(x).vector)


X_train, X_test, y_train, y_test = train_test_split(
    df_balanced.vector.values,
    df_balanced.label,
    test_size=0.2,
    random_state=2022
)
X_train_2d = np.stack(X_train)
X_test_2d = np.stack(X_test)



scaler = MinMaxScaler()
scaled_train_embed = scaler.fit_transform(X_train_2d)
scaled_test_embed = scaler.transform(X_test_2d)


clf = MultinomialNB()
clf.fit(scaled_train_embed, y_train)
y_pred = clf.predict(scaled_test_embed)

print(classification_report(y_test, y_pred))





