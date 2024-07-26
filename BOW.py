import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import classification_report

df = pd.read_csv("./data/review.csv")

# 25k positive 25k negative, balanced dataset
#print(df.sentiment.value_counts())

df['positive'] = df['sentiment'].apply(lambda x: 1 if x =='positive' else 0)

# positive 1, negative 0
#print(df.shape)
#print(df.head())

X_train, X_test, y_train, y_test = train_test_split(df.review, df.positive, test_size=0.2)

clf = Pipeline([
    ('vectorizer', CountVectorizer()), # bag of word representation using CountVectorizer, which counts words occurence
    ('nb', MultinomialNB())
])
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

print(classification_report(y_test, y_pred))


