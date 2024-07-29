import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
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


## Let's do some use case of n-grams on real dataset
df = pd.read_json('data/news_dataset.json', lines=True)
df = df.drop(columns=['authors', 'date', 'link','headline'])

#check if there is inbalance
print(df.category.value_counts()) # well there is inbalance lets set 1000 as base

unique_categories = df['category'].unique()
print(unique_categories)
print(len(unique_categories))


target_samples = 1000
sampled_dfs = []

# Sample rows for each category
for category in unique_categories:
    category_df = df[df['category'] == category]
    if len(category_df) > target_samples:
        sampled_df = category_df.sample(n=target_samples, random_state=42)
    else:
        # If there are fewer rows than target, use all available rows
        sampled_df = category_df
    sampled_dfs.append(sampled_df)

# Concatenate all sampled DataFrames, data is balanced now
balanced_df = pd.concat(sampled_dfs, ignore_index=True)
print(balanced_df.category.value_counts())
# there is 42 unique categories

# lets split for train and testing

X_train, X_test, y_train, y_test = train_test_split(df.balanced.text, df.balanced.category, test_size=0.2, random_state=42)



