from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
sentences = [
    "Ilya Sutskever is a leading figure in deep learning and co-founder of OpenAI.",
    "OpenAI developed the GPT-3 model, which has revolutionized natural language processing.",
    "Elon Musk is known for his work with Tesla and SpaceX, and he was a co-founder of OpenAI.",
    "Andrej Karpathy previously served as the Director of AI at Tesla, focusing on autonomous driving.",
    "Mark Zuckerberg's Meta is heavily investing in AI and virtual reality technologies.",
    "NVIDIA's CEO Jensen Huang has been pivotal in advancing GPU technology for AI applications.",
    "Machine learning algorithms are becoming increasingly sophisticated, with new models emerging regularly.",
    "AI research is advancing rapidly, leading to breakthroughs in natural language understanding and computer vision.",
    "Neural networks are the backbone of many modern AI systems, enabling complex pattern recognition.",
    "Tech companies are leveraging AI to enhance their products and services, driving innovation across industries."
]

# lets create tf idf vectorizer
# v = TfidfVectorizer()
# tranformed = v.fit_transform(sentences)
#
# all_feature_names = v.get_feature_names_out()
#
# for word in all_feature_names:
#     # let's get the index in the vocabulary
#     indx = v.vocabulary_.get(word)
#
#     # get the score
#     idf_score = v.idf_[indx]
#
#     # print(f"{word} : {idf_score}")


## lets work on real dataset using TF IDF

df = pd.read_csv('./data/emotions.csv')
# sadness (0), joy (1), love (2), anger (3), fear (4), and surprise (5).
#print(df.label.value_counts())
def normalize_labels(df, target_count=14500):
    # Create an empty DataFrame to store the normalized data
    normalized_df = pd.DataFrame()

    # Iterate through each unique label
    for label in df['label'].unique():
        # Filter the rows with the current label
        label_df = df[df['label'] == label]

        # Sample the rows to reach the target count
        if len(label_df) > target_count:
            # Randomly sample the rows if the label count is greater than the target count
            sampled_df = label_df.sample(n=target_count, random_state=1)
        else:
            # If the count is less than the target count, use all rows
            sampled_df = label_df

        # Append the sampled data to the normalized DataFrame
        normalized_df = pd.concat([normalized_df, sampled_df], ignore_index=True)

    return normalized_df

df = normalize_labels(df, target_count=14500)
print(df.head())
