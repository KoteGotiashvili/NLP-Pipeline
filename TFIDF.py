from sklearn.feature_extraction.text import TfidfVectorizer

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
