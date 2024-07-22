import spacy

nlp = spacy.load("en_core_web_sm")



def get_pos_tag(text):
    """
    Get the part of speech tag for each word in the input text.

    Args
    :param text:
    :return:
    """
    doc = nlp(text)
    for token in doc:
        print(token.text, " | ", token.pos_," | ", spacy.explain(token.pos_), " | ", token.tag_," | ", spacy.explain(token.tag_))

#get_pos_tag("It's only when the dreams go into deemed factually incorrect territory that we label it a hallucination. "
 #             "It looks like a bug, but it's just the LLM doing what it always does")

# lets look at how it can help
def clean_text_using_post(text):
    doc = nlp(text)
    filtered_tokens = []
    count = doc.count_by(spacy.attrs.POS)
    for token in doc:
        if token.pos_ not in ["SPACE", "X", "PUNCT"]:
            filtered_tokens.append(token)
    return filtered_tokens, count

# lets clean this text from unnecesarry data
filtered =clean_text_using_post("OpenAI is on pace for $3.4 billion of annual revenue, Chief Executive Officer Sam Altman told the "
                      "company’s staff, according to a person familiar with the matter."
                      "In an all-hands meeting Wednesday, Altman said the vast majority of that revenue — "
                      "about $3.2 billion — comes from OpenAI’s products and services, according to the person, who "
                      "spoke on condition of anonymity to discuss internal communications. Altman said OpenAI is also on "
                      "track to generate about $200 million by offering access to its AI models through Microsoft Azure, "
                      "the person said.")
# text is cleaned now
# getting numerical values on POS tag but can be accessed which one is which
print(filtered)


