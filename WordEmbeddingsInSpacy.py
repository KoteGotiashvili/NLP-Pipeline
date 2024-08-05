import spacy

nlp = spacy.load("en_core_web_lg")

#doc = nlp("workout running push-ups sleeping playing gamer AI ML ")

#for token in doc:
#    print(token.text, "Vector: ", token.has_vector, "OOV: ", token.is_oov)

#print("print vector and shape")
#print(doc[0].vector, doc[0].vector.shape)

def check_similarity(doc, base_token):
    for token in doc:
        print(f"{token.text} | {base_token.text}: Get similarity: ", token.similarity(base_token))\

base_token = nlp("program")

doc = nlp("workout running pushups sleep playing gamer AI ML food ")

check_similarity(doc, base_token)


