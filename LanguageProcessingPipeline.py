# comes after tokenization

import spacy

nlp = spacy.load("en_core_web_sm")

#print(nlp.pipeline) checking existing pipelines
quote_from_sutskever = ("I lead a very simple life. I go to work; then I go home. I don’t do much else. "
                        "There are a lot of social activities one could engage in, lots of events one could go to. "
                        "Which I don’t. This is not part of quote I just add it for ner(Name Entity Recognition), I worked at Google, I'm "
                        "Founder of OpenAI and chief scientist and now I'm starting new company ")

doc = nlp(quote_from_sutskever)

def language_processing(doc):

    for token in doc:
        print(token, " | ", token.pos_," | ", token.lemma_, " | " )
    print()
    for ent in doc.ents:
        print(ent.text, " | ", ent.label_ ," | ", spacy.explain(ent.label_))



language_processing(doc)