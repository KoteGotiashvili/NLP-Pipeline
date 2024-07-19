import spacy
from spacy.symbols import ORTH
quote_from_sutskever = ("The thing you really want is for the human teachers that teach the AI to collaborate with an AI. "
                        "You might want to think of it as being in a world where the human teachers do 1% of the work and the "
                        "AI does 99% of the work. You don't want it to be 100% AI. But you do want it to be a "
                        "human-machine collaboration, which teaches the next machine. ")

nlp = spacy.blank('en')
nlp.add_pipe('sentencizer')
doc = nlp(quote_from_sutskever)


def word_tokenization(doc):
    """"
        Tokenize the input text into individual words.

        Args:
            doc (str): The input doc.

        Returns:
            list: A list of words in the input text.
    """
    return [token.text for token in doc]


def sentence_tokenization(doc):
    """"
        Tokenize the input text into individual sentences.

        Args:
            doc (str): The input doc.

        Returns:
            list: A list of sentences in the input text.
    """
    return [sent.text for sent in doc.sents]


# word tokenizer
#print(word_tokenization(quote_from_sutskever))

# sentence tokenizer
#print(sentence_tokenization(doc))


