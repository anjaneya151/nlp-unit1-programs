"""
Program 1 : Tokenization of Sentences and Words using NLTK and spaCy
Course Outcome : CO1

Description
-----------
This program takes a sample paragraph and performs:
    1. Sentence tokenization  (splitting text into sentences)
    2. Word tokenization      (splitting sentences into words/tokens)
using two popular NLP libraries - NLTK and spaCy.

How to run
----------
    pip install nltk spacy
    python -m spacy download en_core_web_sm
    python tokenization_nltk_spacy.py
"""

import nltk
from nltk.tokenize import sent_tokenize, word_tokenize

import spacy


# ---------------------------------------------------------------- sample data
TEXT = (
    "Natural Language Processing is a branch of Artificial Intelligence. "
    "It helps computers understand human language. "
    "Dr. Rao teaches NLP at the university, and his classes start at 9 a.m."
)


# -------------------------------------------------------------- NLTK approach
def tokenize_with_nltk(text: str) -> None:
    """Sentence and word tokenization using NLTK."""
    # Required NLTK models (downloaded only once).
    nltk.download("punkt", quiet=True)
    nltk.download("punkt_tab", quiet=True)

    sentences = sent_tokenize(text)
    words = word_tokenize(text)

    print("----- NLTK -----")
    print(f"Number of sentences : {len(sentences)}")
    for i, sentence in enumerate(sentences, start=1):
        print(f"  Sentence {i}: {sentence}")

    print(f"\nNumber of words : {len(words)}")
    print(f"  Words: {words}\n")


# ------------------------------------------------------------- spaCy approach
def tokenize_with_spacy(text: str) -> None:
    """Sentence and word tokenization using spaCy."""
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(text)

    sentences = [sent.text for sent in doc.sents]
    words = [token.text for token in doc]

    print("----- spaCy -----")
    print(f"Number of sentences : {len(sentences)}")
    for i, sentence in enumerate(sentences, start=1):
        print(f"  Sentence {i}: {sentence}")

    print(f"\nNumber of words : {len(words)}")
    print(f"  Words: {words}\n")


def main() -> None:
    print("Input text:\n" + TEXT + "\n")
    tokenize_with_nltk(TEXT)
    tokenize_with_spacy(TEXT)


if __name__ == "__main__":
    main()
