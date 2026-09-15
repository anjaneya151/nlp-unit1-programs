"""
Program 5 : Parsing and Chunking using RegEx and spaCy
Course Outcome : CO1

Description
-----------
Chunking (shallow parsing) groups POS-tagged words into phrases such as
Noun Phrases (NP), Verb Phrases (VP) and Prepositional Phrases (PP).

Part A : Regular-expression chunk grammar with NLTK's RegexpParser.
Part B : Dependency parsing + noun-chunk extraction with spaCy.

How to run
----------
    pip install nltk spacy
    python -m spacy download en_core_web_sm
    python parsing_chunking.py
"""

import nltk
from nltk import RegexpParser
from nltk.tokenize import word_tokenize

import spacy


SENTENCE = (
    "The talented young students of the university are building "
    "a powerful chatbot with modern machine learning tools."
)

# Chunk grammar:
#   NP -> optional determiner + any adjectives + one or more nouns
#   PP -> preposition followed by a noun phrase
#   VP -> verb(s) followed by a noun phrase or prepositional phrase
CHUNK_GRAMMAR = r"""
    NP: {<DT>?<JJ.*>*<NN.*>+}
    PP: {<IN><NP>}
    VP: {<VB.*>+<NP|PP>?}
"""


def chunk_with_regex(sentence: str) -> None:
    nltk.download("punkt", quiet=True)
    nltk.download("punkt_tab", quiet=True)
    nltk.download("averaged_perceptron_tagger", quiet=True)
    nltk.download("averaged_perceptron_tagger_eng", quiet=True)

    tokens = word_tokenize(sentence)
    tagged = nltk.pos_tag(tokens)

    parser = RegexpParser(CHUNK_GRAMMAR)
    tree = parser.parse(tagged)

    print("----- RegEx chunking (NLTK) -----")
    print("POS tags:", tagged, "\n")
    print("Parse tree:")
    print(tree, "\n")

    for label in ("NP", "VP", "PP"):
        phrases = [
            " ".join(word for word, _ in subtree.leaves())
            for subtree in tree.subtrees()
            if subtree.label() == label
        ]
        print(f"{label} chunks: {phrases}")
    print()

    # tree.draw()  # uncomment to open a graphical parse-tree window


def parse_with_spacy(sentence: str) -> None:
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(sentence)

    print("----- Dependency parsing (spaCy) -----")
    print(f"{'TOKEN':<14}{'POS':<8}{'DEP':<12}{'HEAD':<14}")
    print("-" * 50)
    for token in doc:
        print(f"{token.text:<14}{token.pos_:<8}{token.dep_:<12}{token.head.text:<14}")

    print("\nNoun chunks found by spaCy:")
    for chunk in doc.noun_chunks:
        print(f"  {chunk.text:<40} (root: {chunk.root.text}, dep: {chunk.root.dep_})")
    print()


def main() -> None:
    print("Input sentence:\n" + SENTENCE + "\n")
    chunk_with_regex(SENTENCE)
    parse_with_spacy(SENTENCE)


if __name__ == "__main__":
    main()
