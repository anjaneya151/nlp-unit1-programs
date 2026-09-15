"""
Program 2 : Stemming and Lemmatization on Sample Text
Course Outcome : CO1

Description
-----------
Stemming    -> chops word endings using rules (output may not be a real word).
Lemmatization -> reduces a word to its dictionary form (lemma) using vocabulary
                 and part-of-speech information.

The program compares:
    * Porter Stemmer   (NLTK)
    * Snowball Stemmer (NLTK)
    * WordNet Lemmatizer (NLTK)
    * spaCy lemmatizer

How to run
----------
    pip install nltk spacy
    python -m spacy download en_core_web_sm
    python stemming_lemmatization.py
"""

import nltk
from nltk.stem import PorterStemmer, SnowballStemmer, WordNetLemmatizer
from nltk.tokenize import word_tokenize

import spacy


TEXT = (
    "The programmers were programming happily while the studies studied by "
    "children ran better than the flies flying over the leaves."
)


def get_wordnet_pos(nltk_tag: str) -> str:
    """Map an NLTK POS tag to the tag expected by the WordNet lemmatizer."""
    if nltk_tag.startswith("J"):
        return "a"  # adjective
    if nltk_tag.startswith("V"):
        return "v"  # verb
    if nltk_tag.startswith("R"):
        return "r"  # adverb
    return "n"      # noun (default)


def stem_and_lemmatize_with_nltk(text: str) -> None:
    nltk.download("punkt", quiet=True)
    nltk.download("punkt_tab", quiet=True)
    nltk.download("wordnet", quiet=True)
    nltk.download("omw-1.4", quiet=True)
    nltk.download("averaged_perceptron_tagger", quiet=True)
    nltk.download("averaged_perceptron_tagger_eng", quiet=True)

    porter = PorterStemmer()
    snowball = SnowballStemmer("english")
    lemmatizer = WordNetLemmatizer()

    words = word_tokenize(text)
    tagged = nltk.pos_tag(words)

    print("----- NLTK: Stemming vs Lemmatization -----")
    print(f"{'WORD':<15}{'PORTER':<15}{'SNOWBALL':<15}{'LEMMA':<15}")
    print("-" * 60)
    for word, tag in tagged:
        if not word.isalpha():
            continue
        lemma = lemmatizer.lemmatize(word.lower(), pos=get_wordnet_pos(tag))
        print(
            f"{word:<15}{porter.stem(word):<15}"
            f"{snowball.stem(word):<15}{lemma:<15}"
        )
    print()


def lemmatize_with_spacy(text: str) -> None:
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(text)

    print("----- spaCy: Lemmatization -----")
    print(f"{'WORD':<15}{'LEMMA':<15}{'POS':<10}")
    print("-" * 40)
    for token in doc:
        if token.is_alpha:
            print(f"{token.text:<15}{token.lemma_:<15}{token.pos_:<10}")
    print()


def main() -> None:
    print("Input text:\n" + TEXT + "\n")
    stem_and_lemmatize_with_nltk(TEXT)
    lemmatize_with_spacy(TEXT)


if __name__ == "__main__":
    main()
