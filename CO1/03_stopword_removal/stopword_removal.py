"""
Program 3 : Stop-word Removal from a Document
Course Outcome : CO1

Description
-----------
Stop words are very frequent words (the, is, at, which ...) that usually carry
little meaning. Removing them reduces noise before further NLP processing.

Three approaches are shown:
    1. Custom stop-word list (no external library)
    2. NLTK stop-word corpus
    3. spaCy built-in stop words

The document is read from `sample_document.txt` placed next to this file.

How to run
----------
    pip install nltk spacy
    python -m spacy download en_core_web_sm
    python stopword_removal.py
"""

import os
import re
from collections import Counter

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

import spacy


DOC_PATH = os.path.join(os.path.dirname(__file__), "sample_document.txt")

# A small hand-written list, used for the "without library" approach.
CUSTOM_STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "if", "of", "at", "by", "for",
    "with", "about", "into", "to", "from", "in", "on", "is", "am", "are",
    "was", "were", "be", "been", "being", "it", "its", "this", "that",
    "these", "those", "as", "so", "such", "than", "too", "very", "can",
    "will", "just", "not", "no", "we", "you", "they", "he", "she", "i",
}


def read_document(path: str) -> str:
    with open(path, "r", encoding="utf-8") as file_handle:
        return file_handle.read()


def remove_with_custom_list(text: str) -> list:
    words = re.findall(r"[a-zA-Z']+", text.lower())
    return [word for word in words if word not in CUSTOM_STOPWORDS]


def remove_with_nltk(text: str) -> list:
    nltk.download("punkt", quiet=True)
    nltk.download("punkt_tab", quiet=True)
    nltk.download("stopwords", quiet=True)

    stop_words = set(stopwords.words("english"))
    words = word_tokenize(text.lower())
    return [word for word in words if word.isalpha() and word not in stop_words]


def remove_with_spacy(text: str) -> list:
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(text)
    return [token.text for token in doc if token.is_alpha and not token.is_stop]


def report(title: str, original_count: int, filtered: list) -> None:
    print(f"----- {title} -----")
    print(f"Tokens after removal : {len(filtered)} (removed "
          f"{original_count - len(filtered)} stop words)")
    print(f"Filtered tokens      : {filtered}")
    print(f"Top 5 frequent words : {Counter(filtered).most_common(5)}\n")


def main() -> None:
    text = read_document(DOC_PATH)
    print("Original document:\n" + text.strip() + "\n")

    total_tokens = len(re.findall(r"[a-zA-Z']+", text))
    print(f"Total word tokens in document: {total_tokens}\n")

    report("Custom list (no library)", total_tokens, remove_with_custom_list(text))
    report("NLTK stop words", total_tokens, remove_with_nltk(text))
    report("spaCy stop words", total_tokens, remove_with_spacy(text))


if __name__ == "__main__":
    main()
