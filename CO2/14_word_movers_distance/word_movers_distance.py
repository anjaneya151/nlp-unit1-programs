"""
Program 14 : Text Similarity using Word Mover's Distance (WMD)
Course Outcome : CO2

Aim
---
Measure the semantic distance between two documents with Word Mover's Distance.

Idea
----
WMD is the minimum total "distance" the words of document A must travel in
embedding space to become the words of document B. Because it works on word
embeddings, two sentences with NO common words can still be close:

    "Obama speaks to the media in Illinois"
    "The President greets the press in Chicago"

    smaller WMD  -> more similar        larger WMD -> less similar

Requirements
------------
    pip install gensim POT          (POT = Python Optimal Transport, used by gensim)

Embeddings : pre-trained GloVe (same loading logic as Program 13). Use the
local file glove.6B.50d.txt if present, otherwise gensim downloads it.

What this program does
----------------------
1. Computes WMD for sentence pairs and compares it with cosine similarity of the
   averaged word vectors.
2. Builds a pairwise WMD distance matrix for a small set of documents.
3. Retrieves the document closest to a query.

Run:  python word_movers_distance.py
"""

import os
import re

import numpy as np
from gensim.models import KeyedVectors
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

LOCAL_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "glove.6B.50d.txt")
GENSIM_NAME = "glove-wiki-gigaword-50"


def section(title):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def load_glove():
    if os.path.exists(LOCAL_FILE):
        print(f"Loading local file : {LOCAL_FILE}")
        return KeyedVectors.load_word2vec_format(LOCAL_FILE, binary=False, no_header=True)
    print(f"Local file not found - downloading '{GENSIM_NAME}' with gensim (first run only)...")
    import gensim.downloader as api
    return api.load(GENSIM_NAME)


def preprocess(text, model):
    """Lower-case, remove stop-words and drop words that have no embedding."""
    tokens = re.findall(r"[a-z]+", text.lower())
    return [t for t in tokens if t not in ENGLISH_STOP_WORDS and t in model]


def avg_cosine(tokens_a, tokens_b, model):
    a = np.mean([model[t] for t in tokens_a], axis=0)
    b = np.mean([model[t] for t in tokens_b], axis=0)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


try:
    glove = load_glove()
except Exception as err:                       # no internet and no local file
    raise SystemExit(f"Could not load GloVe embeddings: {err}\n"
                     f"Download glove.6B.50d.txt and place it next to this script.")

# ----------------------------------------------------------------------------
# Part A : WMD between sentence pairs
# ----------------------------------------------------------------------------
section("PART A : WMD between sentence pairs")
pairs = [
    ("Obama speaks to the media in Illinois", "The President greets the press in Chicago"),
    ("Obama speaks to the media in Illinois", "The band gave a concert in Japan"),
    ("The cat sat on the mat", "A kitten rested on the rug"),
    ("The cat sat on the mat", "Stock prices fell sharply today"),
]
print(f"{'WMD':>7} {'cosine(avg)':>12}   Sentence pair")
for s1, s2 in pairs:
    t1, t2 = preprocess(s1, glove), preprocess(s2, glove)
    wmd = glove.wmdistance(t1, t2)
    cos = avg_cosine(t1, t2, glove)
    print(f"{wmd:7.3f} {cos:12.3f}   '{s1}'  <->  '{s2}'")
print("\nLower WMD = more similar. Related sentences get a clearly lower distance")
print("even when they share almost no words.")

# ----------------------------------------------------------------------------
# Part B : Distance matrix
# ----------------------------------------------------------------------------
section("PART B : Pairwise WMD matrix")
documents = [
    "The government announced a new tax policy for businesses",
    "The ministry revealed fresh economic rules for companies",
    "The football team won the championship final",
    "Our side lifted the trophy after the last match",
    "Scientists discovered a new planet near a distant star",
]
tokens = [preprocess(d, glove) for d in documents]
n = len(documents)
matrix = np.zeros((n, n))
for i in range(n):
    for j in range(i + 1, n):
        matrix[i, j] = matrix[j, i] = glove.wmdistance(tokens[i], tokens[j])

for i, d in enumerate(documents):
    print(f"   D{i + 1}: {d}")
print("\n      " + "".join(f"{'D' + str(j + 1):>8}" for j in range(n)))
for i in range(n):
    print(f"   D{i + 1} " + "".join(f"{matrix[i, j]:8.3f}" for j in range(n)))

# ----------------------------------------------------------------------------
# Part C : Nearest document to a query
# ----------------------------------------------------------------------------
section("PART C : Retrieve the closest document for a query")
queries = ["Cabinet approves budget for industries",
           "The cricket squad celebrated its victory"]
for q in queries:
    q_tokens = preprocess(q, glove)
    ranked = sorted(((glove.wmdistance(q_tokens, tokens[i]), i) for i in range(n)))
    print(f"\nQuery: '{q}'")
    for rank, (dist, i) in enumerate(ranked[:3], start=1):
        print(f"   {rank}. WMD = {dist:.3f}   D{i + 1}: {documents[i]}")
