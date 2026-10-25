"""
Program 13 : GloVe Embeddings Loading and Vector Representation
Course Outcome : CO2

Aim
---
Load pre-trained GloVe word embeddings and use them to represent words and
sentences as vectors.

How the embeddings are loaded
-----------------------------
Option 1 (automatic) : gensim downloads "glove-wiki-gigaword-50" (about 65 MB,
                       50-dimensional GloVe trained on Wikipedia + Gigaword).
                       Internet is needed only the first time.
Option 2 (offline)   : put the Stanford file  glove.6B.50d.txt  in this folder
                       (download glove.6B.zip from https://nlp.stanford.edu/projects/glove/).
                       The program uses the local file if it exists.

What this program does
----------------------
1. Loads GloVe and inspects the format of the vectors.
2. Gets the vector of a word, similar words and similarity scores.
3. Solves word analogies (king - man + woman = queen).
4. Builds sentence vectors by averaging word vectors and compares sentences.
5. Handles out-of-vocabulary (OOV) words.

Run:  python glove_embeddings.py
"""

import os
import re

import numpy as np
from gensim.models import KeyedVectors

LOCAL_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "glove.6B.50d.txt")
GENSIM_NAME = "glove-wiki-gigaword-50"


def section(title):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def load_glove():
    """Load GloVe vectors from a local file if present, else via gensim downloader."""
    if os.path.exists(LOCAL_FILE):
        print(f"Loading local file : {LOCAL_FILE}")
        # GloVe text files have no header line, hence no_header=True
        return KeyedVectors.load_word2vec_format(LOCAL_FILE, binary=False, no_header=True)
    print(f"Local file not found - downloading '{GENSIM_NAME}' with gensim (first run only)...")
    import gensim.downloader as api
    return api.load(GENSIM_NAME)


def cosine(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def sentence_vector(sentence, glove):
    """Average of the GloVe vectors of all in-vocabulary words of a sentence."""
    words = [w for w in re.findall(r"[a-z]+", sentence.lower()) if w in glove]
    if not words:
        return np.zeros(glove.vector_size)
    return np.mean([glove[w] for w in words], axis=0)


glove = load_glove()

# ----------------------------------------------------------------------------
# Part A : Basic information
# ----------------------------------------------------------------------------
section("PART A : Loaded embeddings")
print(f"Vocabulary size     : {len(glove):,} words")
print(f"Embedding dimension : {glove.vector_size}")
print(f"First 10 words      : {glove.index_to_key[:10]}  (ordered by frequency)")

# ----------------------------------------------------------------------------
# Part B : Vector of a word
# ----------------------------------------------------------------------------
section("PART B : Vector representation of a word")
word = "computer"
vec = glove[word]
print(f"Word            : {word}")
print(f"Vector shape    : {vec.shape}")
print(f"First 10 values : {np.round(vec[:10], 4)}")
print(f"Vector length   : {np.linalg.norm(vec):.4f}")

# ----------------------------------------------------------------------------
# Part C : Similar words and similarity scores
# ----------------------------------------------------------------------------
section("PART C : Similar words and similarity scores")
for query in ["india", "python", "happy"]:
    similar = ", ".join(f"{w} ({s:.2f})" for w, s in glove.most_similar(query, topn=5))
    print(f"   {query:<8} -> {similar}")

print()
for a, b in [("king", "queen"), ("cat", "dog"), ("car", "truck"), ("car", "banana")]:
    print(f"   similarity({a:<6}, {b:<7}) = {glove.similarity(a, b):.3f}")

# ----------------------------------------------------------------------------
# Part D : Analogies
# ----------------------------------------------------------------------------
section("PART D : Word analogies   A - B + C = ?")
analogies = [("king", "man", "woman"), ("paris", "france", "italy"),
             ("walking", "walk", "swim")]
for a, b, c in analogies:
    result = glove.most_similar(positive=[a, c], negative=[b], topn=3)
    answer = ", ".join(f"{w} ({s:.2f})" for w, s in result)
    print(f"   {a} - {b} + {c}  ->  {answer}")

# ----------------------------------------------------------------------------
# Part E : Sentence vectors
# ----------------------------------------------------------------------------
section("PART E : Sentence representation (average of word vectors)")
sentences = [
    "The cat sits on the mat",
    "A kitten is resting on the rug",
    "The stock market crashed today",
    "Investors lost money as shares fell",
]
vectors = [sentence_vector(s, glove) for s in sentences]
for i, s in enumerate(sentences):
    print(f"   S{i + 1}: {s}")
print("\nCosine similarity between sentence vectors:")
for i in range(len(sentences)):
    for j in range(i + 1, len(sentences)):
        print(f"   S{i + 1} - S{j + 1} : {cosine(vectors[i], vectors[j]):.3f}")

# ----------------------------------------------------------------------------
# Part F : Out-of-vocabulary words
# ----------------------------------------------------------------------------
section("PART F : Out-of-vocabulary (OOV) words")
for w in ["language", "xyzzyqwerty"]:
    print(f"   '{w}' in vocabulary : {w in glove}")
print("\nGloVe has a fixed vocabulary: unknown words have no vector, so they must be")
print("skipped (as done in sentence_vector) or replaced by a zero / random vector.")
