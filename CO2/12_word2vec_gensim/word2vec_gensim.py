"""
Program 12 : Word2Vec Word Embeddings using Gensim on a Custom Corpus
Course Outcome : CO2

Aim
---
Train Word2Vec models (CBOW and Skip-gram) on a small custom corpus with Gensim
and explore the learned word vectors.

What this program does
----------------------
1. Pre-processes a custom corpus (lower-case, tokenize, remove stop-words).
2. Trains a CBOW model (sg=0) and a Skip-gram model (sg=1).
3. Shows a word vector, similar words, pairwise similarity and the odd-one-out.
4. Saves / reloads the model and draws a 2-D PCA plot of the vectors.

Note: the corpus is tiny, so the results only illustrate the method. Real
embeddings need millions of sentences (see Program 13 for pre-trained GloVe).

Run:  python word2vec_gensim.py
"""

import re

import numpy as np
from gensim.models import Word2Vec
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

# ----------------------------------------------------------------------------
# Custom corpus : three small topics (royalty, animals, technology)
# ----------------------------------------------------------------------------
corpus = [
    "The king rules the kingdom with his queen",
    "The queen and the king live in the royal palace",
    "A prince is the son of the king and the queen",
    "A princess is the daughter of the king and the queen",
    "The king and the prince protect the kingdom",
    "The queen and the princess love the royal palace",
    "The man serves the king in the palace",
    "The woman serves the queen in the palace",
    "The dog chases the cat in the garden",
    "The cat sleeps on the sofa and the dog sleeps on the floor",
    "The dog and the cat are friendly animals",
    "The puppy is a young dog and the kitten is a young cat",
    "Animals like the dog and the cat need food and water",
    "The horse and the cow live on the farm",
    "The farmer feeds the horse and the cow on the farm",
    "Python is a programming language for data science",
    "Machine learning models learn patterns from data",
    "Programmers write code in Python and test the code",
    "Data scientists use Python and machine learning to analyse data",
    "A computer runs the code and processes the data",
    "Deep learning is a part of machine learning",
    "The programmer fixes bugs in the code on the computer",
]

# Word2Vec expects a list of token lists
STOP = set(ENGLISH_STOP_WORDS) - {"king", "queen"}


def preprocess(sentence):
    tokens = re.findall(r"[a-z]+", sentence.lower())
    return [t for t in tokens if t not in STOP]


def section(title):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


sentences = [preprocess(s) for s in corpus]

section("PART A : Pre-processed corpus")
for original, tokens in list(zip(corpus, sentences))[:3]:
    print(f"   {original}\n   -> {tokens}\n")
print(f"Total sentences : {len(sentences)}")

# ----------------------------------------------------------------------------
# Part B : Train CBOW and Skip-gram
# ----------------------------------------------------------------------------
section("PART B : Training CBOW and Skip-gram models")
PARAMS = dict(vector_size=50, window=3, min_count=1, epochs=300,
              seed=42, workers=1)           # workers=1 + seed => reproducible

cbow = Word2Vec(sentences, sg=0, **PARAMS)
skipgram = Word2Vec(sentences, sg=1, **PARAMS)

print(f"Vocabulary size       : {len(cbow.wv)}")
print(f"Embedding dimension   : {cbow.wv.vector_size}")
print(f"Vocabulary (first 15) : {cbow.wv.index_to_key[:15]}")

# ----------------------------------------------------------------------------
# Part C : Word vector
# ----------------------------------------------------------------------------
section("PART C : The vector of a word")
word = "king"
print(f"Vector of '{word}' (first 10 of {cbow.wv.vector_size} values, CBOW):")
print(np.round(cbow.wv[word][:10], 4))

# ----------------------------------------------------------------------------
# Part D : Similar words & similarity scores
# ----------------------------------------------------------------------------
section("PART D : Most similar words")
for query in ["king", "dog", "python"]:
    for name, model in [("CBOW", cbow), ("Skip-gram", skipgram)]:
        if query in model.wv:
            similar = ", ".join(f"{w} ({s:.2f})" for w, s in model.wv.most_similar(query, topn=4))
            print(f"   [{name:<9}] {query:<7} -> {similar}")
    print()

section("PART E : Pairwise similarity and odd-one-out (Skip-gram)")
for a, b in [("king", "queen"), ("dog", "cat"), ("python", "code"), ("king", "dog"), ("cat", "python")]:
    print(f"   similarity({a:<7}, {b:<7}) = {skipgram.wv.similarity(a, b):.3f}")

group = ["king", "queen", "prince", "dog"]
print(f"\nOdd one out of {group} : {skipgram.wv.doesnt_match(group)}")

# ----------------------------------------------------------------------------
# Part F : Save and load
# ----------------------------------------------------------------------------
section("PART F : Saving and loading the model")
skipgram.save("word2vec_skipgram.model")
reloaded = Word2Vec.load("word2vec_skipgram.model")
same = np.allclose(reloaded.wv["king"], skipgram.wv["king"])
print(f"Model saved to word2vec_skipgram.model and reloaded. Vectors identical: {same}")

# ----------------------------------------------------------------------------
# Part G : 2-D visualisation
# ----------------------------------------------------------------------------
section("PART G : 2-D PCA plot of the embeddings")
try:
    import matplotlib
    matplotlib.use("Agg")                 # no window needed - saves to file
    import matplotlib.pyplot as plt

    words = [w for w in skipgram.wv.index_to_key[:40]]
    coords = PCA(n_components=2, random_state=42).fit_transform(
        np.array([skipgram.wv[w] for w in words]))

    plt.figure(figsize=(9, 6))
    plt.scatter(coords[:, 0], coords[:, 1], s=30)
    for w, (x, y) in zip(words, coords):
        plt.annotate(w, (x, y), fontsize=9, xytext=(3, 3), textcoords="offset points")
    plt.title("Word2Vec (Skip-gram) embeddings projected to 2-D with PCA")
    plt.tight_layout()
    plt.savefig("word2vec_plot.png", dpi=150)
    print("Plot saved as word2vec_plot.png")
except ImportError:
    print("matplotlib is not installed - skipping the plot (pip install matplotlib).")
