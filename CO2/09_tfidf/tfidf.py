"""
Program 9 : TF-IDF Implementation and Comparison with BoW
Course Outcome : CO2

Aim
---
Implement TF-IDF, verify it against scikit-learn, and compare it with Bag-of-Words.

What this program does
----------------------
1. Computes TF, IDF and TF-IDF step by step from scratch (same formula as scikit-learn).
2. Verifies the result with TfidfVectorizer.
3. Compares BoW and TF-IDF: common words lose weight, rare/informative words gain weight.

Formulae
--------
    TF(t, d)     = number of times term t occurs in document d
    IDF(t)       = ln((1 + N) / (1 + df(t))) + 1          (smoothed IDF)
    TF-IDF(t, d) = TF(t, d) * IDF(t)   -> then L2-normalised per document

Run:  python tfidf.py
"""

import math
import re
from collections import Counter

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

corpus = [
    "The cat sat on the mat.",
    "The dog sat on the log.",
    "The cats and dogs are common pets.",
    "The stock market fell sharply on Monday.",
]

TOKEN_PATTERN = r"(?u)\b[a-zA-Z]+\b"


def tokenize(text):
    return re.findall(TOKEN_PATTERN, text.lower())


def section(title):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


docs = [tokenize(d) for d in corpus]
vocab = sorted({w for d in docs for w in d})
N = len(docs)
idx = [f"Doc{i + 1}" for i in range(N)]

pd.set_option("display.width", 250)
pd.set_option("display.max_columns", None)

# ----------------------------------------------------------------------------
# Part A : TF
# ----------------------------------------------------------------------------
section("PART A : Term Frequency (TF) - raw counts")
tf = np.array([[Counter(d)[w] for w in vocab] for d in docs], dtype=float)
print(pd.DataFrame(tf.astype(int), columns=vocab, index=idx).to_string())

# ----------------------------------------------------------------------------
# Part B : IDF
# ----------------------------------------------------------------------------
section("PART B : Inverse Document Frequency (IDF)")
df_count = np.array([sum(1 for d in docs if w in d) for w in vocab])
idf = np.array([math.log((1 + N) / (1 + c)) + 1 for c in df_count])
idf_table = pd.DataFrame({"word": vocab, "doc_freq": df_count, "idf": idf.round(4)})
print(idf_table.sort_values("idf").to_string(index=False))
print("\nA word that appears in many documents (e.g. 'the', 'on') gets a LOW idf;")
print("a word that appears in only one document gets a HIGH idf.")

# ----------------------------------------------------------------------------
# Part C : TF-IDF
# ----------------------------------------------------------------------------
section("PART C : TF-IDF (from scratch, L2-normalised)")
tfidf_raw = tf * idf
norms = np.linalg.norm(tfidf_raw, axis=1, keepdims=True)
tfidf_scratch = tfidf_raw / norms
print(pd.DataFrame(tfidf_scratch.round(3), columns=vocab, index=idx).to_string())

# ----------------------------------------------------------------------------
# Part D : scikit-learn verification
# ----------------------------------------------------------------------------
section("PART D : Verification with scikit-learn TfidfVectorizer")
tfidf_vec = TfidfVectorizer(token_pattern=TOKEN_PATTERN)
X_tfidf = tfidf_vec.fit_transform(corpus)
print(f"Same vocabulary         : {list(tfidf_vec.get_feature_names_out()) == vocab}")
print(f"Scratch == scikit-learn : {np.allclose(X_tfidf.toarray(), tfidf_scratch)}")

# ----------------------------------------------------------------------------
# Part E : Comparison with BoW
# ----------------------------------------------------------------------------
section("PART E : Comparison of BoW and TF-IDF")
count_vec = CountVectorizer(token_pattern=TOKEN_PATTERN)
X_bow = count_vec.fit_transform(corpus)
features = count_vec.get_feature_names_out()

for i in (0, 3):
    print(f"\nDocument {i + 1}: {corpus[i]}")
    bow_row = X_bow.toarray()[i]
    tfidf_row = X_tfidf.toarray()[i]
    rows = [(features[j], bow_row[j], tfidf_row[j]) for j in range(len(features)) if bow_row[j] > 0]
    table = pd.DataFrame(rows, columns=["word", "BoW count", "TF-IDF weight"])
    table["TF-IDF weight"] = table["TF-IDF weight"].round(3)
    print(table.sort_values("TF-IDF weight", ascending=False).to_string(index=False))

# How much of document 1's total weight does the stop-word 'the' take?
word = "the"
j = list(features).index(word)
bow_row = X_bow.toarray()[0]
tfidf_row = X_tfidf.toarray()[0]
bow_share = 100 * bow_row[j] / bow_row.sum()
tfidf_share = 100 * tfidf_row[j] / tfidf_row.sum()
print(f"\nShare of Doc 1's total weight taken by '{word}':")
print(f"   Bag-of-Words : {bow_share:.1f}%")
print(f"   TF-IDF       : {tfidf_share:.1f}%")
print("\n'the' occurs in every document, so its IDF is the lowest possible and its share drops.")
print("Words that are specific to few documents (e.g. 'cat', 'mat') gain relative weight.")

section("PART F : Summary of differences")
summary = pd.DataFrame({
    "Aspect": ["Value stored", "Common words", "Rare words", "Document length", "Best used for"],
    "Bag-of-Words": ["Raw word count", "High weight", "Same as any other word",
                     "Longer docs -> bigger numbers", "Simple baselines"],
    "TF-IDF": ["Count x rarity", "Down-weighted", "Up-weighted",
               "Normalised (L2)", "Search, classification, similarity"],
})
print(summary.to_string(index=False))
