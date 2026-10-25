"""
Program 8 : Bag-of-Words (BoW) Vectorization and Representation
Course Outcome : CO2

Aim
---
Convert text documents into numerical vectors using the Bag-of-Words model.

What this program does
----------------------
1. Builds a BoW matrix from scratch (pure Python) so the idea is clear.
2. Builds the same matrix with scikit-learn's CountVectorizer and verifies both match.
3. Shows binary BoW, stop-word removal and sparsity of the matrix.
4. Converts a brand-new sentence into a vector using the learned vocabulary.

Run:  python bag_of_words.py
"""

import re
from collections import Counter

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

# ----------------------------------------------------------------------------
# Sample corpus
# ----------------------------------------------------------------------------
corpus = [
    "Natural language processing helps computers understand human language.",
    "Machine learning models learn patterns from data.",
    "Language models are trained on large text data.",
    "Computers process data and learn from the data.",
]

TOKEN_PATTERN = r"(?u)\b[a-zA-Z]+\b"      # words made only of letters


def preprocess(text):
    """Lower-case the text and split it into word tokens."""
    return re.findall(TOKEN_PATTERN, text.lower())


def section(title):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


# ----------------------------------------------------------------------------
# Part A : BoW from scratch
# ----------------------------------------------------------------------------
section("PART A : Bag-of-Words from scratch")

tokenized_docs = [preprocess(doc) for doc in corpus]
vocabulary = sorted({word for doc in tokenized_docs for word in doc})
print(f"Vocabulary size : {len(vocabulary)}")
print(f"Vocabulary      : {vocabulary}\n")

bow_scratch = []
for tokens in tokenized_docs:
    counts = Counter(tokens)
    bow_scratch.append([counts[word] for word in vocabulary])
bow_scratch = np.array(bow_scratch)

df_scratch = pd.DataFrame(bow_scratch, columns=vocabulary,
                          index=[f"Doc{i + 1}" for i in range(len(corpus))])
with pd.option_context("display.max_columns", None, "display.width", 250):
    print(df_scratch)

# ----------------------------------------------------------------------------
# Part B : BoW using scikit-learn
# ----------------------------------------------------------------------------
section("PART B : Bag-of-Words using scikit-learn CountVectorizer")

vectorizer = CountVectorizer(token_pattern=TOKEN_PATTERN)
X = vectorizer.fit_transform(corpus)
features = vectorizer.get_feature_names_out()

print(f"Matrix shape (documents x vocabulary) : {X.shape}")
print(f"Same vocabulary as scratch version    : {list(features) == vocabulary}")
print(f"Same matrix as scratch version        : {np.array_equal(X.toarray(), bow_scratch)}")

# ----------------------------------------------------------------------------
# Part C : Representation of one document
# ----------------------------------------------------------------------------
section("PART C : Representation of a single document")

doc_id = 3
print(f"Document {doc_id + 1}: {corpus[doc_id]}")
vector = X.toarray()[doc_id]
print("Non-zero entries (word -> count):")
for word, count in zip(features, vector):
    if count > 0:
        print(f"   {word:<12} -> {count}")

# ----------------------------------------------------------------------------
# Part D : Variations (binary BoW, stop-word removal)
# ----------------------------------------------------------------------------
section("PART D : Binary BoW and stop-word removal")

binary_vec = CountVectorizer(token_pattern=TOKEN_PATTERN, binary=True)
Xb = binary_vec.fit_transform(corpus)
print("Binary BoW (1 = word present, 0 = absent):")
print(pd.DataFrame(Xb.toarray(), columns=binary_vec.get_feature_names_out(),
                   index=[f"Doc{i + 1}" for i in range(len(corpus))]).to_string())

stop_vec = CountVectorizer(token_pattern=TOKEN_PATTERN, stop_words="english")
Xs = stop_vec.fit_transform(corpus)
print(f"\nVocabulary size without stop-words : {Xs.shape[1]}  (with stop-words: {X.shape[1]})")
print(f"Vocabulary without stop-words      : {list(stop_vec.get_feature_names_out())}")

# ----------------------------------------------------------------------------
# Part E : Word frequencies and sparsity
# ----------------------------------------------------------------------------
section("PART E : Corpus-level word frequencies and sparsity")

total_counts = np.asarray(X.sum(axis=0)).ravel()
top = sorted(zip(features, total_counts), key=lambda p: (-p[1], p[0]))[:7]
print("Top 7 most frequent words in the corpus:")
for word, count in top:
    print(f"   {word:<12} {count}")

sparsity = 100 * (1 - X.count_nonzero() / (X.shape[0] * X.shape[1]))
print(f"\nSparsity of the BoW matrix : {sparsity:.1f}% of the entries are zero")

# ----------------------------------------------------------------------------
# Part F : Vectorize a new, unseen sentence
# ----------------------------------------------------------------------------
section("PART F : Vectorizing a new sentence with the learned vocabulary")

new_sentence = "Computers learn language from data and new data"
new_vec = vectorizer.transform([new_sentence]).toarray()[0]
print(f"New sentence : {new_sentence}")
for word, count in zip(features, new_vec):
    if count > 0:
        print(f"   {word:<12} -> {count}")
print("\nNote: words that are not in the training vocabulary (e.g. 'new') are ignored.")
