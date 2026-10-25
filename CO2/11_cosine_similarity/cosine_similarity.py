"""
Program 11 : Cosine Similarity Computation between Text Documents
Course Outcome : CO2

Aim
---
Measure how similar two text documents are using cosine similarity on their
vector representations (BoW and TF-IDF).

    cosine(A, B) = (A . B) / (||A|| * ||B||)

    1.0 -> same direction (very similar)      0.0 -> nothing in common

What this program does
----------------------
1. Computes cosine similarity from scratch with NumPy.
2. Builds the full document-to-document similarity matrix (BoW vs TF-IDF).
3. Finds the most / least similar pair of documents.
4. Ranks all documents against a query.

Run:  python cosine_similarity.py
"""

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity as sk_cosine

documents = {
    "D1": "Python is a popular programming language for data science.",
    "D2": "Data science uses Python and machine learning to analyse data.",
    "D3": "Machine learning algorithms learn patterns from data.",
    "D4": "The cricket team won the match in the final over.",
    "D5": "The football team lost the final match of the season.",
    "D6": "Chefs cook pasta with tomato sauce and fresh basil.",
}
names = list(documents.keys())
texts = list(documents.values())


def section(title):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def cosine_scratch(a, b):
    """Cosine similarity of two 1-D vectors."""
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / denom) if denom else 0.0


pd.set_option("display.width", 200)

# ----------------------------------------------------------------------------
# Part A : Cosine similarity of two documents
# ----------------------------------------------------------------------------
section("PART A : Cosine similarity of two documents (from scratch)")
tfidf = TfidfVectorizer(stop_words="english")
X = tfidf.fit_transform(texts).toarray()

pairs = [("D1", "D2"), ("D1", "D3"), ("D4", "D5"), ("D1", "D4")]
for a, b in pairs:
    i, j = names.index(a), names.index(b)
    scratch = cosine_scratch(X[i], X[j])
    library = sk_cosine(X[i:i + 1], X[j:j + 1])[0][0]
    print(f"cos({a}, {b}) = {scratch:.4f}   (scikit-learn: {library:.4f})")
    print(f"     {a}: {documents[a]}")
    print(f"     {b}: {documents[b]}\n")

# ----------------------------------------------------------------------------
# Part B : Similarity matrix
# ----------------------------------------------------------------------------
section("PART B : Similarity matrix - TF-IDF vectors")
sim_tfidf = sk_cosine(X)
print(pd.DataFrame(sim_tfidf, index=names, columns=names).round(3).to_string())

section("PART B2 : Similarity matrix - Bag-of-Words vectors")
bow = CountVectorizer(stop_words="english").fit_transform(texts)
sim_bow = sk_cosine(bow)
print(pd.DataFrame(sim_bow, index=names, columns=names).round(3).to_string())

# ----------------------------------------------------------------------------
# Part C : Most / least similar pairs
# ----------------------------------------------------------------------------
section("PART C : Most similar and least similar pair (TF-IDF)")
pair_scores = [(names[i], names[j], sim_tfidf[i, j])
               for i in range(len(names)) for j in range(i + 1, len(names))]
pair_scores.sort(key=lambda p: p[2], reverse=True)
print("Top 3 most similar pairs:")
for a, b, s in pair_scores[:3]:
    print(f"   {a} - {b}  : {s:.4f}")
print("Least similar pairs (score 0 = no shared words):")
for a, b, s in pair_scores[-3:]:
    print(f"   {a} - {b}  : {s:.4f}")

# ----------------------------------------------------------------------------
# Part D : Rank documents against a query
# ----------------------------------------------------------------------------
section("PART D : Ranking documents for a query")
query = "machine learning with python for data"
q_vec = tfidf.transform([query])
scores = sk_cosine(q_vec, X)[0]
print(f"Query: '{query}'\n")
for rank, i in enumerate(np.argsort(-scores), start=1):
    print(f"   {rank}. {names[i]}  score = {scores[i]:.4f}   {texts[i]}")

# ----------------------------------------------------------------------------
# Part E : Why cosine and not Euclidean distance?
# ----------------------------------------------------------------------------
section("PART E : Cosine similarity ignores document length")
short = "machine learning"
long_ = "machine learning " * 10
cv = CountVectorizer().fit([short, long_])
v1, v2 = cv.transform([short]).toarray()[0], cv.transform([long_]).toarray()[0]
print(f"Short document vector : {v1}")
print(f"Long document vector  : {v2}")
print(f"Cosine similarity     : {cosine_scratch(v1, v2):.4f}   (same topic -> 1.0)")
print(f"Euclidean distance    : {np.linalg.norm(v1 - v2):.4f}   (large only because of length)")
