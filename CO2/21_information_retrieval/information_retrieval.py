"""
Program 21 : Information Retrieval System with Ranking using TF-IDF
Course Outcome : CO2

Aim
---
Build a small search engine: given a text query, return the most relevant
documents, ranked by TF-IDF cosine similarity.

Pipeline
--------
    documents -> pre-processing -> inverted index -> TF-IDF vectors
    query     -> pre-processing -> TF-IDF vector  -> cosine similarity -> ranked results

TF-IDF formula (same as scikit-learn, so the result can be cross-checked):
    idf(t)     = ln((1 + N) / (1 + df(t))) + 1
    weight     = tf(t, d) * idf(t)         then L2-normalised

What this program does
----------------------
1. Pre-processes the document collection (lower-case, tokenize, stop-words, light stemming).
2. Builds an inverted index  (term -> documents containing it).
3. Ranks documents for several queries and prints a snippet of each result.
4. Cross-checks the ranking with scikit-learn.
5. Evaluates the system with Precision@k and Recall@k (top k results) on a few relevance judgements.
6. Optional interactive search:   python information_retrieval.py --interactive

Run:  python information_retrieval.py
"""

import math
import re
import sys
from collections import Counter, defaultdict

import numpy as np
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer

# ----------------------------------------------------------------------------
# Document collection
# ----------------------------------------------------------------------------
documents = {
    "D01": "Python is a popular programming language used for web development, automation and data science.",
    "D02": "Machine learning algorithms learn patterns from data to make predictions.",
    "D03": "Deep learning uses neural networks with many layers to recognise images and speech.",
    "D04": "Natural language processing allows computers to understand and generate human language.",
    "D05": "The cricket world cup final was won by the home team in the last over.",
    "D06": "The football league season ended with a thrilling final match and a record crowd.",
    "D07": "Regular exercise and a balanced diet help prevent heart disease and diabetes.",
    "D08": "Doctors recommend vaccination to protect people against serious infections.",
    "D09": "The stock market rallied as investors reacted to lower inflation and interest rates.",
    "D10": "Central banks raise interest rates to control inflation and stabilise the economy.",
    "D11": "Search engines use indexing and ranking algorithms to retrieve relevant web pages.",
    "D12": "Data scientists use Python libraries such as pandas and scikit-learn to train machine learning models.",
    "D13": "The new smartphone offers a faster processor, a better camera and longer battery life.",
    "D14": "Astronomers used a powerful telescope to discover a new planet orbiting a distant star.",
    "D15": "Information retrieval systems rank documents by their relevance to a user query.",
}

# Relevance judgements for evaluation (query -> set of truly relevant documents)
relevance = {
    "machine learning with python":   {"D02", "D12", "D01"},
    "interest rates and inflation":   {"D09", "D10"},
    "winning the final match":        {"D05", "D06"},
    "how do search engines rank pages": {"D11", "D15"},
}


def section(title):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


# ----------------------------------------------------------------------------
# Pre-processing
# ----------------------------------------------------------------------------
STOP_WORDS = set(ENGLISH_STOP_WORDS)


def stem(word):
    """Very light suffix stripper (so 'rates' ~ 'rate', 'ranking' ~ 'rank')."""
    for suffix in ("ing", "ed", "es", "s"):
        if word.endswith(suffix) and len(word) - len(suffix) >= 3:
            return word[: -len(suffix)]
    return word


def preprocess(text):
    tokens = re.findall(r"[a-z]+", text.lower())
    return [stem(t) for t in tokens if t not in STOP_WORDS and len(t) > 1]


# ----------------------------------------------------------------------------
# Search engine
# ----------------------------------------------------------------------------
class SearchEngine:
    def __init__(self, docs):
        self.doc_ids = list(docs.keys())
        self.docs = docs
        self.tokens = {d: preprocess(t) for d, t in docs.items()}
        self.N = len(docs)

        # Inverted index : term -> {doc_id: term frequency}
        self.index = defaultdict(dict)
        for d, toks in self.tokens.items():
            for term, tf in Counter(toks).items():
                self.index[term][d] = tf

        # IDF values
        self.idf = {t: math.log((1 + self.N) / (1 + len(postings))) + 1
                    for t, postings in self.index.items()}

        # TF-IDF vector (dict) and norm of every document
        self.doc_vectors = {}
        for d, toks in self.tokens.items():
            vec = {t: tf * self.idf[t] for t, tf in Counter(toks).items()}
            norm = math.sqrt(sum(w * w for w in vec.values())) or 1.0
            self.doc_vectors[d] = {t: w / norm for t, w in vec.items()}

    def search(self, query, top_k=5):
        """Return [(doc_id, score)] ranked by cosine similarity with the query."""
        q_tokens = [t for t in preprocess(query) if t in self.idf]
        if not q_tokens:
            return []
        q_vec = {t: tf * self.idf[t] for t, tf in Counter(q_tokens).items()}
        q_norm = math.sqrt(sum(w * w for w in q_vec.values())) or 1.0
        q_vec = {t: w / q_norm for t, w in q_vec.items()}

        # Only documents that contain at least one query term are scored
        candidates = {d for t in q_vec for d in self.index[t]}
        scores = {d: sum(w * self.doc_vectors[d].get(t, 0.0) for t, w in q_vec.items())
                  for d in candidates}
        ranked = sorted(scores.items(), key=lambda p: (-p[1], p[0]))
        return ranked[:top_k]


engine = SearchEngine(documents)

# ----------------------------------------------------------------------------
# Part A : Index statistics
# ----------------------------------------------------------------------------
section("PART A : Collection and inverted index")
print(f"Documents          : {engine.N}")
print(f"Unique index terms : {len(engine.index)}")
print("\nExample of the inverted index (term -> documents):")
for term in ["learn", "rate", "python", "final", "data"]:
    if term in engine.index:
        print(f"   {term:<8} -> {sorted(engine.index[term])}   idf = {engine.idf[term]:.3f}")
print("\nPre-processing example:")
print(f"   '{documents['D12']}'")
print(f"   -> {engine.tokens['D12']}")

# ----------------------------------------------------------------------------
# Part B : Ranked retrieval
# ----------------------------------------------------------------------------
section("PART B : Ranked retrieval")
queries = ["machine learning with python",
           "interest rates and inflation",
           "winning the final match",
           "how do search engines rank pages",
           "healthy food for the heart"]
for q in queries:
    print(f"\nQuery: '{q}'")
    results = engine.search(q, top_k=3)
    if not results:
        print("   No matching documents.")
    for rank, (d, score) in enumerate(results, start=1):
        print(f"   {rank}. [{d}] score = {score:.4f}   {documents[d][:75]}")

# ----------------------------------------------------------------------------
# Part C : Cross-check with scikit-learn
# ----------------------------------------------------------------------------
section("PART C : Cross-check against scikit-learn TfidfVectorizer")
vectorizer = TfidfVectorizer(tokenizer=preprocess, lowercase=False, token_pattern=None)
D = vectorizer.fit_transform(documents.values())
ids = list(documents.keys())
all_match = True
for q in queries[:4]:
    sk_scores = (vectorizer.transform([q]) @ D.T).toarray()[0]
    order = np.argsort(-sk_scores, kind="stable")
    sk_top = [ids[i] for i in order if sk_scores[i] > 0][:3]    # ignore zero-score documents
    my_top = [d for d, _ in engine.search(q, top_k=3)]
    same = sk_top == my_top
    all_match &= same
    print(f"   '{q}'\n      scratch: {my_top}   scikit-learn: {sk_top}   same ranking: {same}")
print(f"\nRankings identical for all checked queries : {all_match}")

# ----------------------------------------------------------------------------
# Part D : Evaluation
# ----------------------------------------------------------------------------
section("PART D : Evaluation (Precision@k and Recall@k)")
K = 3
print(f"k = {K}\n{'Query':<36}{'Precision@k':>12}{'Recall@k':>10}")
precisions, recalls = [], []
for q, relevant in relevance.items():
    retrieved = [d for d, _ in engine.search(q, top_k=K)]
    hits = len(set(retrieved) & relevant)
    p = hits / len(retrieved) if retrieved else 0.0    # precision of what was returned
    r = hits / len(relevant)                           # share of relevant docs found
    precisions.append(p)
    recalls.append(r)
    print(f"{q:<36}{p:12.2f}{r:10.2f}")
print(f"{'Average':<36}{np.mean(precisions):12.2f}{np.mean(recalls):10.2f}")

# ----------------------------------------------------------------------------
# Part E : Interactive search (optional)
# ----------------------------------------------------------------------------
if "--interactive" in sys.argv:
    section("PART E : Interactive search (press Enter on an empty line to quit)")
    while True:
        q = input("\nSearch > ").strip()
        if not q:
            break
        results = engine.search(q, top_k=5)
        if not results:
            print("   No matching documents.")
        for rank, (d, score) in enumerate(results, start=1):
            print(f"   {rank}. [{d}] {score:.4f}  {documents[d]}")
else:
    print("\nTip: run  python information_retrieval.py --interactive  to type your own queries.")
