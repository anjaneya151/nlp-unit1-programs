"""
Program 18 : Topic Modeling using Latent Semantic Analysis (LSA)
Course Outcome : CO2

Aim
---
Find latent topics with LSA (also called LSI): a TF-IDF matrix is decomposed
with Truncated Singular Value Decomposition (SVD).

    X (docs x terms)  ~  U_k  *  S_k  *  V_k^T

    U_k * S_k  -> documents in a k-dimensional "topic" space
    V_k^T      -> how strongly each term belongs to each topic (component)

Unlike LDA (Program 17), LSA is a linear-algebra method: the values can be
negative and are not probabilities.

What this program does
----------------------
1. Builds a TF-IDF matrix.
2. Applies TruncatedSVD with 3 components.
3. Prints the explained variance and the top terms of each component.
4. Projects documents into topic space, finds the dominant topic of each
   document and runs a semantic search (query vs documents).

Run:  python lsa_topic_modeling.py
"""

import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import Normalizer

documents = [
    # --- Space
    "NASA launched a rocket carrying a satellite into orbit around the earth",
    "The astronauts on the space station studied the planet mars and distant stars",
    "Scientists use a powerful telescope to observe galaxies and stars in space",
    "The mission to the moon will send astronauts in a rocket to explore the lunar surface",
    "A new satellite will study the orbit of planets and the solar system",
    # --- Cooking / Food
    "Add chopped onions and garlic to the pan and cook the sauce slowly",
    "Bake the cake in a hot oven and add sugar butter and flour to the mixture",
    "The chef cooked fresh pasta with tomato sauce garlic and basil",
    "Fry the vegetables in oil and season the dish with salt and pepper",
    "Mix flour sugar and eggs to make the dough then bake the bread in the oven",
    # --- Cricket / Sports
    "The batsman scored a century and the team won the cricket match",
    "The bowler took five wickets and the captain praised the team",
    "Fans cheered as the team won the tournament final at the stadium",
    "The coach trained the players for the match and the season",
    "The team scored in the last over and won the league match",
]

N_TOPICS = 3
TOP_TERMS = 8


def section(title):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


# ----------------------------------------------------------------------------
# Part A : TF-IDF
# ----------------------------------------------------------------------------
section("PART A : TF-IDF matrix")
tfidf = TfidfVectorizer(stop_words="english", token_pattern=r"(?u)\b[a-zA-Z]{3,}\b")
X = tfidf.fit_transform(documents)
terms = tfidf.get_feature_names_out()
print(f"Documents x terms : {X.shape}")

# ----------------------------------------------------------------------------
# Part B : Truncated SVD
# ----------------------------------------------------------------------------
section(f"PART B : Truncated SVD with {N_TOPICS} components")
svd = TruncatedSVD(n_components=N_TOPICS, random_state=42)
doc_topic = svd.fit_transform(X)

print("Explained variance ratio per component :")
for i, ratio in enumerate(svd.explained_variance_ratio_):
    print(f"   Component {i}: {100 * ratio:5.1f}%")
print(f"   Total          : {100 * svd.explained_variance_ratio_.sum():5.1f}%")
print("Singular values :", np.round(svd.singular_values_, 3))

# ----------------------------------------------------------------------------
# Part C : Top terms per topic
# ----------------------------------------------------------------------------
section("PART C : Top terms of each LSA component")
for t, comp in enumerate(svd.components_):
    top = comp.argsort()[::-1][:TOP_TERMS]
    print(f"   Topic {t}: " + ", ".join(f"{terms[i]} ({comp[i]:.2f})" for i in top))
print("\nThe sign and the order of components are arbitrary labels; read the top terms")
print("to name each topic (here: cricket, cooking, space).")

# ----------------------------------------------------------------------------
# Part D : Documents in topic space
# ----------------------------------------------------------------------------
section("PART D : Documents in the latent topic space")
table = pd.DataFrame(doc_topic, columns=[f"Topic {t}" for t in range(N_TOPICS)],
                     index=[f"Doc{i + 1}" for i in range(len(documents))])
table["Text"] = [d[:50] + "..." for d in documents]
pd.set_option("display.width", 250)
pd.set_option("display.max_colwidth", 60)
print(table.round(3).to_string())

# ----------------------------------------------------------------------------
# Part E : Semantic search in LSA space
# ----------------------------------------------------------------------------
section("PART E : Semantic search - TF-IDF space vs LSA space")
lsa_norm = Normalizer(copy=False)
doc_lsa = lsa_norm.fit_transform(doc_topic)

query = "astronauts travelling to a planet"
q_tfidf = tfidf.transform([query])
q_lsa = Normalizer(copy=False).fit_transform(svd.transform(q_tfidf))

tfidf_scores = cosine_similarity(q_tfidf, X)[0]
lsa_scores = cosine_similarity(q_lsa, doc_lsa)[0]

print(f"Query: '{query}'\n")
print(f"{'Rank':<5}{'TF-IDF cosine':<34}{'LSA cosine'}")
order_t = np.argsort(-tfidf_scores)[:5]
order_l = np.argsort(-lsa_scores)[:5]
for r in range(5):
    a, b = order_t[r], order_l[r]
    print(f"{r + 1:<5}Doc{a + 1:<3} score {tfidf_scores[a]:.3f}{'':<17}Doc{b + 1:<3} score {lsa_scores[b]:.3f}")
print("\nTF-IDF only scores documents that contain a query word (Doc2, Doc4). LSA also")
print("finds Doc1 and Doc5, which share NO word with the query but belong to the same")
print("space topic. With only 3 components every space document collapses onto the same")
print("topic direction, which is why they all score 1.000 - more components give finer scores.")

# ----------------------------------------------------------------------------
# Part F : Term-term similarity
# ----------------------------------------------------------------------------
section("PART F : Related terms in LSA space")
term_vecs = Normalizer().fit_transform(svd.components_.T)       # terms x topics
for word in ["rocket", "bake", "team"]:
    if word in terms:
        i = list(terms).index(word)
        sims = cosine_similarity(term_vecs[i:i + 1], term_vecs)[0]
        nearest = [terms[j] for j in np.argsort(-sims) if j != i][:5]
        print(f"   {word:<7} -> {', '.join(nearest)}")
