"""
Program 17 : Topic Modeling using Latent Dirichlet Allocation (LDA)
Course Outcome : CO2

Aim
---
Discover hidden topics in a collection of documents with LDA.

Idea
----
LDA is a probabilistic model. It assumes
   * every document is a MIXTURE of topics, and
   * every topic is a distribution over words.
Given only word counts (Bag-of-Words), LDA learns both distributions.

What this program does
----------------------
1. Builds a Bag-of-Words matrix (LDA works on counts, not TF-IDF).
2. Fits LDA with 3 topics (scikit-learn).
3. Prints the top words of each topic and the topic mixture of every document.
4. Infers the topics of unseen documents and compares perplexity for different
   numbers of topics.

Run:  python lda_topic_modeling.py
"""

import numpy as np
import pandas as pd
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer

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
TOP_WORDS = 5


def section(title):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


# ----------------------------------------------------------------------------
# Part A : Bag-of-Words
# ----------------------------------------------------------------------------
section("PART A : Bag-of-Words matrix")
vectorizer = CountVectorizer(stop_words="english", min_df=2, token_pattern=r"(?u)\b[a-zA-Z]{3,}\b")
X = vectorizer.fit_transform(documents)
vocab = vectorizer.get_feature_names_out()
print(f"Documents : {X.shape[0]}   Vocabulary (min_df=2) : {X.shape[1]} words")

# ----------------------------------------------------------------------------
# Part B : Fit LDA
# ----------------------------------------------------------------------------
section(f"PART B : Fitting LDA with {N_TOPICS} topics")
lda = LatentDirichletAllocation(n_components=N_TOPICS, learning_method="batch",
                                max_iter=100, random_state=42)
doc_topic = lda.fit_transform(X)          # shape: (documents, topics)
print("LDA model trained.")
print(f"Log-likelihood : {lda.score(X):.2f}")
print(f"Perplexity     : {lda.perplexity(X):.2f}   (lower is better)")

# ----------------------------------------------------------------------------
# Part C : Top words per topic
# ----------------------------------------------------------------------------
section("PART C : Top words of each topic")
for t, weights in enumerate(lda.components_):
    top = weights.argsort()[::-1][:TOP_WORDS]
    print(f"   Topic {t}: " + ", ".join(vocab[i] for i in top))
print("\n(Topic numbers are arbitrary - read the words and give each topic a name,")
print(" e.g. 'space', 'cooking', 'cricket'.)")

# ----------------------------------------------------------------------------
# Part D : Document-topic distribution
# ----------------------------------------------------------------------------
section("PART D : Topic mixture of every document")
table = pd.DataFrame(doc_topic, columns=[f"Topic {t}" for t in range(N_TOPICS)],
                     index=[f"Doc{i + 1}" for i in range(len(documents))])
table["Dominant"] = ["Topic " + str(i) for i in doc_topic.argmax(axis=1)]
table["Text"] = [d[:48] + "..." for d in documents]
pd.set_option("display.width", 250)
pd.set_option("display.max_colwidth", 60)
print(table.round(2).to_string())

# ----------------------------------------------------------------------------
# Part E : New documents
# ----------------------------------------------------------------------------
section("PART E : Inferring topics for unseen documents")
new_docs = [
    "The rocket will carry astronauts to the space station orbit",
    "Cook the pasta in the pan and add garlic and sauce",
    "The team won the match and the captain scored a century",
]
new_dist = lda.transform(vectorizer.transform(new_docs))
for doc, dist in zip(new_docs, new_dist):
    mix = "  ".join(f"T{t}={p:.2f}" for t, p in enumerate(dist))
    print(f"   '{doc}'\n      -> {mix}   (dominant: Topic {dist.argmax()})\n")

# ----------------------------------------------------------------------------
# Part F : Choosing the number of topics
# ----------------------------------------------------------------------------
section("PART F : Perplexity for different numbers of topics")
for k in [2, 3, 4, 5]:
    model = LatentDirichletAllocation(n_components=k, max_iter=100, random_state=42).fit(X)
    print(f"   topics = {k}   perplexity = {model.perplexity(X):8.2f}")
print("\nPerplexity does not fall steadily with k on such a tiny corpus, so it is not a")
print("reliable way to pick the number of topics here. In practice, also check topic")
print("coherence and whether the top words of each topic are easy to name.")
