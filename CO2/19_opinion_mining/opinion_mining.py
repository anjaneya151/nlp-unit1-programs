"""
Program 19 : Opinion Mining on Product / Service Reviews Dataset
Course Outcome : CO2

Aim
---
Mine opinions from customer reviews of a smartphone: find the overall sentiment
and the sentiment towards each ASPECT (battery, camera, display, price, ...).

Dataset : sample_reviews.csv  (30 reviews with a 1-5 star rating)
          Replace it with your own file that has the columns  review_id, rating, review.

Install : pip install pandas scikit-learn vaderSentiment textblob

What this program does
----------------------
1. Loads and cleans the reviews.
2. Review-level sentiment with VADER (and TextBlob) -> Positive / Neutral / Negative.
3. Checks the predicted sentiment against the star rating (accuracy).
4. Aspect-based opinion mining: finds the sentences that talk about each aspect
   and scores them -> which features customers like or dislike.
5. Lists the words most associated with positive and negative reviews.
6. Saves the results to opinion_mining_results.csv.

Run:  python opinion_mining.py
"""

import os
import re

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score, classification_report
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_FILE = os.path.join(HERE, "sample_reviews.csv")
OUTPUT_FILE = os.path.join(HERE, "opinion_mining_results.csv")

vader = SentimentIntensityAnalyzer()

# Aspects and the keywords that mention them
ASPECTS = {
    "Battery":     ["battery", "charger", "charges", "charging", "charge"],
    "Camera":      ["camera", "photos", "photo"],
    "Display":     ["display", "screen"],
    "Performance": ["performance", "fast", "slow", "lag", "laggy", "hangs", "smooth", "speed"],
    "Price":       ["price", "priced", "overpriced", "expensive", "value", "money", "worth"],
    "Delivery":    ["delivery", "box", "packaging", "shipping"],
    "Support":     ["support", "service", "refund", "team"],
    "Design":      ["design", "look", "sleek"],
}


def section(title):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def clean(text):
    return re.sub(r"\s+", " ", str(text)).strip()


def label(score, threshold=0.05):
    if score >= threshold:
        return "Positive"
    if score <= -threshold:
        return "Negative"
    return "Neutral"


def label_from_rating(rating):
    """Ground truth derived from stars: 4-5 positive, 3 neutral, 1-2 negative."""
    if rating >= 4:
        return "Positive"
    if rating <= 2:
        return "Negative"
    return "Neutral"


def split_sentences(text):
    """Split a review into sentences and then into clauses at 'but' / 'however'."""
    parts = []
    for sent in re.split(r"(?<=[.!?])\s+", text):
        parts.extend(re.split(r"\b(?:but|however|though|although)\b", sent, flags=re.I))
    return [p.strip(" ,.;") for p in parts if p.strip(" ,.;")]


pd.set_option("display.width", 250)
pd.set_option("display.max_colwidth", 70)

# ----------------------------------------------------------------------------
# Part A : Load and clean
# ----------------------------------------------------------------------------
section("PART A : Loading the review dataset")
df = pd.read_csv(DATA_FILE)
df["review"] = df["review"].map(clean)
df = df.dropna(subset=["review"]).drop_duplicates(subset="review").reset_index(drop=True)
print(f"Reviews loaded : {len(df)}")
print(f"Average rating : {df['rating'].mean():.2f} / 5")
print("Rating distribution:")
print(df["rating"].value_counts().sort_index().to_string())

# ----------------------------------------------------------------------------
# Part B : Review-level sentiment
# ----------------------------------------------------------------------------
section("PART B : Review-level sentiment (VADER and TextBlob)")
df["vader_score"] = df["review"].map(lambda t: vader.polarity_scores(t)["compound"])
df["vader_label"] = df["vader_score"].map(label)
df["textblob_score"] = df["review"].map(lambda t: TextBlob(t).sentiment.polarity)
df["textblob_label"] = df["textblob_score"].map(label)
df["true_label"] = df["rating"].map(label_from_rating)

print("VADER sentiment distribution:")
print(df["vader_label"].value_counts().to_string())
print("\nMost positive reviews (VADER):")
for _, r in df.nlargest(3, "vader_score").iterrows():
    print(f"   {r['vader_score']:+.3f}  {r['review']}")
print("\nMost negative reviews (VADER):")
for _, r in df.nsmallest(3, "vader_score").iterrows():
    print(f"   {r['vader_score']:+.3f}  {r['review']}")

# ----------------------------------------------------------------------------
# Part C : Accuracy against star ratings
# ----------------------------------------------------------------------------
section("PART C : Do the predicted sentiments match the star ratings?")
for name, col in [("VADER", "vader_label"), ("TextBlob", "textblob_label")]:
    print(f"{name:<9} accuracy vs. rating-based label : {accuracy_score(df['true_label'], df[col]):.3f}")
print("\nVADER detailed report (neutral = 3-star reviews):")
print(classification_report(df["true_label"], df["vader_label"], zero_division=0))

# ----------------------------------------------------------------------------
# Part D : Aspect-based opinion mining
# ----------------------------------------------------------------------------
section("PART D : Aspect-based opinion mining")
mentions = []
for _, row in df.iterrows():
    for clause in split_sentences(row["review"]):
        words = set(re.findall(r"[a-z]+", clause.lower()))
        for aspect, keywords in ASPECTS.items():
            if words & set(keywords):
                mentions.append({"review_id": row["review_id"], "aspect": aspect,
                                 "text": clause,
                                 "score": vader.polarity_scores(clause)["compound"]})

mentions_df = pd.DataFrame(mentions)
mentions_df["label"] = mentions_df["score"].map(label)

summary = (mentions_df.groupby("aspect")
           .agg(mentions=("score", "size"),
                avg_sentiment=("score", "mean"),
                positive=("label", lambda s: (s == "Positive").sum()),
                negative=("label", lambda s: (s == "Negative").sum()))
           .sort_values("avg_sentiment", ascending=False))
summary["positive_%"] = (100 * summary["positive"] / summary["mentions"]).round(0)
print(summary.round(3).to_string())

best = summary.index[0]
worst = summary.index[-1]
print(f"\nCustomers are happiest with : {best} (avg sentiment {summary.loc[best, 'avg_sentiment']:+.2f})")
print(f"Customers are least happy with : {worst} (avg sentiment {summary.loc[worst, 'avg_sentiment']:+.2f})")

print("\nExample opinions for the weakest aspect:")
for _, r in mentions_df[mentions_df["aspect"] == worst].nsmallest(3, "score").iterrows():
    print(f"   {r['score']:+.3f}  '{r['text']}'")

# ----------------------------------------------------------------------------
# Part E : Words associated with each sentiment
# ----------------------------------------------------------------------------
section("PART E : Words most associated with positive and negative reviews")
polar = df[df["vader_label"].isin(["Positive", "Negative"])]
cv = CountVectorizer(stop_words="english", binary=True, token_pattern=r"(?u)\b[a-zA-Z]{3,}\b")
X = cv.fit_transform(polar["review"])
words = cv.get_feature_names_out()
is_pos = (polar["vader_label"] == "Positive").values
pos_counts = X[is_pos].sum(axis=0).A1
neg_counts = X[~is_pos].sum(axis=0).A1
score = (pos_counts + 1) / (is_pos.sum() + 2) / ((neg_counts + 1) / ((~is_pos).sum() + 2))
ranked = pd.DataFrame({"word": words, "pos": pos_counts, "neg": neg_counts, "ratio": score})
print("Most positive words (appear mainly in positive reviews):")
print(ranked[ranked["pos"] >= 2].nlargest(6, "ratio")[["word", "pos", "neg"]].to_string(index=False))
print("\nMost negative words (appear mainly in negative reviews):")
print(ranked[ranked["neg"] >= 2].nsmallest(6, "ratio")[["word", "pos", "neg"]].to_string(index=False))

# ----------------------------------------------------------------------------
# Part F : Overall conclusion and saving results
# ----------------------------------------------------------------------------
section("PART F : Overall opinion summary")
counts = df["vader_label"].value_counts(normalize=True) * 100
print(f"Positive : {counts.get('Positive', 0):.0f}%   Neutral : {counts.get('Neutral', 0):.0f}%   "
      f"Negative : {counts.get('Negative', 0):.0f}%")
print(f"Net sentiment score (mean VADER compound) : {df['vader_score'].mean():+.3f}")

df.to_csv(OUTPUT_FILE, index=False)
print(f"\nPer-review results saved to : {OUTPUT_FILE}")
