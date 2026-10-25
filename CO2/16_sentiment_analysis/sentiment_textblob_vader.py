"""
Program 16 : Sentiment Analysis using TextBlob and VADER
Course Outcome : CO2

Aim
---
Find the sentiment (positive / negative / neutral) of sentences with two
lexicon-based tools and compare them.

TextBlob : returns  polarity     in [-1, +1]   (negative -> positive)
                    subjectivity in [ 0,  1]   (objective -> subjective)
VADER    : returns  neg / neu / pos proportions and a normalised "compound"
                    score in [-1, +1]. Made for social-media text: understands
                    emojis, CAPS, punctuation (!!!), intensifiers and negation.

    VADER label rule (standard)   : compound >=  0.05 -> positive
                                    compound <= -0.05 -> negative, else neutral
    TextBlob label rule (common)  : polarity  >   0.05 -> positive
                                    polarity  <  -0.05 -> negative, else neutral

Install : pip install textblob vaderSentiment

What this program does
----------------------
1. Scores a set of sentences with both tools and prints a comparison table.
2. Demonstrates how VADER handles negation, intensifiers, CAPS, "!" and emojis.
3. Reports how often the two tools agree and the overall sentiment of the set.

Run:  python sentiment_textblob_vader.py
"""

import re

import pandas as pd
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

vader = SentimentIntensityAnalyzer()


def section(title):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def label_from_score(score, threshold=0.05):
    if score >= threshold:
        return "Positive"
    if score <= -threshold:
        return "Negative"
    return "Neutral"


def analyse(text):
    """Return TextBlob and VADER results for one piece of text."""
    blob = TextBlob(text).sentiment
    v = vader.polarity_scores(text)
    return {
        "tb_polarity": blob.polarity,
        "tb_subjectivity": blob.subjectivity,
        "tb_label": label_from_score(blob.polarity),
        "vader_neg": v["neg"], "vader_neu": v["neu"], "vader_pos": v["pos"],
        "vader_compound": v["compound"],
        "vader_label": label_from_score(v["compound"]),
    }


pd.set_option("display.width", 250)
pd.set_option("display.max_columns", None)
pd.set_option("display.max_colwidth", 55)

sentences = [
    "I love this phone, the camera is amazing!",
    "The delivery was late and the packaging was damaged.",
    "The movie was okay, nothing special.",
    "This is the worst service I have ever experienced.",
    "The package arrived on Tuesday.",
    "The teacher explained the topic very clearly and I enjoyed the class.",
    "I am not happy with the quality of this product.",
    "Absolutely fantastic experience, highly recommended!!!",
    "The food was not bad at all.",
    "Battery life is terrible but the display is gorgeous.",
]

# ----------------------------------------------------------------------------
# Part A : Comparison table
# ----------------------------------------------------------------------------
section("PART A : TextBlob vs VADER on sample sentences")
rows = []
for s in sentences:
    r = analyse(s)
    r["text"] = s
    rows.append(r)
table = pd.DataFrame(rows)

print(table[["text", "tb_polarity", "tb_label", "vader_compound", "vader_label"]]
      .round(3).to_string(index=False))

# ----------------------------------------------------------------------------
# Part B : TextBlob subjectivity and VADER score breakdown
# ----------------------------------------------------------------------------
section("PART B : TextBlob subjectivity and VADER neg / neu / pos split")
print(table[["text", "tb_subjectivity", "vader_neg", "vader_neu", "vader_pos"]]
      .round(3).to_string(index=False))
print("\nSubjectivity close to 0 = factual statement ('The package arrived on Tuesday'),")
print("close to 1 = personal opinion.")

# ----------------------------------------------------------------------------
# Part C : How VADER treats language features
# ----------------------------------------------------------------------------
section("PART C : VADER compound score and language features")
tests = [
    ("Baseline",           "The food is good."),
    ("Intensifier",        "The food is very good."),
    ("Capitalisation",     "The food is GOOD."),
    ("Exclamation marks",  "The food is good!!!"),
    ("Negation",           "The food is not good."),
    ("Emoji",              "The food is good :)"),
    ("Contrast with 'but'", "The food is good but the service is bad."),
]
print(f"{'Feature':<22} {'Sentence':<42} {'VADER':>7} {'TextBlob':>9}")
for feature, text in tests:
    v = vader.polarity_scores(text)["compound"]
    t = TextBlob(text).sentiment.polarity
    print(f"{feature:<22} {text:<42} {v:7.3f} {t:9.3f}")

# ----------------------------------------------------------------------------
# Part D : Agreement and overall sentiment
# ----------------------------------------------------------------------------
section("PART D : Agreement between the two tools")
agree = (table["tb_label"] == table["vader_label"])
print(f"Both tools give the same label for {agree.sum()} of {len(table)} sentences "
      f"({100 * agree.mean():.0f}%).")

disagree = table[~agree]
if not disagree.empty:
    print("\nSentences where they disagree:")
    for _, row in disagree.iterrows():
        print(f"   '{row['text']}'")
        print(f"      TextBlob: {row['tb_label']} ({row['tb_polarity']:.2f})   "
              f"VADER: {row['vader_label']} ({row['vader_compound']:.2f})")

section("PART E : Overall sentiment of the whole set")
print("VADER label distribution:")
print(table["vader_label"].value_counts().to_string())
print(f"\nMean VADER compound score   : {table['vader_compound'].mean():.3f}")
print(f"Mean TextBlob polarity      : {table['tb_polarity'].mean():.3f}")

# ----------------------------------------------------------------------------
# Part F : Sentence-by-sentence analysis of a paragraph
# ----------------------------------------------------------------------------
section("PART F : Sentence-level sentiment of a paragraph")
paragraph = ("I bought this laptop last month. The screen is bright and the keyboard feels great. "
             "However, the fan is noisy and the battery drains quickly. "
             "Overall, I am satisfied but expected more for the price.")
for text in re.split(r"(?<=[.!?])\s+", paragraph.strip()):
    v = vader.polarity_scores(text)["compound"]
    print(f"   VADER {v:+.3f} {label_from_score(v):<9} | {text}")
print(f"\nWhole paragraph (VADER compound) : {vader.polarity_scores(paragraph)['compound']:+.3f}")
