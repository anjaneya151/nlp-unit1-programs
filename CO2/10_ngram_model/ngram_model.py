"""
Program 10 : N-Gram Model (Uni-, Bi-, Tri-gram) Generation from Corpus
Course Outcome : CO2

Aim
---
Generate unigrams, bigrams and trigrams from a corpus, count them, and use the
counts to build a simple bigram language model that predicts the next word.

What this program does
----------------------
1. Generates n-grams from scratch for n = 1, 2, 3.
2. Counts and ranks the most frequent n-grams of the corpus.
3. Verifies the result with scikit-learn's CountVectorizer(ngram_range=...).
4. Builds a bigram language model, predicts the next word and scores a sentence.

Run:  python ngram_model.py
"""

import math
import re
from collections import Counter, defaultdict

from sklearn.feature_extraction.text import CountVectorizer

corpus = """
Natural language processing is a field of artificial intelligence.
Natural language processing helps computers understand human language.
Machine learning is a field of artificial intelligence.
Machine learning helps computers learn from data.
Deep learning is a part of machine learning.
Language models predict the next word in a sentence.
Computers understand human language using natural language processing.
"""

TOKEN_PATTERN = r"(?u)\b[a-zA-Z]+\b"


def section(title):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def split_sentences(text):
    return [s.strip() for s in re.split(r"[.!?]\s*", text) if s.strip()]


def tokenize(sentence):
    return re.findall(TOKEN_PATTERN, sentence.lower())


def generate_ngrams(tokens, n):
    """Return the list of n-grams (as tuples) of a token list."""
    return [tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]


sentences = split_sentences(corpus)
tokenized = [tokenize(s) for s in sentences]

# ----------------------------------------------------------------------------
# Part A : N-grams of a single sentence
# ----------------------------------------------------------------------------
section("PART A : Uni-, Bi- and Tri-grams of one sentence")
sample = tokenized[0]
print(f"Sentence : {sentences[0]}")
print(f"Tokens   : {sample}")
for n, name in [(1, "Unigrams"), (2, "Bigrams"), (3, "Trigrams")]:
    grams = generate_ngrams(sample, n)
    print(f"\n{name} (n={n}) -> {len(grams)} items")
    print("   " + ", ".join(" ".join(g) for g in grams))

# ----------------------------------------------------------------------------
# Part B : N-gram frequencies over the whole corpus
# ----------------------------------------------------------------------------
section("PART B : N-gram frequencies over the whole corpus")
counters = {}
for n, name in [(1, "unigram"), (2, "bigram"), (3, "trigram")]:
    counter = Counter()
    for tokens in tokenized:
        counter.update(generate_ngrams(tokens, n))
    counters[n] = counter
    print(f"\nTotal {name}s: {sum(counter.values())}   Unique {name}s: {len(counter)}")
    print(f"Top 5 {name}s:")
    for gram, count in counter.most_common(5):
        print(f"   {' '.join(gram):<35} {count}")

# ----------------------------------------------------------------------------
# Part C : Verification with scikit-learn
# ----------------------------------------------------------------------------
section("PART C : Verification using CountVectorizer(ngram_range)")
for n in (1, 2, 3):
    cv = CountVectorizer(token_pattern=TOKEN_PATTERN, ngram_range=(n, n))
    cv.fit(sentences)
    print(f"n = {n} : scikit-learn found {len(cv.get_feature_names_out()):>3} unique n-grams, "
          f"scratch code found {len(counters[n]):>3}")

# ----------------------------------------------------------------------------
# Part D : Bigram language model
# ----------------------------------------------------------------------------
section("PART D : Bigram language model with <s> and </s> markers")
unigram_lm = Counter()
bigram_lm = Counter()
for tokens in tokenized:
    padded = ["<s>"] + tokens + ["</s>"]
    unigram_lm.update(padded)
    bigram_lm.update(generate_ngrams(padded, 2))

next_words = defaultdict(Counter)
for (w1, w2), c in bigram_lm.items():
    next_words[w1][w2] = c

vocab_size = len(unigram_lm)


def bigram_prob(w1, w2, smoothing=True):
    """P(w2 | w1) = count(w1, w2) / count(w1)   (optionally with add-one smoothing)."""
    if smoothing:
        return (bigram_lm[(w1, w2)] + 1) / (unigram_lm[w1] + vocab_size)
    return bigram_lm[(w1, w2)] / unigram_lm[w1] if unigram_lm[w1] else 0.0


print("P(w2 | w1) without smoothing:")
for w1, w2 in [("natural", "language"), ("language", "processing"),
               ("machine", "learning"), ("learning", "is")]:
    print(f"   P({w2:<10} | {w1:<9}) = {bigram_prob(w1, w2, smoothing=False):.3f}")

print("\nNext-word prediction (most likely continuations):")
for word in ["natural", "machine", "computers", "is"]:
    options = next_words[word].most_common(3)
    total = sum(next_words[word].values())
    shown = ", ".join(f"{w} ({c / total:.2f})" for w, c in options)
    print(f"   after '{word}': {shown}")


def sentence_log_prob(sentence):
    tokens = ["<s>"] + tokenize(sentence) + ["</s>"]
    return sum(math.log(bigram_prob(a, b)) for a, b in generate_ngrams(tokens, 2))


section("PART E : Scoring sentences with the bigram model (add-one smoothing)")
tests = ["Natural language processing helps computers understand human language",
         "Language natural helps processing computers",
         "Machine learning is a field of artificial intelligence"]
for t in tests:
    lp = sentence_log_prob(t)
    n_tokens = len(tokenize(t)) + 1
    print(f"   log P = {lp:8.3f}   perplexity = {math.exp(-lp / n_tokens):8.2f}   | {t}")
print("\nPerplexity is the fair comparison because it is normalised by sentence length:")
print("a well-formed sentence gets a LOWER perplexity than a scrambled sentence.")


# ----------------------------------------------------------------------------
# Part F : Text generation
# ----------------------------------------------------------------------------
section("PART F : Generating text with the bigram model (greedy)")
word, output = "natural", ["natural"]
for _ in range(7):
    if not next_words[word]:
        break
    word = next_words[word].most_common(1)[0][0]
    if word == "</s>":
        break
    output.append(word)
print("Generated:", " ".join(output))
