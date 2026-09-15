"""
Program 4 : Part-of-Speech (POS) Tagging of a Given Sentence
Course Outcome : CO1

Description
-----------
Two tagging techniques are demonstrated:

    1. NLTK's in-built averaged perceptron tagger  (nltk.pos_tag)
    2. A Hidden Markov Model (HMM) tagger trained by us on the NLTK
       `treebank` corpus using nltk.tag.hmm.HiddenMarkovModelTrainer

The HMM tagger learns:
    * transition probabilities  P(tag_i | tag_i-1)
    * emission probabilities    P(word | tag)
and then uses the Viterbi algorithm to find the most probable tag sequence.

How to run
----------
    pip install nltk
    python pos_tagging_hmm_nltk.py
"""

import nltk
from nltk.corpus import treebank
from nltk.probability import LidstoneProbDist
from nltk.tag import hmm
from nltk.tokenize import word_tokenize


SENTENCE = "The quick brown fox jumps over the lazy dog near the old bank."


def download_resources() -> None:
    for resource in (
        "punkt",
        "punkt_tab",
        "treebank",
        "averaged_perceptron_tagger",
        "averaged_perceptron_tagger_eng",
        "tagsets",
        "tagsets_json",
    ):
        nltk.download(resource, quiet=True)


def pos_tagging_with_nltk(sentence: str) -> None:
    """Built-in NLTK tagger (Penn Treebank tag set)."""
    tokens = word_tokenize(sentence)
    tagged = nltk.pos_tag(tokens)

    print("----- NLTK in-built POS tagger -----")
    print(f"{'WORD':<12}{'TAG':<8}")
    print("-" * 20)
    for word, tag in tagged:
        print(f"{word:<12}{tag:<8}")
    print()


def train_hmm_tagger():
    """Train an HMM POS tagger on the treebank corpus and report accuracy."""
    tagged_sentences = treebank.tagged_sents()
    split = int(0.9 * len(tagged_sentences))
    train_data = tagged_sentences[:split]
    test_data = tagged_sentences[split:]

    # Lidstone smoothing gives unseen words a small non-zero probability,
    # otherwise the tagger labels every unknown word with the same tag.
    def smoothed_estimator(freqdist, bins):
        return LidstoneProbDist(freqdist, 0.1, bins)

    trainer = hmm.HiddenMarkovModelTrainer()
    hmm_tagger = trainer.train_supervised(train_data, estimator=smoothed_estimator)

    accuracy = hmm_tagger.accuracy(test_data)
    print("----- HMM POS tagger (trained by us) -----")
    print(f"Training sentences : {len(train_data)}")
    print(f"Testing sentences  : {len(test_data)}")
    print(f"Test accuracy      : {accuracy:.4f}\n")
    return hmm_tagger


def pos_tagging_with_hmm(hmm_tagger, sentence: str) -> None:
    tokens = word_tokenize(sentence)
    tagged = hmm_tagger.tag(tokens)

    print(f"{'WORD':<12}{'TAG':<8}")
    print("-" * 20)
    for word, tag in tagged:
        print(f"{word:<12}{tag:<8}")
    print()


def main() -> None:
    download_resources()
    print("Input sentence:\n" + SENTENCE + "\n")

    pos_tagging_with_nltk(SENTENCE)

    hmm_tagger = train_hmm_tagger()
    pos_tagging_with_hmm(hmm_tagger, SENTENCE)

    # Meaning of a few common Penn Treebank tags.
    print("Tag meanings: DT=determiner, JJ=adjective, NN=noun, "
          "VBZ=verb(3rd person singular), IN=preposition, RB=adverb.")


if __name__ == "__main__":
    main()
