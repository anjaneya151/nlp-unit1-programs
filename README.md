# NLP Lab – Unit 1 Programs

Natural Language Processing (NLP) laboratory programs for **Unit 1**, implemented in Python using **NLTK** and **spaCy**.

**Student:** Anjaneya Mallick
**GitHub:** [@anjaneya151](https://github.com/anjaneya151)
**Course Outcome covered:** CO1

---

## Programs

| S. No. | Program | Folder | File | CO |
|:--:|---|---|---|:--:|
| 1 | Tokenization of Sentences and Words using NLTK and spaCy | `01_tokenization` | `tokenization_nltk_spacy.py` | CO1 |
| 2 | Stemming and Lemmatization on Sample Text | `02_stemming_lemmatization` | `stemming_lemmatization.py` | CO1 |
| 3 | Stop-word Removal from a Document | `03_stopword_removal` | `stopword_removal.py` | CO1 |
| 4 | Part-of-Speech (POS) Tagging using HMM and NLTK | `04_pos_tagging` | `pos_tagging_hmm_nltk.py` | CO1 |
| 5 | Parsing and Chunking using RegEx and spaCy | `05_parsing_chunking` | `parsing_chunking.py` | CO1 |
| 6 | Named Entity Recognition (NER) using spaCy | `06_named_entity_recognition` | `ner_spacy.py` | CO1 |

---

## Repository structure

```text
nlp-unit1-programs/
├── 01_tokenization/
│   └── tokenization_nltk_spacy.py
├── 02_stemming_lemmatization/
│   └── stemming_lemmatization.py
├── 03_stopword_removal/
│   ├── sample_document.txt
│   └── stopword_removal.py
├── 04_pos_tagging/
│   └── pos_tagging_hmm_nltk.py
├── 05_parsing_chunking/
│   └── parsing_chunking.py
├── 06_named_entity_recognition/
│   └── ner_spacy.py
├── requirements.txt
└── README.md
```

---

## Setup

```bash
# 1. (optional) create a virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 2. install libraries
pip install -r requirements.txt

# 3. download the spaCy English model
python -m spacy download en_core_web_sm
```

NLTK corpora (`punkt`, `wordnet`, `stopwords`, `treebank`, taggers) are downloaded automatically by the programs on first run.

## Running a program

```bash
python 01_tokenization/tokenization_nltk_spacy.py
```

Every program is self-contained and prints clearly labelled output.

---

## Program details

### 1. Tokenization
Splits a paragraph into sentences and words with **NLTK** (`sent_tokenize`, `word_tokenize`) and with **spaCy** (`doc.sents`, token iteration), then compares the token counts of both libraries.

### 2. Stemming and Lemmatization
Applies the **Porter** and **Snowball** stemmers, the **WordNet lemmatizer** (with POS mapping for better accuracy) and the **spaCy** lemmatizer to the same text, printed side by side so the difference between stem and lemma is clear.

### 3. Stop-word Removal
Reads `sample_document.txt` and removes stop words three ways: a hand-written custom list, the **NLTK** stop-word corpus and **spaCy**'s built-in stop-word list. Also reports how many tokens were removed and the most frequent remaining words.

### 4. POS Tagging (HMM + NLTK)
Tags a sentence with NLTK's in-built averaged-perceptron tagger, then trains a **Hidden Markov Model** tagger (`HiddenMarkovModelTrainer.train_supervised`) on 90% of the NLTK `treebank` corpus, evaluates its accuracy on the remaining 10% and tags the same sentence with it for comparison.

### 5. Parsing and Chunking
Defines a **regular-expression chunk grammar** (NP, VP, PP) and parses POS-tagged tokens with `nltk.RegexpParser`, printing the parse tree and the extracted phrases. Then performs **dependency parsing** with spaCy and lists its noun chunks.

### 6. Named Entity Recognition
Extracts named entities with spaCy, prints each entity with its label, character offsets and an explanation, groups entities by type and saves a colour-highlighted `ner_output.html` using `displacy`.

---

## Requirements

- Python 3.8+
- nltk >= 3.8
- spacy >= 3.7 with `en_core_web_sm`
