# NLP Lab Programs – CO1 & CO2

Natural Language Processing (NLP) laboratory programs implemented in Python using **NLTK**, **spaCy**, **scikit-learn**, **Gensim**, **TextBlob** and **VADER**.

**Student:** Anjaneya Mallick
**GitHub:** [@anjaneya151](https://github.com/anjaneya151)
**Course Outcomes covered:** CO1 (Programs 1–6) and CO2 (Programs 8–21, updated as the course progresses)

---

## Programs

### CO1 – Text Pre-processing and Linguistic Analysis

| S. No. | Program | Folder | File | CO |
|:------:|---------|--------|------|:--:|
| 1 | Tokenization of Sentences and Words using NLTK and spaCy | `01_tokenization` | `tokenization_nltk_spacy.py` | CO1 |
| 2 | Stemming and Lemmatization on Sample Text | `02_stemming_lemmatization` | `stemming_lemmatization.py` | CO1 |
| 3 | Stop-word Removal from a Document | `03_stopword_removal` | `stopword_removal.py` | CO1 |
| 4 | Part-of-Speech (POS) Tagging using HMM and NLTK | `04_pos_tagging` | `pos_tagging_hmm_nltk.py` | CO1 |
| 5 | Parsing and Chunking using RegEx and spaCy | `05_parsing_chunking` | `parsing_chunking.py` | CO1 |
| 6 | Named Entity Recognition (NER) using spaCy | `06_named_entity_recognition` | `ner_spacy.py` | CO1 |

### CO2 – Text Representation, Embeddings and Applications

All CO2 programs are inside the [`CO2/`](CO2) folder.

| S. No. | Program | Folder (inside `CO2/`) | File | CO |
|:------:|---------|------------------------|------|:--:|
| 8 | Bag-of-Words (BoW) Vectorization and Representation | `08_bag_of_words` | `bag_of_words.py` | CO2 |
| 9 | TF-IDF Implementation and Comparison with BoW | `09_tfidf` | `tfidf.py` | CO2 |
| 10 | N-Gram Model (Uni-, Bi-, Tri-gram) Generation from Corpus | `10_ngram_model` | `ngram_model.py` | CO2 |
| 11 | Cosine Similarity Computation between Text Documents | `11_cosine_similarity` | `cosine_similarity.py` | CO2 |
| 12 | Word2Vec Word Embeddings using Gensim on a Custom Corpus | `12_word2vec_gensim` | `word2vec_gensim.py` | CO2 |
| 13 | GloVe Embeddings Loading and Vector Representation | `13_glove_embeddings` | `glove_embeddings.py` | CO2 |
| 14 | Text Similarity using Word Mover's Distance (WMD) | `14_word_movers_distance` | `word_movers_distance.py` | CO2 |
| 15 | Text Classification using Naïve Bayes / SVM with TF-IDF | `15_text_classification` | `text_classification.py` | CO2 |
| 16 | Sentiment Analysis using TextBlob and VADER | `16_sentiment_analysis` | `sentiment_textblob_vader.py` | CO2 |
| 17 | Topic Modeling using Latent Dirichlet Allocation (LDA) | `17_lda_topic_modeling` | `lda_topic_modeling.py` | CO2 |
| 18 | Topic Modeling using Latent Semantic Analysis (LSA) | `18_lsa_topic_modeling` | `lsa_topic_modeling.py` | CO2 |
| 19 | Opinion Mining on Product / Service Reviews Dataset | `19_opinion_mining` | `opinion_mining.py` | CO2 |
| 20 | Information Extraction (IE) from Structured / Unstructured Documents | `20_information_extraction` | `information_extraction.py` | CO2 |
| 21 | Information Retrieval System with Ranking using TF-IDF | `21_information_retrieval` | `information_retrieval.py` | CO2 |

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
├── CO2/
│   ├── 08_bag_of_words/
│   │   └── bag_of_words.py
│   ├── 09_tfidf/
│   │   └── tfidf.py
│   ├── 10_ngram_model/
│   │   └── ngram_model.py
│   ├── 11_cosine_similarity/
│   │   └── cosine_similarity.py
│   ├── 12_word2vec_gensim/
│   │   └── word2vec_gensim.py
│   ├── 13_glove_embeddings/
│   │   └── glove_embeddings.py
│   ├── 14_word_movers_distance/
│   │   └── word_movers_distance.py
│   ├── 15_text_classification/
│   │   └── text_classification.py
│   ├── 16_sentiment_analysis/
│   │   └── sentiment_textblob_vader.py
│   ├── 17_lda_topic_modeling/
│   │   └── lda_topic_modeling.py
│   ├── 18_lsa_topic_modeling/
│   │   └── lsa_topic_modeling.py
│   ├── 19_opinion_mining/
│   │   ├── opinion_mining.py
│   │   └── sample_reviews.csv
│   ├── 20_information_extraction/
│   │   └── information_extraction.py
│   └── 21_information_retrieval/
│       └── information_retrieval.py
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

* NLTK corpora used by the CO1 programs are downloaded automatically on first run.
* **Programs 13 and 14** need GloVe vectors. They are downloaded automatically through Gensim (`glove-wiki-gigaword-50`, ≈ 65 MB, internet needed once). For offline use, place `glove.6B.50d.txt` (from [Stanford GloVe](https://nlp.stanford.edu/projects/glove/)) in the program's folder.

## Running a program

```bash
python 01_tokenization/tokenization_nltk_spacy.py
python CO2/09_tfidf/tfidf.py
python CO2/21_information_retrieval/information_retrieval.py --interactive
```

Every program is self-contained and prints clearly labelled output (PART A, PART B, …).

---

## CO2 program details

### 8. Bag-of-Words
Builds a BoW matrix from scratch and with `CountVectorizer`, verifies that both are identical, and shows binary BoW, stop-word removal, sparsity and how to vectorize an unseen sentence.

### 9. TF-IDF and comparison with BoW
Computes TF, IDF and TF-IDF step by step, verifies the result against `TfidfVectorizer`, and compares BoW weights with TF-IDF weights to show how common words are down-weighted.

### 10. N-Gram model
Generates unigrams, bigrams and trigrams, ranks them by frequency, verifies counts with `CountVectorizer(ngram_range)`, then builds a **bigram language model** (conditional probabilities, add-one smoothing, perplexity, next-word prediction and text generation).

### 11. Cosine similarity
Implements cosine similarity with NumPy, builds document–document similarity matrices for BoW and TF-IDF, finds the most/least similar pairs, ranks documents for a query, and shows why cosine similarity ignores document length.

### 12. Word2Vec with Gensim
Trains **CBOW** and **Skip-gram** models on a custom corpus, shows word vectors, most-similar words, similarity scores and odd-one-out, saves/reloads the model and plots the embeddings in 2-D with PCA (`word2vec_plot.png`).

### 13. GloVe embeddings
Loads pre-trained GloVe vectors, inspects word vectors, finds similar words, solves analogies (king − man + woman), builds sentence vectors by averaging, and handles out-of-vocabulary words.

### 14. Word Mover's Distance
Computes WMD between sentences using GloVe embeddings, compares it with cosine similarity of averaged vectors, builds a pairwise distance matrix and retrieves the closest document for a query.

### 15. Text classification
Classifies short texts into Sports / Technology / Health / Politics with **TF-IDF + Multinomial Naïve Bayes** and **TF-IDF + Linear SVM** using scikit-learn pipelines. Reports accuracy, precision/recall/F1, confusion matrix, cross-validation and top words per class. *(The embedded dataset is small, so scores are for demonstration.)*

### 16. Sentiment analysis
Scores sentences with **TextBlob** (polarity, subjectivity) and **VADER** (compound score), shows how VADER handles negation, intensifiers, capitalisation, "!" and emojis, and measures agreement between the two tools.

### 17. LDA topic modeling
Fits **Latent Dirichlet Allocation** on a Bag-of-Words matrix, prints the top words of each topic, the topic mixture of each document, infers topics for unseen documents and compares perplexity for different numbers of topics.

### 18. LSA topic modeling
Applies **Truncated SVD** on a TF-IDF matrix, reports explained variance and top terms per component, projects documents into topic space, and shows semantic search and related terms in LSA space.

### 19. Opinion mining on reviews
Analyses 30 smartphone reviews (`sample_reviews.csv`): review-level sentiment, comparison with star ratings, **aspect-based opinion mining** (battery, camera, display, price, support, …), words associated with positive/negative reviews, and saves `opinion_mining_results.csv`.

### 20. Information extraction
Extracts information from **structured** data (CSV, JSON) and **unstructured** text: regular expressions for emails, phone numbers, dates, URLs and amounts; spaCy **NER**; subject–verb–object **relation extraction**; and saves everything as `extracted_information.json`.

### 21. Information retrieval system
Builds a search engine with pre-processing, an **inverted index** and **TF-IDF cosine ranking**, cross-checks the ranking with scikit-learn, evaluates it with Precision@k / Recall@k, and offers an interactive search mode.

---

## Requirements

- Python 3.9+
- numpy, pandas, scikit-learn
- nltk >= 3.8, spacy >= 3.7 with `en_core_web_sm`
- gensim >= 4.3, POT (for Word Mover's Distance)
- textblob, vaderSentiment
- matplotlib (optional – only for the Word2Vec plot)

> This repository is updated as new CO2 programs are completed in the course.
