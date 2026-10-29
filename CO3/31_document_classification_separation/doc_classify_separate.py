"""
Program 31: Automatic Document Classification and Separation using ML
1. Train a TF-IDF + Logistic Regression page classifier (invoice / resume / news / legal).
2. Take a mixed stream of scanned-page texts, classify each page, and separate the stream
   into documents: a new document starts when the predicted class changes OR the TF-IDF
   cosine similarity with the previous page drops below a threshold.
All data is generated synthetically so the program runs offline.
"""
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.metrics.pairwise import cosine_similarity

rng = np.random.default_rng(42)

CLASS_WORDS = {
    "invoice": "invoice amount total tax payment due customer quantity price bill order paid balance".split(),
    "resume":  "experience education skills project university degree internship objective python team".split(),
    "news":    "government minister reported yesterday police city market election said official today".split(),
    "legal":   "agreement party hereby clause liability contract court terms jurisdiction witness section".split(),
}
COMMON = "the of and to in a is for on with by as at".split()
VOCAB_POOL = [f"w{i}" for i in range(300)]   # doc-specific words (names, topics...)


def make_document(cls, n_pages):
    """Pages of one document share a small set of document-specific words."""
    doc_words = list(rng.choice(VOCAB_POOL, 8, replace=False))
    pages = []
    for _ in range(n_pages):
        words = (list(rng.choice(CLASS_WORDS[cls], 10)) + list(rng.choice(doc_words, 6))
                 + list(rng.choice(COMMON, 6)))
        rng.shuffle(words)
        pages.append(" ".join(words))
    return pages


# ---------- 1. train page classifier ----------
classes = list(CLASS_WORDS)
train_X, train_y = [], []
for _ in range(60):
    c = classes[rng.integers(len(classes))]
    for p in make_document(c, int(rng.integers(1, 5))):
        train_X.append(p)
        train_y.append(c)

tfidf = TfidfVectorizer()
Xtr = tfidf.fit_transform(train_X)
clf = LogisticRegression(max_iter=1000).fit(Xtr, train_y)

# hold-out page classification report
test_X, test_y = [], []
for _ in range(30):
    c = classes[rng.integers(len(classes))]
    for p in make_document(c, int(rng.integers(1, 5))):
        test_X.append(p)
        test_y.append(c)
print("=== Page classification (hold-out) ===")
print(classification_report(test_y, clf.predict(tfidf.transform(test_X)), digits=3))

# ---------- 2. separate a mixed stream ----------
stream, truth_cls, truth_boundary = [], [], []
for c in ["invoice", "invoice", "resume", "news", "legal", "legal", "invoice", "news"]:
    pages = make_document(c, int(rng.integers(2, 5)))
    for i, p in enumerate(pages):
        stream.append(p)
        truth_cls.append(c)
        truth_boundary.append(1 if i == 0 else 0)

Xs = tfidf.transform(stream)
pred_cls = clf.predict(Xs)
SIM_THRESHOLD = 0.15
pred_boundary = [1]
for i in range(1, len(stream)):
    sim = cosine_similarity(Xs[i], Xs[i - 1])[0, 0]
    pred_boundary.append(int(pred_cls[i] != pred_cls[i - 1] or sim < SIM_THRESHOLD))

tp = sum(1 for t, p in zip(truth_boundary, pred_boundary) if t and p)
fp = sum(1 for t, p in zip(truth_boundary, pred_boundary) if not t and p)
fn = sum(1 for t, p in zip(truth_boundary, pred_boundary) if t and not p)
prec, rec = tp / max(tp + fp, 1), tp / max(tp + fn, 1)
f1 = 2 * prec * rec / max(prec + rec, 1e-9)

print("=== Document separation ===")
print(f"Pages in stream: {len(stream)} | true documents: {sum(truth_boundary)}")
print(f"Boundary precision={prec:.2f} recall={rec:.2f} F1={f1:.2f}\n")

doc_no = 0
for i, (c, b) in enumerate(zip(pred_cls, pred_boundary)):
    if b:
        doc_no += 1
        print(f"\n--- Document {doc_no} (predicted class: {c}) ---")
    print(f"  page {i + 1:02d}  pred={c:<8s} true={truth_cls[i]}")
