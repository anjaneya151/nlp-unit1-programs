"""
Program 32: Hybrid Probabilistic + FSM-Based Sequence Modeling for document boundary detection

Each page in a scanned stream is in one of two hidden states:
    NEW  = first page of a document        CONT = continuation page
Probabilistic part:
    * Naive Bayes over TF-IDF gives P(state | page text)  -> emission scores
    * Transition probabilities P(state_t | state_{t-1}) are learned from labelled streams (HMM)
Finite-state-machine part (hard constraints on legal sequences):
    * the stream must start in NEW
    * NEW -> NEW is forbidden (every document has at least 2 pages)
    * legal transitions: NEW->CONT, CONT->CONT, CONT->NEW
Viterbi decoding finds the most probable state sequence that the FSM allows.
Compared against a baseline that thresholds the Naive Bayes output alone.
"""
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

rng = np.random.default_rng(1)
CONT, NEW = 0, 1

CLASS_WORDS = {
    "invoice": "amount total tax payment due customer quantity price bill order paid".split(),
    "resume":  "experience education skills project university degree internship team python".split(),
    "news":    "government minister reported yesterday police city market election said official".split(),
    "legal":   "party hereby clause liability contract court terms jurisdiction witness section".split(),
}
HEADERS = {
    "invoice": ["invoice", "billto"], "resume": ["curriculum", "vitae"],
    "news": ["headline", "dateline"], "legal": ["agreement", "whereas"],
}
COMMON = "the of and to in a is for on with by as at".split()


def make_stream(n_docs):
    pages, labels = [], []
    for _ in range(n_docs):
        c = list(CLASS_WORDS)[rng.integers(4)]
        n_pages = int(rng.integers(2, 6))
        for i in range(n_pages):
            w = list(rng.choice(CLASS_WORDS[c], 10)) + list(rng.choice(COMMON, 8))
            if i == 0 and rng.random() < 0.8:        # header cue (noisy: 80% of first pages)
                w += HEADERS[c]
            elif i > 0 and rng.random() < 0.06:      # occasional header-like noise
                w += HEADERS[c][:1]
            rng.shuffle(w)
            pages.append(" ".join(w))
            labels.append(NEW if i == 0 else CONT)
    return pages, np.array(labels)


# ---------- train ----------
train_pages, train_labels = make_stream(80)
tfidf = TfidfVectorizer()
nb = MultinomialNB().fit(tfidf.fit_transform(train_pages), train_labels)

prior = np.array([(train_labels == CONT).mean(), (train_labels == NEW).mean()])
counts = np.ones((2, 2))                              # Laplace smoothing
for a, b in zip(train_labels[:-1], train_labels[1:]):
    counts[a, b] += 1
trans = counts / counts.sum(1, keepdims=True)
trans[NEW, NEW] = 0.0                                 # FSM constraint: NEW -> NEW not allowed
trans = trans / trans.sum(1, keepdims=True)
print("Learned transition matrix (rows: from CONT/NEW, cols: to CONT/NEW):\n", np.round(trans, 3))


def viterbi(page_texts):
    post = nb.predict_proba(tfidf.transform(page_texts))          # P(state | text)
    emit = np.log(post + 1e-12) - np.log(prior)                   # scaled likelihood
    T = len(page_texts)
    logt = np.log(trans + 1e-12)
    dp = np.full((T, 2), -np.inf)
    bp = np.zeros((T, 2), dtype=int)
    dp[0, NEW] = emit[0, NEW]                                     # FSM: must start in NEW
    for t in range(1, T):
        for s in (CONT, NEW):
            cand = dp[t - 1] + logt[:, s]
            bp[t, s] = int(np.argmax(cand))
            dp[t, s] = cand[bp[t, s]] + emit[t, s]
    path = [int(np.argmax(dp[-1]))]
    for t in range(T - 1, 0, -1):
        path.append(bp[t, path[-1]])
    return np.array(path[::-1]), post


def prf(true, pred):
    tp = int(((true == NEW) & (pred == NEW)).sum())
    fp = int(((true == CONT) & (pred == NEW)).sum())
    fn = int(((true == NEW) & (pred == CONT)).sum())
    p, r = tp / max(tp + fp, 1), tp / max(tp + fn, 1)
    return p, r, 2 * p * r / max(p + r, 1e-9)


# ---------- evaluate on an unseen stream ----------
test_pages, test_labels = make_stream(25)
hybrid_pred, post = viterbi(test_pages)
baseline_pred = (post[:, NEW] > 0.5).astype(int)
baseline_pred[0] = NEW

print(f"\nTest stream: {len(test_pages)} pages, {int((test_labels == NEW).sum())} documents")
for name, pred in [("Baseline (Naive Bayes only)", baseline_pred), ("Hybrid (NB + HMM + FSM)", hybrid_pred)]:
    p, r, f = prf(test_labels, pred)
    print(f"{name:<30s} precision={p:.3f} recall={r:.3f} F1={f:.3f}")

print("\nFirst 15 pages  (true | hybrid):")
names = {CONT: "CONT", NEW: "NEW "}
for i in range(15):
    print(f"  page {i + 1:02d}: {names[test_labels[i]]} | {names[hybrid_pred[i]]}")
