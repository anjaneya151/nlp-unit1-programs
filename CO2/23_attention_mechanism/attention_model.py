"""
Program 23: Implementing an Attention Mechanism (basic custom model)
A custom additive (Bahdanau-style) attention layer on top of an LSTM encoder.
The attention weights show which words the model focuses on.
"""
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models

tf.random.set_seed(7)
np.random.seed(7)

pos = [
    "I loved this movie, it was fantastic",
    "Great acting and a wonderful story",
    "Absolutely brilliant, highly recommended",
    "The film was enjoyable and heartwarming",
    "Superb direction and beautiful music",
    "An amazing experience, I would watch it again",
    "Really good plot, loved every minute",
    "Outstanding performance by the cast",
    "What a wonderful and moving film",
    "Fantastic story with great characters",
]
neg = [
    "Terrible movie, a complete waste of time",
    "The plot was boring and the acting was bad",
    "I hated this film, absolutely awful",
    "Dull, slow and very disappointing",
    "Worst movie I have ever seen",
    "Poor script and terrible direction",
    "I regret watching this, it was horrible",
    "Boring story with weak characters",
    "An awful and disappointing film",
    "Bad acting and a terrible plot",
]
texts = pos + neg
labels = np.array([1] * len(pos) + [0] * len(neg), dtype="float32")
SEQ_LEN = 12

vectorizer = layers.TextVectorization(max_tokens=500, output_sequence_length=SEQ_LEN)
vectorizer.adapt(texts)
X = vectorizer(tf.constant(texts))


class AdditiveAttention(layers.Layer):
    """score_t = v^T tanh(W h_t); alpha = softmax(score); context = sum(alpha_t * h_t)"""

    def __init__(self, units=32, **kwargs):
        super().__init__(**kwargs)
        self.supports_masking = True
        self.W = layers.Dense(units, activation="tanh")
        self.v = layers.Dense(1, use_bias=False)

    def call(self, h, mask=None):
        score = self.v(self.W(h))                       # (B, T, 1)
        if mask is not None:
            m = tf.cast(mask[:, :, None], score.dtype)
            score += (1.0 - m) * -1e9                    # ignore padding
        alpha = tf.nn.softmax(score, axis=1)            # (B, T, 1)
        context = tf.reduce_sum(alpha * h, axis=1)      # (B, units)
        return context, tf.squeeze(alpha, -1)

    def compute_mask(self, inputs, mask=None):
        return [None, None]


inp = layers.Input(shape=(SEQ_LEN,), dtype="int64")
x = layers.Embedding(500, 32, mask_zero=True)(inp)
h = layers.LSTM(32, return_sequences=True)(x)
context, alpha = AdditiveAttention(32)(h)
out = layers.Dense(1, activation="sigmoid")(layers.Dense(16, activation="relu")(context))

model = models.Model(inp, out)
attn_model = models.Model(inp, alpha)  # shares layers, exposes attention weights
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
model.fit(X, labels, epochs=60, batch_size=4, verbose=0)
print("Train accuracy: %.2f" % model.evaluate(X, labels, verbose=0)[1])

vocab = vectorizer.get_vocabulary()
tests = ["What a great and wonderful movie", "This was a boring and terrible film"]
Xt = vectorizer(tf.constant(tests))
probs = model.predict(Xt, verbose=0).ravel()
weights = attn_model.predict(Xt, verbose=0)

for sent, p, w, ids in zip(tests, probs, weights, Xt.numpy()):
    print(f"\n'{sent}' -> {'POSITIVE' if p >= 0.5 else 'NEGATIVE'} ({p:.3f})")
    print("Attention weights per word:")
    for i, wt in zip(ids, w):
        if i != 0:
            print(f"  {vocab[i]:<12s} {wt:.3f}")
