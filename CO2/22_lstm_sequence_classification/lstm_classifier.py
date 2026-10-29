"""
Program 22: Sequence Classification using RNN/LSTM (Keras / TensorFlow)
Task: binary sentiment classification of short movie-review sentences.
"""
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models

tf.random.set_seed(42)
np.random.seed(42)

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

test_texts = [
    "What a great and wonderful movie",
    "This was a boring and terrible film",
    "I really loved the story and the music",
    "Awful script, I hated it",
]
test_labels = np.array([1, 0, 1, 0], dtype="float32")

# ---- Text -> integer sequences ----
vectorizer = layers.TextVectorization(max_tokens=500, output_sequence_length=12)
vectorizer.adapt(texts)
X_train = vectorizer(tf.constant(texts))
X_test = vectorizer(tf.constant(test_texts))

# ---- LSTM model ----
model = models.Sequential([
    layers.Embedding(input_dim=500, output_dim=32, mask_zero=True),
    layers.LSTM(32),
    layers.Dense(16, activation="relu"),
    layers.Dense(1, activation="sigmoid"),
])
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
model.fit(X_train, labels, epochs=40, batch_size=4, verbose=0)
train_loss, train_acc = model.evaluate(X_train, labels, verbose=0)
test_loss, test_acc = model.evaluate(X_test, test_labels, verbose=0)
print(f"Train accuracy: {train_acc:.2f}")
print(f"Test accuracy : {test_acc:.2f}")

probs = model.predict(X_test, verbose=0).ravel()
for t, p in zip(test_texts, probs):
    print(f"{p:.3f} -> {'POSITIVE' if p >= 0.5 else 'NEGATIVE'} | {t}")
