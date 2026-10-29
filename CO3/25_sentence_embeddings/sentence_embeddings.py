"""
Program 25: Sentence Embeddings using BERT and GPT-2
Mean-pooled hidden states from both models, compared with cosine similarity.
"""
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel

sentences = [
    "The cat sits on the mat.",
    "A kitten is resting on the rug.",
    "The stock market fell sharply today.",
    "Shares dropped heavily in trading.",
]


def embed(model_name, sents):
    tok = AutoTokenizer.from_pretrained(model_name)
    if tok.pad_token is None:          # GPT-2 has no pad token
        tok.pad_token = tok.eos_token
    model = AutoModel.from_pretrained(model_name).eval()
    enc = tok(sents, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        hidden = model(**enc).last_hidden_state            # (B, T, H)
    mask = enc["attention_mask"].unsqueeze(-1).float()
    return ((hidden * mask).sum(1) / mask.sum(1)).numpy()   # mean pooling


def cosine_matrix(E):
    E = E / np.linalg.norm(E, axis=1, keepdims=True)
    return E @ E.T


for name in ["bert-base-uncased", "gpt2"]:
    E = embed(name, sentences)
    print(f"\n=== {name}  | embedding shape: {E.shape} ===")
    sim = cosine_matrix(E)
    for i in range(len(sentences)):
        for j in range(i + 1, len(sentences)):
            print(f"sim(S{i+1}, S{j+1}) = {sim[i, j]:.3f}")
