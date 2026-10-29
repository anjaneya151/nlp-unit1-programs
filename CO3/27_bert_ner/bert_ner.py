"""
Program 27: Named Entity Recognition using Transformers (BERT)
Uses dslim/bert-base-NER (BERT fine-tuned on CoNLL-2003: PER, ORG, LOC, MISC).
"""
from transformers import pipeline

ner = pipeline("ner", model="dslim/bert-base-NER", aggregation_strategy="simple")

texts = [
    "Sundar Pichai is the CEO of Google, which is headquartered in Mountain View, California.",
    "Narendra Modi visited Paris and met Emmanuel Macron at the UNESCO headquarters.",
    "Infosys opened a new office in Noida, Uttar Pradesh in 2024.",
]

for text in texts:
    print("\nText:", text)
    for ent in ner(text):
        print(f"  {ent['word']:<22s} {ent['entity_group']:<5s} score={ent['score']:.3f}")
