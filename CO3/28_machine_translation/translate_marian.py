"""
Program 28: Machine Translation using MarianMT (Hugging Face)
English -> Hindi and English -> French with Helsinki-NLP opus-mt models.
"""
from transformers import MarianMTModel, MarianTokenizer

sentences = [
    "Natural language processing is fun.",
    "Where is the nearest railway station?",
    "I am learning machine translation.",
]


def translate(model_name, texts):
    tok = MarianTokenizer.from_pretrained(model_name)
    model = MarianMTModel.from_pretrained(model_name)
    batch = tok(texts, return_tensors="pt", padding=True)
    out = model.generate(**batch, max_new_tokens=60, num_beams=4)
    return tok.batch_decode(out, skip_special_tokens=True)


for name, lang in [("Helsinki-NLP/opus-mt-en-hi", "Hindi"), ("Helsinki-NLP/opus-mt-en-fr", "French")]:
    print(f"\n=== English -> {lang} ({name}) ===")
    for src, tgt in zip(sentences, translate(name, sentences)):
        print(f"{src}\n  -> {tgt}")
