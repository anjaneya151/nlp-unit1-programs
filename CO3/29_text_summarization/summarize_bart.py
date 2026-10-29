"""
Program 29: Text Summarization using BART (Hugging Face)
facebook/bart-large-cnn - abstractive summarization.
"""
from transformers import pipeline

summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

article = """
Natural language processing (NLP) is a subfield of artificial intelligence that focuses on the
interaction between computers and human language. Early NLP systems relied on hand-written rules,
but modern systems use statistical and neural methods that learn from large amounts of text.
The introduction of the transformer architecture in 2017 changed the field, enabling models such as
BERT and GPT that can be pre-trained on huge corpora and then fine-tuned for tasks like translation,
question answering, sentiment analysis and summarization. Today NLP powers search engines, voice
assistants, chatbots and many tools that help people write, translate and understand text.
"""

for max_len, min_len in [(60, 20), (110, 40)]:
    result = summarizer(article.strip(), max_length=max_len, min_length=min_len, do_sample=False)
    print(f"\nSummary (max_length={max_len}):\n{result[0]['summary_text']}")
