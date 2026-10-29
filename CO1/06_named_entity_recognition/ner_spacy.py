"""
Program 6 : Named Entity Recognition (NER) using spaCy
Course Outcome : CO1

Description
-----------
NER finds and classifies real-world entities in text: persons, organisations,
locations, dates, money values, etc. This program:
    * extracts all entities with their labels and character positions
    * prints the meaning of every label using spacy.explain()
    * groups entities by their type
    * saves a highlighted HTML visualisation (ner_output.html)

How to run
----------
    pip install spacy
    python -m spacy download en_core_web_sm
    python ner_spacy.py
"""

import os
from collections import defaultdict

import spacy
from spacy import displacy


TEXT = (
    "Sundar Pichai, the CEO of Google, announced on 15 September 2025 that the "
    "company will invest $10 billion in India to open a new AI research centre "
    "in Bengaluru. Microsoft and Apple made similar announcements last year, "
    "while Elon Musk said Tesla would follow in Europe."
)


def extract_entities(text: str) -> None:
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(text)

    print("----- Named Entities -----")
    print(f"{'ENTITY':<28}{'LABEL':<12}{'START':<7}{'END':<7}DESCRIPTION")
    print("-" * 90)
    for ent in doc.ents:
        print(
            f"{ent.text:<28}{ent.label_:<12}{ent.start_char:<7}"
            f"{ent.end_char:<7}{spacy.explain(ent.label_)}"
        )

    grouped = defaultdict(list)
    for ent in doc.ents:
        grouped[ent.label_].append(ent.text)

    print("\n----- Entities grouped by type -----")
    for label, values in grouped.items():
        print(f"{label:<12}: {sorted(set(values))}")

    # Save a colour-highlighted HTML file of the entities.
    html = displacy.render(doc, style="ent", page=True)
    output_path = os.path.join(os.path.dirname(__file__), "ner_output.html")
    with open(output_path, "w", encoding="utf-8") as file_handle:
        file_handle.write(html)
    print(f"\nVisualisation saved to: {output_path}")


def main() -> None:
    print("Input text:\n" + TEXT + "\n")
    extract_entities(TEXT)


if __name__ == "__main__":
    main()
