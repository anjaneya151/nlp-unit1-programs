"""
Program 20 : Information Extraction (IE) from Structured / Unstructured Documents
Course Outcome : CO2

Aim
---
Turn documents into structured facts.

    Structured data   -> already in rows / columns / key-value form (CSV, JSON)
    Unstructured text -> free text (emails, news, invoices)

What this program does
----------------------
PART A : Structured data    - read a CSV table and a JSON record, query and extract fields.
PART B : Unstructured text  - rule-based extraction with regular expressions
                              (emails, phone numbers, dates, URLs, money amounts).
PART C : Named Entity Recognition with spaCy (PERSON, ORG, GPE, DATE, MONEY).
PART D : Relation extraction (Subject - Verb - Object) with spaCy dependency parse.
PART E : Putting everything together into a structured record and saving it as JSON.

Setup (once):
    pip install spacy pandas
    python -m spacy download en_core_web_sm

Run:  python information_extraction.py
"""

import io
import json
import os
import re

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
OUTPUT_FILE = os.path.join(HERE, "extracted_information.json")


def section(title):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


pd.set_option("display.width", 250)
pd.set_option("display.max_columns", None)

# ============================================================================
# PART A : STRUCTURED DOCUMENTS
# ============================================================================
section("PART A : Information extraction from STRUCTURED data (CSV + JSON)")

csv_text = """employee_id,name,department,city,salary,joining_date
101,Aarav Sharma,Engineering,Bengaluru,85000,2021-06-15
102,Priya Nair,Marketing,Mumbai,62000,2020-02-01
103,Rohan Gupta,Engineering,Delhi,91000,2019-11-20
104,Sneha Iyer,Finance,Chennai,70000,2022-01-10
105,Vikram Singh,Marketing,Delhi,58000,2023-03-05
"""
table = pd.read_csv(io.StringIO(csv_text), parse_dates=["joining_date"])
print("Table read from CSV:")
print(table.to_string(index=False))

print("\nQuery 1 - Employees of the Engineering department:")
print(table[table["department"] == "Engineering"][["name", "city"]].to_string(index=False))

print("\nQuery 2 - Average salary per department:")
print(table.groupby("department")["salary"].mean().round(0).to_string())

print("\nQuery 3 - Employees who joined after 1 Jan 2021:")
recent = table[table["joining_date"] > "2021-01-01"]
print(", ".join(recent["name"]))

json_text = """
{
  "invoice_no": "INV-2024-0457",
  "customer": {"name": "Meera Kapoor", "email": "meera.kapoor@example.com"},
  "items": [{"product": "Laptop", "qty": 1, "price": 55000},
            {"product": "Mouse",  "qty": 2, "price": 700}],
  "paid": false
}
"""
record = json.loads(json_text)
total = sum(i["qty"] * i["price"] for i in record["items"])
print("\nFields extracted from the JSON invoice:")
print(f"   Invoice no : {record['invoice_no']}")
print(f"   Customer   : {record['customer']['name']} <{record['customer']['email']}>")
print(f"   Items      : {[i['product'] for i in record['items']]}")
print(f"   Total      : Rs. {total:,}   |   Paid: {record['paid']}")

# ============================================================================
# PART B : UNSTRUCTURED TEXT - REGULAR EXPRESSIONS
# ============================================================================
section("PART B : Rule-based extraction from UNSTRUCTURED text (regex)")

email_text = """
Subject: Meeting about the NLP project

Hello team,

Dr. Ananya Rao from Infosys will visit our Pune office on 15 March 2025 to discuss the
NLP project. The project budget is Rs. 2,50,000 and the first review is on 02/04/2025.
Please send your updates to ananya.rao@infosys.com or contact her on +91 98765 43210.
More details: https://www.example.com/nlp-project

Regards,
Rahul Verma
"""

PATTERNS = {
    "emails": r"[\w.+-]+@[\w-]+\.[\w.-]+\w",
    "phones": r"(?:\+91[\s-]?)?[6-9]\d{4}[\s-]?\d{5}",
    "urls":   r"https?://[^\s]+",
    "dates":  (r"\b\d{1,2}\s(?:January|February|March|April|May|June|July|August|"
               r"September|October|November|December)\s\d{4}\b|\b\d{1,2}/\d{1,2}/\d{4}\b"),
    "money":  r"(?:Rs\.?|INR|\$|₹)\s?\d[\d,]*(?:\.\d+)?",
}

regex_results = {name: re.findall(pattern, email_text) for name, pattern in PATTERNS.items()}

print("Input document:")
print(email_text)
print("Extracted with regular expressions:")
for name, found in regex_results.items():
    print(f"   {name:<7}: {found}")

# ============================================================================
# PART C / D : SPACY - NER AND RELATIONS
# ============================================================================
try:
    import spacy
    nlp = spacy.load("en_core_web_sm")
except (ImportError, OSError):
    raise SystemExit("\nspaCy or its English model is missing. Install them with:\n"
                     "    pip install spacy\n"
                     "    python -m spacy download en_core_web_sm\n"
                     "(Parts A and B above ran without spaCy.)")

section("PART C : Named Entity Recognition with spaCy")
news = ("Tata Consultancy Services announced on Monday that it will invest $200 million in a new "
        "research centre in Hyderabad. Chief executive K. Krithivasan said the centre will hire "
        "5,000 engineers by December 2026. The announcement was welcomed by Telangana minister "
        "Sridhar Babu in Mumbai.")
doc = nlp(news)
print(f"Text:\n   {news}\n")
print(f"{'Entity':<28}{'Label':<10}Meaning")
for ent in doc.ents:
    print(f"{ent.text:<28}{ent.label_:<10}{spacy.explain(ent.label_)}")

grouped = {}
for ent in doc.ents:
    grouped.setdefault(ent.label_, []).append(ent.text)
print("\nGrouped by entity type:")
for label, items in grouped.items():
    print(f"   {label:<8}: {items}")

section("PART D : Relation extraction (Subject - Verb - Object)")


def extract_svo(sentence_doc):
    """Very simple SVO triples from the dependency parse of a spaCy Doc."""
    triples = []
    for token in sentence_doc:
        if token.pos_ == "VERB":
            subjects = [w for w in token.lefts if w.dep_ in ("nsubj", "nsubjpass")]
            objects = [w for w in token.rights if w.dep_ in ("dobj", "obj", "attr")]
            for s in subjects:
                for o in objects:
                    triples.append((" ".join(t.text for t in s.subtree),
                                    token.lemma_,
                                    " ".join(t.text for t in o.subtree)))
    return triples


triples = []
for sent in doc.sents:
    triples.extend(extract_svo(sent))
if triples:
    for s, v, o in triples:
        print(f"   ({s})  --[{v}]-->  ({o})")
else:
    print("   No SVO triples found.")

# ============================================================================
# PART E : COMBINE INTO ONE STRUCTURED RECORD
# ============================================================================
section("PART E : Structured record built from the unstructured text")
email_doc = nlp(email_text)
record = {
    "document_type": "email",
    "people": sorted({e.text for e in email_doc.ents if e.label_ == "PERSON"}),
    "organisations": sorted({e.text for e in email_doc.ents if e.label_ == "ORG"}),
    "locations": sorted({e.text for e in email_doc.ents if e.label_ in ("GPE", "LOC")}),
    "dates": regex_results["dates"],
    "emails": regex_results["emails"],
    "phones": regex_results["phones"],
    "urls": regex_results["urls"],
    "amounts": regex_results["money"],
}
print(json.dumps(record, indent=2, ensure_ascii=False))

with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(record, f, indent=2, ensure_ascii=False)
print(f"\nSaved to : {OUTPUT_FILE}")
