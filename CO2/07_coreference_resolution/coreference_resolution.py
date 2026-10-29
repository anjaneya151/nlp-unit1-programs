"""
Program 7: Coreference Resolution (neuralcoref or similar)

neuralcoref only works with spaCy 2.x / Python <= 3.7, so this program:
  PART A - tries a neural resolver (fastcoref, the modern neuralcoref replacement)
  PART B - always runs a rule-based resolver (Hobbs-style heuristic:
           pronoun -> nearest preceding mention that agrees in gender & number)
  PART C - prints the text with pronouns replaced by their antecedents

Run:  python coreference_resolution.py
Optional neural model:  pip install fastcoref
"""
import re

TEXT = (
    "Ruchika is a professor at NIET. She teaches Natural Language Processing. "
    "Rahul and Priya are her students. They submitted their programs on time. "
    "Rahul pushed his code to GitHub because he wanted it to be reviewed. "
    "Priya said the repository was clear and it was easy to follow."
)

# --- small gazetteer (gender / number of known names) ---
MALE = {"rahul", "amit", "arjun", "anjaneya", "john"}
FEMALE = {"ruchika", "priya", "sita", "mary"}

PRONOUNS = {
    "he": ("m", "sg"), "him": ("m", "sg"), "his": ("m", "sg"), "himself": ("m", "sg"),
    "she": ("f", "sg"), "her": ("f", "sg"), "hers": ("f", "sg"), "herself": ("f", "sg"),
    "it": ("n", "sg"), "its": ("n", "sg"), "itself": ("n", "sg"),
    "they": ("any", "pl"), "them": ("any", "pl"), "their": ("any", "pl"), "themselves": ("any", "pl"),
}
DETERMINERS = {"the", "a", "an"}
STOP_NOUN = {"is", "are", "was", "and", "on", "to", "at", "because", "said", "be"}


def tokenize(text):
    return re.findall(r"[A-Za-z]+|[.,!?]", text)


def find_mentions(tokens):
    """Return candidate mentions: (index, text, gender, number)."""
    mentions = []
    i = 0
    while i < len(tokens):
        t, low = tokens[i], tokens[i].lower()
        # coordinated proper names: "Rahul and Priya" -> plural mention
        if (t[0].isupper() and low not in PRONOUNS and i + 2 < len(tokens)
                and tokens[i + 1] == "and" and tokens[i + 2][0].isupper()):
            mentions.append((i, f"{t} and {tokens[i+2]}", "any", "pl"))
            mentions.append((i, t, gender_of(low), "sg"))
            mentions.append((i + 2, tokens[i + 2], gender_of(tokens[i + 2].lower()), "sg"))
            i += 3
            continue
        if t[0].isupper() and low not in PRONOUNS and low not in DETERMINERS and i > 0 or (
                i == 0 and low in MALE | FEMALE):
            if low in MALE | FEMALE:
                mentions.append((i, t, gender_of(low), "sg"))
            elif t.isalpha() and low not in PRONOUNS:
                mentions.append((i, t, "n", "sg"))  # organisation / place
        # "the repository", "a professor" -> neuter noun phrase
        if low in DETERMINERS and i + 1 < len(tokens) and tokens[i + 1].isalpha():
            noun = tokens[i + 1]
            if noun.lower() not in STOP_NOUN:
                mentions.append((i + 1, f"{t} {noun}", "n", "sg"))
        # bare plural/neuter nouns that are objects: "code", "programs"
        if low in {"code", "programs", "repository"} and not any(m[0] == i for m in mentions):
            mentions.append((i, t, "n", "pl" if low.endswith("s") else "sg"))
        i += 1
    return mentions


def gender_of(name):
    return "m" if name in MALE else "f" if name in FEMALE else "n"


def agrees(p_gender, p_number, m_gender, m_number):
    if p_number != m_number:
        return False
    return p_gender == "any" or p_gender == m_gender


def resolve(tokens):
    mentions = find_mentions(tokens)
    links = []
    for i, tok in enumerate(tokens):
        low = tok.lower()
        if low not in PRONOUNS:
            continue
        g, n = PRONOUNS[low]
        # prefer persons for he/she, nearest preceding agreeing mention
        candidates = [m for m in mentions if m[0] < i and agrees(g, n, m[2], m[3])]
        # Binding rule: a singular pronoun cannot refer to a member of a plural
        # subject in the same clause ("Rahul and Priya are her students").
        sent_start = max([k for k in range(i) if tokens[k] in ".!?"] + [-1])
        plural_members = set()
        for m in mentions:
            if m[3] == "pl" and " and " in m[1] and m[0] > sent_start:
                plural_members.update(m[1].split(" and "))
        if n == "sg":
            candidates = [m for m in candidates
                          if not (m[0] > sent_start and m[1] in plural_members)]
        if candidates:
            # salience: common noun phrases beat proper names for "it",
            # then the nearest preceding mention wins
            def score(m):
                common = g == "n" and not m[1][0].isupper()
                return (common, m[0])
            ante = max(candidates, key=score)
            links.append((i, tok, ante[1]))
    return mentions, links


def build_clusters(links):
    clusters = {}
    for _, pron, ante in links:
        clusters.setdefault(ante, []).append(pron)
    return clusters


def substitute(tokens, links):
    out = list(tokens)
    for i, pron, ante in links:
        possessive = pron.lower() in {"his", "her", "their", "its"}
        out[i] = f"{ante}'s" if possessive else ante
    text = " ".join(out)
    return re.sub(r" ([.,!?])", r"\1", text)


def neural_part(text):
    print("=" * 70)
    print("PART A: Neural coreference (fastcoref, successor of neuralcoref)")
    print("=" * 70)
    try:
        from fastcoref import FCoref
        model = FCoref(device="cpu")
        pred = model.predict(texts=[text])[0]
        for k, cluster in enumerate(pred.get_clusters(), 1):
            print(f"Cluster {k}: {cluster}")
    except Exception as e:  # library not installed / no internet
        print(f"fastcoref not available ({type(e).__name__}). "
              "Install with `pip install fastcoref` to run the neural model.")
        print("Continuing with the rule-based resolver below.\n")


def main():
    print("Input text:\n", TEXT, "\n")
    neural_part(TEXT)

    tokens = tokenize(TEXT)
    mentions, links = resolve(tokens)

    print("=" * 70)
    print("PART B: Rule-based coreference resolution")
    print("=" * 70)
    print("Candidate mentions:")
    for idx, txt, g, n in mentions:
        print(f"  [{idx:>2}] {txt:<20} gender={g:<3} number={n}")
    print("\nPronoun -> antecedent links:")
    for i, pron, ante in links:
        print(f"  [{i:>2}] {pron:<8} -> {ante}")
    print("\nCoreference clusters:")
    for ante, prons in build_clusters(links).items():
        print(f"  {ante}: {[ante] + prons}")

    print("\n" + "=" * 70)
    print("PART C: Resolved text (pronouns replaced)")
    print("=" * 70)
    print(substitute(tokens, links))


if __name__ == "__main__":
    main()
