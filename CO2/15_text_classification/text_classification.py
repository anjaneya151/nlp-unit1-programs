"""
Program 15 : Text Classification using Naive Bayes / SVM with TF-IDF
Course Outcome : CO2

Aim
---
Classify short texts into categories (Sports, Technology, Health, Politics) using
TF-IDF features with a Multinomial Naive Bayes and a linear SVM classifier.

What this program does
----------------------
1. Loads a small labelled dataset (embedded in the script).
2. Splits it into train / test sets (stratified).
3. Builds scikit-learn Pipelines: TF-IDF -> Naive Bayes and TF-IDF -> LinearSVC.
4. Evaluates both: accuracy, precision / recall / F1, confusion matrix, cross-validation.
5. Shows the most informative words per class and predicts new sentences.

Note: the dataset is small, so scores are only for demonstration. For a real
project, replace `data` with a bigger dataset (e.g. 20 Newsgroups).

Run:  python text_classification.py
"""

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

# ----------------------------------------------------------------------------
# Dataset : (text, label)
# ----------------------------------------------------------------------------
data = [
    # ---- Sports
    ("The striker scored twice as the team won the football match", "Sports"),
    ("India won the cricket series after a thrilling final over", "Sports"),
    ("The tennis player reached the final of the grand slam tournament", "Sports"),
    ("The coach praised the players after the league victory", "Sports"),
    ("She broke the national record in the hundred metre sprint", "Sports"),
    ("The goalkeeper saved a penalty in the last minute of the game", "Sports"),
    ("The batsman hit a century and the bowlers took ten wickets", "Sports"),
    ("The basketball team qualified for the championship playoffs", "Sports"),
    ("Fans filled the stadium to watch the Olympic athletes compete", "Sports"),
    ("The captain lifted the trophy after the tournament final", "Sports"),
    ("The boxer defended his title with a knockout in the third round", "Sports"),
    ("The swimmer won gold and set a new world record", "Sports"),
    # ---- Technology
    ("The new smartphone has a faster processor and a better camera", "Technology"),
    ("Engineers released a software update that fixes security bugs", "Technology"),
    ("The startup built an artificial intelligence tool for coding", "Technology"),
    ("Cloud computing lets companies store data on remote servers", "Technology"),
    ("The laptop comes with a high resolution display and long battery life", "Technology"),
    ("Researchers trained a neural network to recognise images", "Technology"),
    ("The company launched a new operating system for tablets", "Technology"),
    ("Developers use Python and open source libraries to build apps", "Technology"),
    ("Cybersecurity experts warned about a new malware attack", "Technology"),
    ("The robot uses sensors and machine learning to navigate", "Technology"),
    ("Fast 5G networks improve internet speed on mobile devices", "Technology"),
    ("The chip maker unveiled a powerful processor for data centres", "Technology"),
    # ---- Health
    ("Doctors recommend regular exercise and a balanced diet", "Health"),
    ("The vaccine reduces the risk of serious infection", "Health"),
    ("Patients with diabetes should monitor their blood sugar", "Health"),
    ("The hospital opened a new clinic for heart disease treatment", "Health"),
    ("Sleeping eight hours helps the body and the brain recover", "Health"),
    ("Nurses and doctors treated hundreds of patients during the outbreak", "Health"),
    ("A healthy diet with fruit and vegetables lowers cholesterol", "Health"),
    ("The medicine helps relieve pain and reduces fever", "Health"),
    ("Yoga and meditation can reduce stress and improve mental health", "Health"),
    ("Researchers found a new therapy for cancer patients", "Health"),
    ("Drinking enough water keeps the body hydrated and healthy", "Health"),
    ("The surgeon performed a successful operation on the patient", "Health"),
    # ---- Politics
    ("The parliament passed the new budget after a long debate", "Politics"),
    ("The prime minister addressed the nation about the election", "Politics"),
    ("Opposition parties criticised the government over the tax bill", "Politics"),
    ("Voters lined up at polling stations on election day", "Politics"),
    ("The senate voted to approve the new foreign policy treaty", "Politics"),
    ("The president signed the law and met the cabinet ministers", "Politics"),
    ("Lawmakers proposed a bill to reform the voting system", "Politics"),
    ("The party announced its candidate for the upcoming election", "Politics"),
    ("Leaders of both countries signed a trade agreement at the summit", "Politics"),
    ("The minister resigned after protests against the government", "Politics"),
    ("The court reviewed the constitution amendment passed by parliament", "Politics"),
    ("Diplomats met to discuss sanctions and international relations", "Politics"),
]

df = pd.DataFrame(data, columns=["text", "label"])


def section(title):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


section("PART A : Dataset")
print(f"Total samples : {len(df)}")
print(df["label"].value_counts().to_string())

# ----------------------------------------------------------------------------
# Part B : Train / test split
# ----------------------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    df["text"], df["label"], test_size=0.25, random_state=42, stratify=df["label"])

section("PART B : Train / test split (75% / 25%, stratified)")
print(f"Training samples : {len(X_train)}")
print(f"Testing samples  : {len(X_test)}")

# ----------------------------------------------------------------------------
# Part C : Pipelines
# ----------------------------------------------------------------------------
tfidf_params = dict(stop_words="english", ngram_range=(1, 2), sublinear_tf=True)

models = {
    "Naive Bayes": Pipeline([("tfidf", TfidfVectorizer(**tfidf_params)),
                             ("clf", MultinomialNB(alpha=0.5))]),
    "Linear SVM":  Pipeline([("tfidf", TfidfVectorizer(**tfidf_params)),
                             ("clf", LinearSVC(C=1.0, random_state=42))]),
}

section("PART C : Training and evaluation")
labels = sorted(df["label"].unique())
results = {}
for name, pipe in models.items():
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    results[name] = acc

    print(f"\n----- {name} -----")
    print(f"Test accuracy : {acc:.3f}\n")
    print(classification_report(y_test, y_pred, labels=labels, zero_division=0))
    cm = pd.DataFrame(confusion_matrix(y_test, y_pred, labels=labels),
                      index=[f"true_{l}" for l in labels],
                      columns=[f"pred_{l}" for l in labels])
    print("Confusion matrix:")
    print(cm.to_string())

# ----------------------------------------------------------------------------
# Part D : Cross-validation
# ----------------------------------------------------------------------------
section("PART D : 4-fold stratified cross-validation (whole dataset)")
cv = StratifiedKFold(n_splits=4, shuffle=True, random_state=42)
for name, pipe in models.items():
    scores = cross_val_score(pipe, df["text"], df["label"], cv=cv, scoring="accuracy")
    print(f"   {name:<12} folds = {np.round(scores, 3)}   mean = {scores.mean():.3f}")

# ----------------------------------------------------------------------------
# Part E : Most informative words
# ----------------------------------------------------------------------------
section("PART E : Top words per class (Naive Bayes, highest log-probability)")
nb = models["Naive Bayes"]
features = nb.named_steps["tfidf"].get_feature_names_out()
log_prob = nb.named_steps["clf"].feature_log_prob_
for i, cls in enumerate(nb.named_steps["clf"].classes_):
    top = np.argsort(log_prob[i])[::-1][:6]
    print(f"   {cls:<11}: {', '.join(features[top])}")

# ----------------------------------------------------------------------------
# Part F : Predict new sentences
# ----------------------------------------------------------------------------
section("PART F : Predictions on new sentences")
new_texts = [
    "The team scored in the final minute to win the cup",
    "The new app uses artificial intelligence to translate languages",
    "Eating vegetables and walking daily keeps your heart healthy",
    "The government will hold a vote on the new law next week",
]
for text in new_texts:
    print(f"\n   '{text}'")
    for name, pipe in models.items():
        print(f"      {name:<12} -> {pipe.predict([text])[0]}")

best = max(results, key=results.get)
print(f"\nBest model on the test set : {best} (accuracy {results[best]:.3f})")
