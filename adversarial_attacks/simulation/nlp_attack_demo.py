# NLP Adversarial Attack Demo: Fooling a Text Classifier
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def add_typos(text):
    # Simple adversarial: replace 'a' with '@', 'e' with '3', etc.
    return text.replace('a', '@').replace('e', '3').replace('o', '0')

# Sample dataset
texts = [
    'free money now', 'win big prize', 'cheap loans', 'urgent offer',
    'hello friend', 'meeting at noon', 'project deadline', 'see you soon'
]
labels = [1, 1, 1, 1, 0, 0, 0, 0]  # 1=spam, 0=ham

X_train, X_test, y_train, y_test = train_test_split(texts, labels, test_size=0.5, random_state=42)

vectorizer = CountVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

clf = MultinomialNB()
clf.fit(X_train_vec, y_train)

# Evaluate on clean test data
y_pred = clf.predict(X_test_vec)
print(f"Accuracy on clean text: {accuracy_score(y_test, y_pred):.2f}")

# Adversarially modify test samples
X_test_adv = [add_typos(t) for t in X_test]
X_test_adv_vec = vectorizer.transform(X_test_adv)
y_adv_pred = clf.predict(X_test_adv_vec)
print(f"Accuracy on adversarial text: {accuracy_score(y_test, y_adv_pred):.2f}")

for orig, adv, true, pred in zip(X_test, X_test_adv, y_test, y_adv_pred):
    print(f"Original: {orig} | Adversarial: {adv} | True: {true} | Predicted: {pred}")
