# NLP Adversarial Attack Demo: Fooling a Text Classifier
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def add_typos(text, perturb_prob):
    # Simple adversarial: replace 'a' with '@', 'e' with '3', etc.
    if np.random.rand() < perturb_prob:
        return text.replace('a', '@').replace('e', '3').replace('o', '0')
    else:
        return text

def run_nlp_demo(perturb_prob=0.3):
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
    clean_acc = accuracy_score(y_test, y_pred)
    print(f"Accuracy on clean text: {clean_acc:.2f}")

    # Adversarially modify test samples
    X_test_adv = [add_typos(t, perturb_prob) for t in X_test]
    X_test_adv_vec = vectorizer.transform(X_test_adv)
    y_adv_pred = clf.predict(X_test_adv_vec)
    adv_acc = accuracy_score(y_test, y_adv_pred)
    print(f"Accuracy on adversarial text: {adv_acc:.2f}")

    results = []
    for orig, adv, true, pred in zip(X_test, X_test_adv, y_test, y_adv_pred):
        print(f"Original: {orig} | Adversarial: {adv} | True: {true} | Predicted: {pred}")
        results.append((orig, adv, true, pred))
    return clean_acc, adv_acc, results

if __name__ == "__main__":
    run_nlp_demo()
