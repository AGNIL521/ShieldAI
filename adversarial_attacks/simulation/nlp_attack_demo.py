# NLP Adversarial Attack Demo: Fooling a Text Classifier
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def add_typos(text, perturb_prob):
    """
    Randomly perturb eligible characters in a string with probability perturb_prob.
    Replaces 'a' with '@', 'e' with '3', 'o' with '0'.
    """
    import random
    chars = list(text)
    for i, c in enumerate(chars):
        if c in 'aeo' and random.random() < perturb_prob:
            chars[i] = {'a': '@', 'e': '3', 'o': '0'}[c]
    return ''.join(chars)


def run_nlp_demo(perturb_prob=0.3, dataset='toy', upload_contents=None):
    """
    Run NLP adversarial attack demo with selectable dataset.
    Args:
        perturb_prob (float): Probability of perturbing each eligible char.
        dataset (str): Which dataset to use ('toy', 'sms', '20news', 'upload').
        upload_contents: base64-encoded uploaded file contents (if any).
    Returns: clean_acc, adv_acc, results
    """
    import numpy as np
    import base64
    import io
    import pandas as pd
    from sklearn.datasets import fetch_20newsgroups
    # Dataset selection logic
    if dataset == 'toy':
        texts = [
            'free money now', 'win big prize', 'cheap loans', 'urgent offer',
            'hello friend', 'meeting at noon', 'project deadline', 'see you soon'
        ]
        labels = [1, 1, 1, 1, 0, 0, 0, 0]
    elif dataset == 'sms':
        # Expect upload_contents to be SMS Spam Collection file
        if upload_contents:
            content_type, content_string = upload_contents.split(',')
            decoded = base64.b64decode(content_string)
            df = pd.read_csv(io.StringIO(decoded.decode()), sep='\t', header=None, names=['label', 'text'])
            texts = df['text'].tolist()
            labels = [1 if l=='spam' else 0 for l in df['label']]
        else:
            texts = [
                'free money now', 'win big prize', 'cheap loans', 'urgent offer',
                'hello friend', 'meeting at noon', 'project deadline', 'see you soon'
            ]
            labels = [1, 1, 1, 1, 0, 0, 0, 0]
    elif dataset == '20news':
        newsgroups = fetch_20newsgroups(subset='train', remove=('headers','footers','quotes'))
        texts = newsgroups.data[:100]
        labels = newsgroups.target[:100]
    elif dataset == 'upload' and upload_contents:
        content_type, content_string = upload_contents.split(',')
        decoded = base64.b64decode(content_string)
        # Try to auto-detect file type
        try:
            if 'csv' in content_type or 'text/csv' in content_type:
                df = pd.read_csv(io.StringIO(decoded.decode()))
            elif 'tsv' in content_type or 'text/tab-separated-values' in content_type:
                df = pd.read_csv(io.StringIO(decoded.decode()), sep='\t')
            elif 'txt' in content_type or '.txt' in content_type:
                # Try tab, then comma, then whitespace
                try:
                    df = pd.read_csv(io.StringIO(decoded.decode()), sep='\t')
                except Exception:
                    try:
                        df = pd.read_csv(io.StringIO(decoded.decode()), sep=',')
                    except Exception:
                        df = pd.read_csv(io.StringIO(decoded.decode()), delim_whitespace=True)
            elif 'excel' in content_type or 'xls' in content_type or 'xlsx' in content_type:
                df = pd.read_excel(io.BytesIO(decoded))
            elif 'json' in content_type:
                df = pd.read_json(io.StringIO(decoded.decode()))
            else:
                raise ValueError('Unsupported file type for text dataset upload.')
            # Assume first column is label, second is text
            texts = df.iloc[:,1].astype(str).tolist()
            labels = df.iloc[:,0].tolist()
        except Exception as e:
            raise RuntimeError(f'Failed to parse uploaded file: {e}')
    else:
        texts = [
            'free money now', 'win big prize', 'cheap loans', 'urgent offer',
            'hello friend', 'meeting at noon', 'project deadline', 'see you soon'
        ]
        labels = [1, 1, 1, 1, 0, 0, 0, 0]


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
