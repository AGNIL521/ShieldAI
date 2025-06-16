import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
import base64, io, pandas as pd

def run_ids_demo(epsilon=1.0, dataset='toy', upload_contents=None):
    # Dataset selection logic
    if dataset == 'toy':
        np.random.seed(42)
        X = np.random.rand(200, 4)
        y = ((X[:, 0] + X[:, 1] > 1) | (X[:, 2] > 0.8)).astype(int)  # 1=attack, 0=benign
    elif dataset == 'nslkdd':
        # Expect NSL-KDD as CSV, last column is label
        try:
            df = pd.read_csv('KDDTrain+.txt', header=None)
            X = df.iloc[:, :-1].values
            y = df.iloc[:, -1].values
        except Exception as e:
            raise RuntimeError(f'Could not load NSL-KDD: {e}')
    elif dataset == 'unsw':
        # Expect UNSW-NB15 as CSV, last column is label
        try:
            df = pd.read_csv('UNSW_NB15_training-set.csv')
            X = df.iloc[:, :-1].values
            y = df.iloc[:, -1].values
        except Exception as e:
            raise RuntimeError(f'Could not load UNSW-NB15: {e}')
    elif dataset == 'upload' and upload_contents:
        content_type, content_string = upload_contents.split(',')
        decoded = base64.b64decode(content_string)
        try:
            if 'csv' in content_type:
                df = pd.read_csv(io.StringIO(decoded.decode()))
            elif 'tsv' in content_type:
                df = pd.read_csv(io.StringIO(decoded.decode()), sep='\t')
            elif 'excel' in content_type or 'xls' in content_type or 'xlsx' in content_type:
                df = pd.read_excel(io.BytesIO(decoded))
            elif 'json' in content_type:
                df = pd.read_json(io.StringIO(decoded.decode()))
            elif 'txt' in content_type:
                df = pd.read_csv(io.StringIO(decoded.decode()), sep=None, engine='python')
            else:
                raise ValueError('Unsupported file type for IDS dataset upload.')
            X = df.iloc[:, :-1].values
            y = df.iloc[:, -1].values
        except Exception as e:
            raise RuntimeError(f'Failed to parse uploaded file: {e}')
    else:
        np.random.seed(42)
        X = np.random.rand(200, 4)
        y = ((X[:, 0] + X[:, 1] > 1) | (X[:, 2] > 0.8)).astype(int)  # 1=attack, 0=benign

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    clf = RandomForestClassifier()
    clf.fit(X_train, y_train)

    # Evaluate on clean test data
    y_pred = clf.predict(X_test)
    clean_acc = accuracy_score(y_test, y_pred)
    print(f"Accuracy on clean IDS data: {clean_acc:.2f}")

    # Adversarial evasion: reduce feature 0 and 1 values to evade detection
    X_test_adv = X_test.copy()
    X_test_adv[:, 0] = X_test_adv[:, 0] * 0.2
    X_test_adv[:, 1] = X_test_adv[:, 1] * 0.2
    y_adv_pred = clf.predict(X_test_adv)
    adv_acc = accuracy_score(y_test, y_adv_pred)
    print(f"Accuracy on adversarial IDS data: {adv_acc:.2f}")
    return clean_acc, adv_acc

if __name__ == "__main__":
    run_ids_demo()
