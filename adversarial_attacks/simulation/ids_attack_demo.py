import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def run_ids_demo(epsilon=1.0):
    # Generate synthetic network traffic data
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
