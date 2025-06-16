import numpy as np
from sklearn.datasets import load_digits
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def run_adversarial_training_demo():
    data = load_digits()
    X, y = data.data, data.target
    X = X / 16.0

    y_binary = (y == 3).astype(int)
    X_train, X_test, y_train, y_test = train_test_split(X, y_binary, test_size=0.2, random_state=42)

    clf = LogisticRegression(max_iter=500)
    clf.fit(X_train, y_train)

    # Generate adversarial examples for training set
    epsilon = 0.3
    grad = clf.coef_[0]
    X_train_adv = X_train + epsilon * np.sign(grad)
    X_train_adv = np.clip(X_train_adv, 0, 1)

    # Combine original and adversarial samples
    X_train_combined = np.vstack([X_train, X_train_adv])
    y_train_combined = np.hstack([y_train, y_train])

    clf_adv = LogisticRegression(max_iter=500)
    clf_adv.fit(X_train_combined, y_train_combined)

    y_pred_clean = clf_adv.predict(X_test)
    y_pred_adv = clf_adv.predict(X_test + epsilon * np.sign(grad))

    clean_acc = accuracy_score(y_test, y_pred_clean)
    adv_acc = accuracy_score(y_test, y_pred_adv)
    print(f'Adversarially trained accuracy on clean: {clean_acc:.2f}')
    print(f'Adversarially trained accuracy on adversarial: {adv_acc:.2f}')
    return clean_acc, adv_acc

if __name__ == "__main__":
    run_adversarial_training_demo()
