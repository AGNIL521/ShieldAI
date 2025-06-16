import numpy as np
from sklearn.datasets import load_digits
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def randomize_input(X):
    noise = np.random.normal(0, 0.05, X.shape)
    return np.clip(X + noise, 0, 1)

def run_input_randomization_demo():
    data = load_digits()
    X, y = data.data, data.target
    X = X / 16.0

    y_binary = (y == 3).astype(int)
    X_train, X_test, y_train, y_test = train_test_split(X, y_binary, test_size=0.2, random_state=42)

    clf = LogisticRegression(max_iter=500)
    clf.fit(X_train, y_train)

    epsilon = 0.3
    grad = clf.coef_[0]
    X_test_adv = X_test + epsilon * np.sign(grad)
    X_test_adv = np.clip(X_test_adv, 0, 1)

    # Defense: randomize inputs
    X_test_rand = randomize_input(X_test_adv)
    y_pred_rand = clf.predict(X_test_rand)
    acc = accuracy_score(y_test, y_pred_rand)
    print(f'Accuracy on adversarial+randomized inputs: {acc:.2f}')
    return acc

if __name__ == "__main__":
    run_input_randomization_demo()
