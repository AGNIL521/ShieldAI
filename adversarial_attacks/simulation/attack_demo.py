import numpy as np
from sklearn.datasets import load_digits
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import random

def run_attack_demo(plot=True, return_images=False, epsilon=0.3, idx=None):
    """
    Run an adversarial attack on a random or specified test image.
    Args:
        plot (bool): Show plots.
        return_images (bool): Return images for dashboard.
        epsilon (float): Attack strength.
        idx (int or None): Index of test sample to attack. If None, choose randomly.
    Returns:
        fooled (bool), orig_img, adv_img, diff_img (if return_images)
    """

    # Load dataset (simple image classification)
    data = load_digits()
    X, y = data.data, data.target
    X = X / 16.0  # Normalize

    # Binary classification for simplicity: digit 3 vs not-3
    y_binary = (y == 3).astype(int)

    X_train, X_test, y_train, y_test = train_test_split(X, y_binary, test_size=0.2, random_state=42)

    # Train a simple classifier
    clf = LogisticRegression(max_iter=500)
    clf.fit(X_train, y_train)

    # Evaluate on clean test data
    y_pred = clf.predict(X_test)
    print(f'Accuracy on clean test data: {accuracy_score(y_test, y_pred):.2f}')

    # Pick a correctly classified sample
    idxs = np.where((y_test == y_pred) & (y_test == 1))[0]
    if len(idxs) == 0:
        print('No correctly classified "3" found in test set.')
        return False
    idx = idxs[0]
    x_orig = X_test[idx]

    # Generate adversarial example (simple FGSM-like attack)
    epsilon = 0.3
    grad = clf.coef_[0]
    x_adv = x_orig + epsilon * np.sign(grad)
    x_adv = np.clip(x_adv, 0, 1)

    # Predict on adversarial example
    y_adv_pred = clf.predict([x_adv])[0]
    print(f'Original label: 1, Predicted (adversarial): {y_adv_pred}')

    if plot:
        plt.figure(figsize=(10,3))
        plt.subplot(1,3,1)
        plt.title('Original')
        plt.imshow(x_orig.reshape(8,8), cmap='gray')
        plt.axis('off')
        plt.subplot(1,3,2)
        plt.title('Adversarial')
        plt.imshow(x_adv.reshape(8,8), cmap='gray')
        plt.axis('off')
        plt.subplot(1,3,3)
        plt.title('Difference')
        plt.imshow((x_adv-x_orig).reshape(8,8), cmap='bwr')
        plt.axis('off')
        plt.show()

    if return_images:
        return bool(y_adv_pred == 0), x_orig.reshape(8,8), x_adv.reshape(8,8), (x_adv-x_orig).reshape(8,8)
    return bool(y_adv_pred == 0)

if __name__ == "__main__":
    fooled = run_attack_demo(plot=True)
    print(f"Adversarial attack successful? {fooled}")
