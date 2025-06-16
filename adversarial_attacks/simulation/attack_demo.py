import numpy as np
from sklearn.datasets import load_digits
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

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
idx = np.where((y_test == y_pred) & (y_test == 1))[0][0]
x_orig = X_test[idx]

# Generate adversarial example (simple FGSM-like attack)
epsilon = 0.3
grad = clf.coef_[0]
x_adv = x_orig + epsilon * np.sign(grad)

# Clip to valid range
x_adv = np.clip(x_adv, 0, 1)

# Predict on adversarial example
y_adv_pred = clf.predict([x_adv])[0]
print(f'Original label: 1, Predicted (adversarial): {y_adv_pred}')

# Visualize original and adversarial samples
plt.subplot(1, 2, 1)
plt.title('Original')
plt.imshow(x_orig.reshape(8, 8), cmap='gray')
plt.axis('off')
plt.subplot(1, 2, 2)
plt.title('Adversarial')
plt.imshow(x_adv.reshape(8, 8), cmap='gray')
plt.axis('off')
plt.show()
