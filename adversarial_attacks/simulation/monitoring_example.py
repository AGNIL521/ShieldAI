import numpy as np
from sklearn.metrics import pairwise_distances

def detect_adversarial(inputs, threshold=2.0):
    # Flag inputs that are far from the mean (simple anomaly detection)
    mean = np.mean(inputs, axis=0)
    dists = pairwise_distances(inputs, [mean]).flatten()
    return np.where(dists > threshold)[0]

def run_monitoring_demo():
    np.random.seed(0)
    X = np.random.rand(10, 5)
    # Inject an outlier (simulated adversarial)
    X[5] = X[5] + 5
    flags = detect_adversarial(X)
    print(f"Potential adversarial samples at indices: {flags}")
    return flags

if __name__ == "__main__":
    run_monitoring_demo()
