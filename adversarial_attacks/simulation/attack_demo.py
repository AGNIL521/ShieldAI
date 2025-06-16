import numpy as np
from sklearn.datasets import load_digits
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import random
import base64
import io
from tensorflow.keras.datasets import fashion_mnist, cifar10

def run_attack_demo(plot=True, return_images=False, epsilon=0.3, idx=None, dataset='digits', upload_contents=None):
    """
    Run an adversarial attack on a random or specified test image.
    Args:
        plot (bool): Show plots.
        return_images (bool): Return images for dashboard.
        epsilon (float): Attack strength.
        idx (int or None): Index of test sample to attack. If None, choose randomly.
        dataset (str): Which dataset to use ('digits', 'mnist', 'fashion-mnist', 'cifar10', 'upload').
        upload_contents: base64-encoded uploaded file contents (if any).
    Returns:
        fooled (bool), orig_img, adv_img, diff_img (if return_images)
    """

    # Dataset selection logic
    if dataset == 'digits':
        data = load_digits()
        X, y = data.data, data.target
    elif dataset == 'mnist':
        mnist = fetch_openml('mnist_784', version=1, as_frame=False)
        X, y = mnist['data'], mnist['target'].astype(int)
    elif dataset == 'fashion-mnist':
        (X, y), _ = fashion_mnist.load_data()
        X = X.reshape((X.shape[0], -1))
    elif dataset == 'cifar10':
        (X, y), _ = cifar10.load_data()
        X = X.reshape((X.shape[0], -1))
        y = y.flatten()
    elif dataset == 'upload' and upload_contents:
        content_type, content_string = upload_contents.split(',')
        decoded = base64.b64decode(content_string)
        # Try to auto-detect file type
        try:
            if 'csv' in content_type or 'text/csv' in content_type:
                import pandas as pd
                df = pd.read_csv(io.StringIO(decoded.decode()))
                X = df.iloc[:, :-1].values
                y = df.iloc[:, -1].values
            elif 'tsv' in content_type or 'text/tab-separated-values' in content_type:
                import pandas as pd
                df = pd.read_csv(io.StringIO(decoded.decode()), sep='\t')
                X = df.iloc[:, :-1].values
                y = df.iloc[:, -1].values
            elif 'excel' in content_type or 'xls' in content_type or 'xlsx' in content_type:
                import pandas as pd
                df = pd.read_excel(io.BytesIO(decoded))
                X = df.iloc[:, :-1].values
                y = df.iloc[:, -1].values
            elif 'json' in content_type:
                import pandas as pd
                df = pd.read_json(io.StringIO(decoded.decode()))
                X = df.iloc[:, :-1].values
                y = df.iloc[:, -1].values
            elif 'zip' in content_type or '.zip' in content_type:
                import zipfile
                with zipfile.ZipFile(io.BytesIO(decoded)) as zf:
                    filelist = zf.namelist()
                    raise ValueError(f'ZIP archive preview: {filelist[:5]} ... (image ZIP support not yet implemented)')
            elif 'npy' in content_type or '.npy' in content_type:
                X = np.load(io.BytesIO(decoded))
                y = np.zeros(X.shape[0])
            elif 'npz' in content_type or '.npz' in content_type:
                npzfile = np.load(io.BytesIO(decoded))
                X = npzfile['X'] if 'X' in npzfile else npzfile[list(npzfile.keys())[0]]
                y = npzfile['y'] if 'y' in npzfile else np.zeros(X.shape[0])
            elif 'txt' in content_type or '.txt' in content_type:
                arr = np.loadtxt(io.StringIO(decoded.decode()))
                if arr.ndim == 1:
                    X = arr.reshape(-1, 1)
                    y = np.zeros(X.shape[0])
                else:
                    X = arr[:, :-1]
                    y = arr[:, -1]
            else:
                raise ValueError('Unsupported file type for image dataset upload.')
        except Exception as e:
            raise RuntimeError(f'Failed to parse uploaded file: {e}')
    else:
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
