import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.metrics import roc_curve
import os

def distance(embeddings1, embeddings2):
    # Ensure float math to prevent wraparound
    diff = np.asarray(embeddings1, dtype=np.float32) - np.asarray(embeddings2, dtype=np.float32)
    return np.sum(diff ** 2, axis=1)

def get_roc_metrics(distances, labels):
    # For Euclidean, we use negative distances as smaller distances mean more similar
    distances = -distances
    fpr, tpr, thresholds = roc_curve(labels, distances)
    j_scores = tpr - fpr
    best_threshold = thresholds[np.argmax(j_scores)]
    return -best_threshold


def quantize_simple(embeds1, embeds2, bits=8):
    """Simple quantization to fixed bit precision"""
    assert np.all(embeds1 >= 0)
    assert np.all(embeds2 >= 0)
    min_val = min(np.min(embeds1), np.min(embeds2))
    max_val = max(np.max(embeds1), np.max(embeds2))
    scale = (2**bits - 1) / (max_val - min_val)

    dtype = np.uint8 if bits <= 8 else np.uint16
    max_qval = 2**bits - 1

    quantized1 = np.clip(np.round(embeds1  * scale), 0, max_qval).astype(dtype)
    quantized2 = np.clip(np.round(embeds2  * scale), 0, max_qval).astype(dtype)

    return quantized1, quantized2, scale

def generate_inputset(size,max_val_int, vector_dim, min_val_int = 0):
    return [ ([np.random.randint(min_val_int, max_val_int+1) for _ in range(vector_dim)],
            [np.random.randint(min_val_int,max_val_int+1) for _ in range(vector_dim)])
            for _ in range(size)
            ]

def preprocess(embeds_in, dim, minmax):
    pca = PCA(n_components=dim)
    embeds_ = pca.fit_transform(embeds_in)
    if minmax:
        m = np.min(embeds_)
        M = np.max(embeds_)
        embeds_ = np.array([(embed-m)/(M-m) for embed in embeds_])
        # scaler = MinMaxScaler()
        # embeds_ = scaler.fit_transform(embeds_)
    return embeds_, sum(pca.explained_variance_ratio_) * 100

def clear_console():
    os.system('cls' if os.name == 'nt' else 'clear')

