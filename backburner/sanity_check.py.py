import numpy as np
from sklearn.decomposition import PCA


def evaluate_lfw(embeddings: np.ndarray,
                 labels: np.ndarray,
                 n_folds: int = 10,
                 pca_dim: int = 44,
                 fpr_targets: list = [0.0001, 0.001, 0.01]):
    """
    Perform 10-fold cross-validation on LFW pair embeddings after preprocessing pipeline
    then computes TPR at specified FPR targets on each test fold.

    Args:
        embeddings: Array of shape (2*N_pairs, D) containing paired embeddings.
        labels: Boolean array of shape (N_pairs,) indicating if pairs match.
        n_folds: Number of cross-validation folds (default 10).
        pca_dim: Number of PCA dimensions to reduce to (default 44).
        fpr_targets: List of false-positive rates at which to evaluate TPR.

    Returns:
        mean_tprs: Array of shape (len(fpr_targets),) with mean TPRs across folds.
        std_tprs: Array of shape (len(fpr_targets),) with std dev of TPRs across folds.
    """
    # Split embeddings into pairs
    e1 = embeddings[0::2]
    e2 = embeddings[1::2]
    labels = np.asarray(labels, dtype=bool)
    n_pairs = labels.shape[0]
    fold_size = n_pairs // n_folds

    # Store TPRs for each fold and target
    tprs = np.zeros((len(fpr_targets), n_folds))

    # Cross-validation
    for fold in range(n_folds):
        start = fold * fold_size
        end = start + fold_size
        test_idx = np.arange(start, end)
        train_idx = np.concatenate((np.arange(0, start), np.arange(end, n_pairs)))

        # Train PCA on all embeddings from training folds
        train_feats = np.vstack((e1[train_idx], e2[train_idx]))
        pca = PCA(n_components=pca_dim)
        pca.fit(train_feats)

        # Transform all embeddings
        proj1 = pca.transform(e1)
        proj2 = pca.transform(e2)

        m = min(np.min(proj1), np.min(proj2))
        M = max(np.max(proj2), np.max(proj2))

        proj1 = (proj1 - m) / (M-m)
        proj2 = (proj2 - m) / (M-m)

        bits     = 4
        max_qval = 2**bits - 1
        dtype = np.uint8 if bits <= 8 else np.uint16
        scale = max_qval

        quantized1 = np.clip(np.round(proj1  * scale), 0, max_qval).astype(dtype)
        quantized2 = np.clip(np.round(proj2  * scale), 0, max_qval).astype(dtype)

        # Compute distances squared Euclidean
        dists = np.sum((quantized1 - quantized2)**2, axis=1)

        # Prepare training distances for threshold selection
        train_dists = dists[train_idx]
        train_labels = labels[train_idx]
        neg_train = train_dists[~train_labels]

        # Test distances and labels
        test_dists = dists[test_idx]
        test_labels = labels[test_idx]

        # Evaluate each FPR target
        for j, fpr in enumerate(fpr_targets):
            # Choose threshold so that fraction of neg_train < thr equals target FPR
            thr = np.percentile(neg_train, 100 * fpr)
            # True-positive rate on test set
            pos = test_labels
            tprs[j, fold] = np.mean(test_dists[pos] < thr)

    # Aggregate results
    mean_tprs = tprs.mean(axis=1)
    std_tprs = tprs.std(axis=1)


    # Display results
    for i, fpr in enumerate(fpr_targets):
        print(f"TPR @ FPR={fpr*100:.2f}%: {mean_tprs[i]:.4f} ± {std_tprs[i]:.4f}")

    return mean_tprs, std_tprs


if __name__ == "__main__":
    # Load embeddings and labels
    data = np.load("./experiments/facenet/pair_embeddings_ceci.npz")
    embeddings = data["embeddings"]
    labels = data["issame_list"]

    # Run evaluation
    evaluate_lfw(embeddings, labels)
