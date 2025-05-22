import numpy as np

def evaluate_lfw(embeddings: np.ndarray,
                 labels: np.ndarray,
                 n_folds: int = 10,
                 fpr_targets: list = [0.0001, 0.001, 0.01]):
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

        # Compute distances squared Euclidean
        dists = np.sum((e1 - e2)**2, axis=1)

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
        print(f"TPR @ FPR={fpr*100:.2f}%: {mean_tprs[i]*100:.4f} ± {std_tprs[i]*100:.4f}")

    return mean_tprs, std_tprs


if __name__ == "__main__":
    # Load embeddings and labels
    data = np.load("./data/pair_embeddings_ceci.npz")
    embeddings = data["embeddings"]
    labels = data["issame_list"]

    # Run evaluation
    evaluate_lfw(embeddings, labels)
