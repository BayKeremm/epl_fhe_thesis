import numpy as np
from sklearn.decomposition import PCA
import pandas as pd

def evaluate_lfw(embeddings: np.ndarray,
                 labels: np.ndarray,
                 n_folds: int = 10,
                 pca_dim: int = 44,
                 fpr_targets: list = [0.0001, 0.001, 0.01]):
    # Split embeddings into pairs
    e1 = embeddings[0::2]
    e2 = embeddings[1::2]
    labels = np.asarray(labels, dtype=bool)
    n_pairs = labels.shape[0]
    fold_size = n_pairs // n_folds

    # Store TPRs for each fold and target
    tprs = np.zeros((len(fpr_targets), n_folds))

    explained_vrcs = np.zeros((n_folds,))

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
        explained_vrcs[fold] = sum(pca.explained_variance_ratio_) * 100

        # Transform all embeddings
        proj1 = pca.transform(e1)
        proj2 = pca.transform(e2)

        # Compute distances squared Euclidean
        dists = np.sum((proj1 - proj2)**2, axis=1)

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

    return mean_tprs, std_tprs, np.mean(explained_vrcs), np.std(explained_vrcs)

if __name__ == "__main__":
    # Load embeddings and labels
    data = np.load("./data/pair_embeddings_ceci.npz")
    embeddings = data["embeddings"]
    labels = data["issame_list"]
    
    dimensions = range(10,51)
    fpr_targets = [0.0001, 0.001, 0.01]
    results = []

    for dim in dimensions:
        print()
        print(f"Results for dimension={dim}:")
        # Run evaluation
        mean_tprs,std_tprs, ev_mean, ev_std = evaluate_lfw(embeddings, labels,
                                          pca_dim=dim,
                                          fpr_targets=fpr_targets)
        # FOR PLOTTING
        results.append({
            "dim": dim,
            "tpr_at_fpr_0001": round(mean_tprs[0] * 100, 2),
            "tpr_at_fpr_0001_std": round(std_tprs[0] * 100, 2),
            "tpr_at_fpr_001": round(mean_tprs[1] * 100, 2),
            "tpr_at_fpr_001_std": round(std_tprs[1] * 100, 2),
            "tpr_at_fpr_01": round(mean_tprs[2] * 100, 2),
            "tpr_at_fpr_01_std": round(std_tprs[2] * 100, 2),
            "ev_mean": round(ev_mean, 2),
            "ev_std": round(ev_std * 100, 2),
        })

        # FOR LATEX
        # results.append({
        #     "dim": dim,
        #     "TPR@0.01%": str(round(mean_tprs[0] * 100, 2))+"$\\pm$"+str(round(std_tprs[0] * 100, 2)),
        #     "TPR@0.1%": str(round(mean_tprs[1] * 100, 2))+"$\\pm$"+str(round(std_tprs[1] * 100, 2)),
        #     "TPR@1%": str(round(mean_tprs[2] * 100, 2))+"$\\pm$"+str(round(std_tprs[2] * 100, 2)),
        #     "Variance": str(round(ev_mean, 2))+"$\\pm$"+str(round(ev_std * 100, 2))
        # })
    # Convert to DataFrame and save
    df = pd.DataFrame(results)
    df.to_csv("./plaintext-domain/results/lfw_tpr_vs_dim.csv", index=False)
    print("\nSaved results to lfw_tpr_vs_dim.csv")
