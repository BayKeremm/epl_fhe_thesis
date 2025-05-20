import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import roc_curve
# from scipy.interpolate import interp1d # np.interp is generally sufficient and simpler

def calculate_tpr_at_fprs(y_true, y_scores, target_fprs):
    """
    Calculates True Positive Rates (TPR) at specified False Positive Rates (FPR).

    Args:
        y_true (np.array): True binary labels.
        y_scores (np.array): Target scores, where higher values indicate greater
                             likelihood of the positive class.
        target_fprs (list of float): A list of FPR values at which to calculate TPR.

    Returns:
        dict: A dictionary mapping each target FPR to its corresponding TPR.
    """
    fpr_actual, tpr_actual, thresholds = roc_curve(y_true, y_scores)
    
    tprs_at_targets = {}
    for target_fpr in target_fprs:
        # np.interp finds the y-value corresponding to a given x-value in a 1-D function.
        # It requires fpr_actual to be sorted, which roc_curve output generally is.
        tprs_at_targets[target_fpr] = np.interp(target_fpr, fpr_actual, tpr_actual)
        
    return tprs_at_targets

def evaluate_embeddings_lfw_protocol():
    """
    Performs 10-fold cross-validation on face embeddings using LFW pairs.txt protocol,
    reduces dimensionality with PCA, and calculates TPR@FPR.
    """
    # 1. Load data
    # Assuming your .npz file is in a directory named 'facenet' relative to your script
    # or you provide the correct path.
    try:
        data = np.load("./experiments/facenet/pair_embeddings_ceci.npz")
    except FileNotFoundError:
        print("Error: The file './facenet/pair_embeddings_ceci.npz' was not found.")
        print("Please ensure the path to your .npz file is correct.")
        return

    embeddings_raw, labels_bool = data["embeddings"], data["issame_list"]
    
    # The 'embeddings' array contains e1_pair1, e2_pair1, e1_pair2, e2_pair2, ...
    # We split them into e1_all (all first embeddings of pairs) and e2_all (all second embeddings of pairs)
    e1_all = embeddings_raw[0::2]
    e2_all = embeddings_raw[1::2]
    labels_all = labels_bool.astype(int) # Convert boolean labels to int (0 or 1)

    num_total_pairs = len(labels_all)
    if num_total_pairs != 6000: # Standard LFW has 6000 pairs
        print(f"Warning: Expected 6000 pairs for LFW, but found {num_total_pairs}.")

    num_folds = 10
    if num_total_pairs % num_folds != 0:
        print(f"Error: Total number of pairs ({num_total_pairs}) is not divisible by num_folds ({num_folds}).")
        return
        
    pairs_per_fold = num_total_pairs // num_folds
    target_dimension = 44
    # Target FPRs: 0.01%, 0.1%, 1%
    target_fpr_values = [0.0001, 0.001, 0.01] 

    all_fold_tprs = {fpr: [] for fpr in target_fpr_values}

    print(f"Starting 10-fold cross-validation with {num_total_pairs} pairs.")
    print(f"Dimensionality reduction to {target_dimension} dimensions using PCA (whiten=True).")

    for i in range(num_folds):
        print(f"\nProcessing fold {i+1}/{num_folds}...")

        # 2. Split data into training and test sets for the current fold
        test_start_idx = i * pairs_per_fold
        test_end_idx = (i + 1) * pairs_per_fold

        # Test set pairs
        e1_test = e1_all[test_start_idx:test_end_idx]
        e2_test = e2_all[test_start_idx:test_end_idx]
        labels_test = labels_all[test_start_idx:test_end_idx]

        # Training set pairs (all other folds)
        train_indices = np.array(list(range(0, test_start_idx)) + list(range(test_end_idx, num_total_pairs)))
        if len(train_indices) == 0: # Handles case where num_total_pairs might be small (e.g. only 1 fold)
             # For PCA training, if there's only one fold, technically no separate training data for PCA
             # This setup assumes num_folds > 1.
             # If using all data for PCA training in case of single fold evaluation:
             # training_embeddings_for_pca = np.vstack((e1_all, e2_all))
             # However, LFW has 10 folds, so this branch is unlikely to be hit with correct data.
            if num_folds > 1 :
                 print(f"Warning: train_indices is empty for fold {i+1}. This might indicate an issue.")
                 training_embeddings_for_pca = np.vstack((e1_test, e2_test)) # Fallback, not ideal CV
            else: # Only one fold total
                 training_embeddings_for_pca = np.vstack((e1_all, e2_all))

        else:
            e1_train_fold_pairs = e1_all[train_indices]
            e2_train_fold_pairs = e2_all[train_indices]
            # PCA is trained on individual embeddings from the training folds
            training_embeddings_for_pca = np.vstack((e1_train_fold_pairs, e2_train_fold_pairs))


        # 3. Fit PCA on training embeddings
        pca = PCA(n_components=target_dimension, whiten=True, random_state=42) # random_state for reproducibility
        if training_embeddings_for_pca.shape[0] < target_dimension:
            print(f"  Warning: Number of training samples ({training_embeddings_for_pca.shape[0]}) is less than target dimensions ({target_dimension}). PCA might not be optimal.")
            # Adjust n_components if needed, or proceed with caution.
            # For now, we proceed, but PCA might behave unexpectedly.
            current_target_dim = min(target_dimension, training_embeddings_for_pca.shape[0])
            if current_target_dim == 0: # Should not happen with LFW data
                print("  Error: No training samples for PCA. Skipping fold.")
                for fpr_val in target_fpr_values: all_fold_tprs[fpr_val].append(np.nan) # Record NaN
                continue
            pca = PCA(n_components=current_target_dim, whiten=True, random_state=42)


        pca.fit(training_embeddings_for_pca)

        # 4. Transform test embeddings
        e1_test_reduced = pca.transform(e1_test)
        e2_test_reduced = pca.transform(e2_test)

        # 5. Calculate distances and scores for the test set
        # Euclidean distance: lower means more similar
        distances = np.linalg.norm(e1_test_reduced - e2_test_reduced, axis=1)
        # Scores for roc_curve: higher means more similar (positive class)
        scores = -distances 

        # 6. Calculate TPR at specified FPRs for the current fold
        # labels_test should be 1 for "issame" (positive class), 0 for "notsame"
        if len(np.unique(labels_test)) < 2:
            print(f"  Warning: Fold {i+1} test set has only one class. ROC analysis may not be meaningful.")
            tprs_current_fold = {fpr_val: np.nan for fpr_val in target_fpr_values}
        else:
            tprs_current_fold = calculate_tpr_at_fprs(labels_test, scores, target_fpr_values)

        for fpr_val, tpr_val in tprs_current_fold.items():
            all_fold_tprs[fpr_val].append(tpr_val)
        
        fold_summary = ", ".join([f"TPR@{f*100:.2f}%FPR: {v:.4f}" if not np.isnan(v) else f"TPR@{f*100:.2f}%FPR: NaN" for f,v in tprs_current_fold.items()])
        print(f"  Fold {i+1} results: {fold_summary}")

    # 7. Average results across folds
    print("\n--- Evaluation Summary ---")
    print(f"Target dimension after PCA: {target_dimension}")
    print("Average TPR @ specified FPRs (and Standard Deviation) over 10 folds:")

    for fpr_val in target_fpr_values:
        tpr_list = [t for t in all_fold_tprs[fpr_val] if not np.isnan(t)] # Exclude NaNs for mean/std calculation
        if not tpr_list: # All were NaN
            mean_tpr = np.nan
            std_dev_tpr = np.nan
        else:
            mean_tpr = np.mean(tpr_list)
            std_dev_tpr = np.std(tpr_list)
            
        fpr_percent = fpr_val * 100
        print(f"  TPR @ {fpr_percent:.2f}% FPR: {mean_tpr:.4f} (std: {std_dev_tpr:.4f})")

if __name__ == '__main__':
    evaluate_embeddings_lfw_protocol()
