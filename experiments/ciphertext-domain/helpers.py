import numpy as np
def generate_inputset(size,max_val_int, vector_dim, min_val = 0):
    return [ ([np.random.randint(min_val, max_val_int+1) for _ in range(vector_dim)],
            [np.random.randint(min_val,max_val_int+1) for _ in range(vector_dim)])
            for _ in range(size)
            ]

def sample_balanced_indices(labels, n_samples_per_class=50):
    # Convert to numpy array if it's not already
    labels = np.array(labels)
    
    # Find indices of True and False values
    true_indices = np.where(labels == True)[0]
    false_indices = np.where(labels == False)[0]
    
    # Check if we have enough samples of each class
    if len(true_indices) < n_samples_per_class:
        raise ValueError(f"Not enough True samples. Requested {n_samples_per_class}, but only {len(true_indices)} available")
    if len(false_indices) < n_samples_per_class:
        raise ValueError(f"Not enough False samples. Requested {n_samples_per_class}, but only {len(false_indices)} available")
    
    # Randomly sample from each class
    sampled_true_indices = np.random.choice(true_indices, size=n_samples_per_class, replace=False)
    sampled_false_indices = np.random.choice(false_indices, size=n_samples_per_class, replace=False)
    
    # Combine and return
    combined_indices = np.concatenate([sampled_true_indices, sampled_false_indices])
    
    # Optional: shuffle the combined indices
    np.random.shuffle(combined_indices)
    
    return combined_indices.tolist()
