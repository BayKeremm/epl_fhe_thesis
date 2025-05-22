import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from matplotlib.ticker import MaxNLocator

def plot_variance_explained(embeddings, max_components=128, 
                            thresholds=[0.9, 0.95], figsize=(14, 7)): 
    plt.style.use('seaborn-v0_8-whitegrid')

    plt.rcParams.update({
        'font.size': 18,           # Base font size
        'axes.titlesize': 18,      # Title font size
        'axes.labelsize': 18,      # Axis label font size
        'xtick.labelsize': 16,     # X-tick font size
        'ytick.labelsize': 16,     # Y-tick font size 
        'legend.fontsize': 16,     # Legend font size
    })   
    colors = {
        'main_line': '#1f77b4',  # Blue
        'threshold': '#d62728',   # Red
        'annotation': '#2ca02c',  # Green
        'drop_line': '#ff7f0e'    # Orange
    }
    
    # Limit max_components to number of features
    n_features = embeddings.shape[1]
    max_components = min(max_components, n_features)
    
    # Fit PCA
    pca = PCA(n_components=max_components)
    pca.fit(embeddings)
    
    # Calculate cumulative explained variance
    cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # Plot 1: Elbow curve (cumulative explained variance)
    ax1.plot(range(1, max_components + 1), cumulative_variance, 
             marker='o', linestyle='-', markersize=4, color=colors['main_line'],
             linewidth=2, alpha=0.9)
    
    ax1.xaxis.set_major_locator(MaxNLocator(integer=True))
    if max_components > 20:
        step = max(1, max_components // 10)
        ax1.set_xticks(np.arange(1, max_components + 1, step))
    
    ax1.set_xlabel('Number of Components', fontsize=16, fontweight='bold')
    ax1.set_ylabel('Cumulative Explained Variance', fontsize=16, fontweight='bold')
    ax1.set_title('Elbow Curve', fontsize=16, fontweight='bold')
    
    # Add threshold lines and annotations
    components_for_thresholds = {}
    i =0
    for threshold in thresholds:
        #ax1.axhline(y=threshold, color=colors['threshold'], linestyle='--', alpha=0.7, linewidth=1.5)
        
        # Find the first component that exceeds this threshold
        component_idx = np.where(cumulative_variance >= threshold)[0][0] + 1
        components_for_thresholds[threshold] = component_idx
        
        if i == 0:
            # Add an arrow pointing to where the threshold is crossed
            ax1.annotate(f'{threshold*100:.0f}% variance: {component_idx} components',
                    xy=(component_idx, threshold),
                    xytext=(component_idx + max_components*0.1, threshold - 0.05),
                    arrowprops=dict(facecolor=colors['annotation'], shrink=0.05, width=1.5, headwidth=7),
                    fontsize=14,
                    bbox=dict(boxstyle="round,pad=0.2", facecolor='white', alpha=0.8))
            i += 1
        else:
            # Add an arrow pointing to where the threshold is crossed
            ax1.annotate(f'{threshold*100:.0f}% variance: {component_idx} components',
                    xy=(component_idx, threshold),
                    xytext=(component_idx + max_components*0.1, threshold - 0.02),
                    arrowprops=dict(facecolor=colors['annotation'], shrink=0.05, width=1.5, headwidth=7),
                    fontsize=14,
                    bbox=dict(boxstyle="round,pad=0.2", facecolor='white', alpha=0.8))
    
    # Plot 2: Scree plot (individual explained variance)
    ax2.plot(range(1, max_components + 1), pca.explained_variance_ratio_, 
             marker='o', markersize=4, linestyle='-', color=colors["main_line"],
             linewidth=2, alpha=0.7)
    
    # Set x-ticks for better readability
    ax2.xaxis.set_major_locator(MaxNLocator(integer=True))
    if max_components > 20:
        step = max(1, max_components // 10)
        ax2.set_xticks(np.arange(1, max_components + 1, step))
    
    # Enhance axis labels and title
    ax2.set_xlabel('Number of Components', fontsize=16, fontweight='bold')
    ax2.set_ylabel('Individual Explained Variance', fontsize=16, fontweight='bold')
    ax2.set_title('Scree Plot', fontsize=16, fontweight='bold')
    
    # Add annotation for significant drop
    drop_threshold = 0.01/2
    drop_indices = np.where(pca.explained_variance_ratio_ < drop_threshold)[0]
    if len(drop_indices) > 0:
        drop_idx = drop_indices[0]-1
        # ax2.axvline(x=drop_idx + 1, color=colors['drop_line'], linestyle='--', linewidth=1.5)
        ax2.annotate(f'Significant drop after\ncomponent {drop_idx}',
                    xy=(drop_idx + 1, pca.explained_variance_ratio_[drop_idx]),
                    xytext=(drop_idx + 1 + max_components*0.1, pca.explained_variance_ratio_[0] * 0.5),
                    arrowprops=dict(facecolor=colors['drop_line'], shrink=0.05, width=1.5, headwidth=8),
                    fontsize=14,
                    bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))
    
    # Add grid but make it subtle
    ax1.grid(True, linestyle='--', linewidth=0.5, alpha=0.3)
    ax2.grid(True, linestyle='--', linewidth=0.5, alpha=0.3)
    
    # Adjust layout and add space for the annotation
    plt.tight_layout(rect=[0, 0.05, 1, 0.95]) #type:ignore
    
    return fig, pca

if __name__ == "__main__":
    data = np.load("./data/pair_embeddings_ceci.npz")
    embeddings = data["embeddings"]
    print(f"Embeddings shape: {embeddings.shape}")
    
    # Create the publication-quality plots
    fig, pca_model = plot_variance_explained(
        embeddings, 
        max_components=100,
        thresholds=[0.9, 0.95]
    )
    
    # Save with high resolution
    # plt.savefig('pca_variance_analysis.pdf', dpi=300, bbox_inches='tight')
    plt.show()
