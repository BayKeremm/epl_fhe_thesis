import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Set the style
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'DejaVu Sans'],
    'font.size': 18,           # Base font size
    'axes.titlesize': 18,      # Title font size
    'axes.labelsize': 18,      # Axis label font size
    'xtick.labelsize': 16,     # X-tick font size
    'ytick.labelsize': 16,     # Y-tick font size 
    'legend.fontsize': 16,     # Legend font size
})   
df = pd.read_csv("./plaintext-domain/results/lfw_tpr_vs_dim.csv")
# Convert to percentages
dimensions = df["dim"].tolist()

# Create a larger figure with better proportions
plt.figure(figsize=(12, 8))

# Create a vibrant but professional color palette
colors = sns.color_palette("viridis", 4)
variance_color = "darkgray"

plt.plot(df["dim"], df["tpr_at_fpr_0001"], 
         label="TPR@0.01%", 
         marker="D", markersize=8, 
         linewidth=2.5, 
         color=colors[0])

plt.plot(df["dim"], df["tpr_at_fpr_001"], 
         label="TPR@0.1%", 
         marker="s", markersize=8, 
         linewidth=2.5, 
         color=colors[1])

plt.plot(df["dim"], df["tpr_at_fpr_01"], 
         label="TPR@1%", 
         marker="^", markersize=8, 
         linewidth=2.5, 
         color=colors[2])

# Plot explained variance with a distinct style
plt.plot(df["dim"], df["ev_mean"], 
         label="Explained Variance", 
         linestyle="--", linewidth=2.5, 
         color=variance_color, alpha=0.8)

# Add subtle grid with lower opacity
plt.grid(True, linestyle='--', alpha=0.6)

# Set better axis limits to focus on the interesting parts of the data
plt.ylim(60, 101)
plt.xlim(min(df["dim"])-5, max(df["dim"])+10)

# Ensure x-ticks show actual dimension values
# plt.xticks(dimensions, [str(d) for d in dimensions], rotation=45)
plt.yticks([i for i in range(20,105,5)])
plt.xticks([i for i in range(5,51,5)])

# Set labels with better formatting
plt.xlabel("PCA Dimension", fontweight='bold')
plt.ylabel("Performance (%)", fontweight='bold')

# Create legend with bold title
legend = plt.legend(title="Performance Metrics", 
                    loc='lower right',
                    frameon=True, 
                    framealpha=0.95,
                    edgecolor='lightgray')

# Make the title bold
title = legend.get_title()
title.set_fontweight('bold')


# Adjust layout
plt.tight_layout()
# plt.savefig("pca_performance_plot.pdf", dpi=300)
plt.show()


