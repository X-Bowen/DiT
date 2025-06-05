import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

# Config
REAL_DATA_PATH = "/data/yyang409/bowen/imagenet_feature/swin_base/patch4_window7_224/image_features_w_label_train.npz"
GENERATED_PATH = "generated_latents.npz"
SELECTED_CLASSES = [7, 12, 24, 35, 47, 68, 73, 89, 91, 99]
OUTPUT_PATH = "tsne_comparison.png"  # Can be .png, .svg, etc.

def load_and_visualize(output_path=OUTPUT_PATH):
    # Load data
    real_data = np.load(REAL_DATA_PATH)
    generated_data = np.load(GENERATED_PATH)
    
    # Filter real data for selected classes
    real_latents, real_labels = [], []
    for class_idx in SELECTED_CLASSES:
        mask = (real_data["labels"] == class_idx)
        real_latents.append(real_data["features"][mask][:100].reshape(100, -1))
        real_labels.extend([class_idx] * 100)
    
    # Combine data
    combined = np.concatenate([
        np.concatenate(real_latents),
        generated_data["generated"]
    ])
    labels = np.concatenate([
        np.array(real_labels),
        generated_data["labels"]
    ])
    
    # t-SNE
    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    embeddings = tsne.fit_transform(combined)
    
    # Plotting
    plt.figure(figsize=(15, 10))
    colors = plt.cm.tab10(np.linspace(0, 1, len(SELECTED_CLASSES)))
    
    for idx, class_id in enumerate(SELECTED_CLASSES):
        # Real samples (circles)
        mask = (labels == class_id) & (np.arange(len(labels)) < len(real_labels))
        plt.scatter(embeddings[mask, 0], embeddings[mask, 1],
                    color=colors[idx], marker='o', label=f'Class {class_id} (Real)')
        
        # Generated samples (crosses)
        mask = (labels == class_id) & (np.arange(len(labels)) >= len(real_labels))
        plt.scatter(embeddings[mask, 0], embeddings[mask, 1],
                    color=colors[idx], marker='x', label=f'Class {class_id} (Gen)')
    
    plt.title("Real (○) vs Generated (×) Latent Vectors")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    
    # Save and show
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    print(f"Plot saved to {output_path}")
    plt.show()

if __name__ == "__main__":
    load_and_visualize()
