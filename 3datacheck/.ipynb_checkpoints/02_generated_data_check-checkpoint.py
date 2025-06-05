import numpy as np
import h5py
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.metrics import pairwise_distances_argmin_min
import time

def analyze_generated_data(real_npz_path, generated_h5_path, num_classes=10, samples_per_class=100):
    """Analyze and compare real vs generated latent vectors with class-wise comparison"""
    start_time = time.time()
    
    # Load real data from NPZ
    with np.load(real_npz_path) as real_data:
        real_features = real_data['features']
        real_labels = real_data['labels']
    
    # Load generated data from HDF5
    with h5py.File(generated_h5_path, 'r') as gen_data:
        gen_features = gen_data['samples'][:]
        gen_labels = gen_data['labels'][:]
    
    print("=== Dataset Structure ===")
    print(f"Real data: {real_features.shape} features, {real_labels.shape} labels")
    print(f"Generated data: {gen_features.shape} samples, {gen_labels.shape} labels")
    
    # Select random classes for visualization
    unique_classes = np.intersect1d(np.unique(real_labels), np.unique(gen_labels))
    selected_classes = np.random.choice(unique_classes, num_classes, replace=False)
    
    # Prepare subsets with balanced class representation
    real_samples, gen_samples = [], []
    for class_idx in selected_classes:
        # Real data
        real_mask = real_labels == class_idx
        real_class = real_features[real_mask][:samples_per_class]
        
        # Generated data
        gen_mask = gen_labels == class_idx
        gen_class = gen_features[gen_mask][:samples_per_class]
        
        real_samples.append(real_class)
        gen_samples.append(gen_class)
    
    real_sub = np.vstack(real_samples)
    gen_sub = np.vstack(gen_samples)
    combined = np.vstack([real_sub, gen_sub])
    
    # Create labels for visualization
    combined_labels = np.concatenate([
        np.repeat(selected_classes, samples_per_class),
        np.repeat(selected_classes, samples_per_class)
    ])
    dataset_labels = np.concatenate([
        np.zeros(len(real_sub)),  # 0 = real
        np.ones(len(gen_sub))     # 1 = generated
    ])
    
    # Dimensionality reduction
    print("\n=== Dimensionality Reduction ===")
    pca = PCA(n_components=50)
    pca_result = pca.fit_transform(combined)
    print(f"Explained variance: {pca.explained_variance_ratio_.sum():.2%}")
    
    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    embeddings = tsne.fit_transform(pca_result)
    
    # Visualization with class distinction
    plt.figure(figsize=(20, 15))
    cmap = plt.cm.get_cmap('tab10', num_classes)
    
    for idx, class_id in enumerate(selected_classes):
        # Real samples (circles)
        mask = (combined_labels == class_id) & (dataset_labels == 0)
        plt.scatter(embeddings[mask, 0], embeddings[mask, 1],
                   color=cmap(idx), marker='o', s=40,
                   label=f'Class {class_id} (Real)' if idx == 0 else "")
        
        # Generated samples (crosses)
        mask = (combined_labels == class_id) & (dataset_labels == 1)
        plt.scatter(embeddings[mask, 0], embeddings[mask, 1],
                   color=cmap(idx), marker='x', s=40,
                   label=f'Class {class_id} (Gen)' if idx == 0 else "")
    
    plt.title(f"t-SNE Comparison of {num_classes} Random Classes (Real=○ vs Generated=×)")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.show()
    
    # Quantitative analysis
    print("\n=== Distribution Statistics ===")
    print(f"{'Metric':<20} | {'Real Data':<15} | {'Generated Data':<15}")
    print(f"{'-'*20} | {'-'*15} | {'-'*15}")
    print(f"{'Global Mean':<20} | {np.mean(real_sub):<15.4f} | {np.mean(gen_sub):<15.4f}")
    print(f"{'Global Std':<20} | {np.std(real_sub):<15.4f} | {np.std(gen_sub):<15.4f}")
    
    # Class-wise statistics
    print("\n=== Class-wise Statistics ===")
    print(f"{'Class':<6} | {'Real Mean':<10} | {'Gen Mean':<10} | {'Real Std':<10} | {'Gen Std':<10}")
    for class_id in selected_classes:
        real_class = real_sub[combined_labels[:len(real_sub)] == class_id]
        gen_class = gen_sub[combined_labels[len(real_sub):] == class_id]
        print(f"{class_id:<6} | {np.mean(real_class):<10.4f} | {np.mean(gen_class):<10.4f} | "
              f"{np.std(real_class):<10.4f} | {np.std(gen_class):<10.4f}")
    
    # Nearest neighbor analysis
    print("\n=== Nearest Neighbor Distances ===")
    closest_indices, closest_distances = pairwise_distances_argmin_min(gen_sub, real_sub)
    print(f"Average distance to nearest real sample: {np.mean(closest_distances):.4f}")
    print(f"Median distance: {np.median(closest_distances):.4f}")
    print(f"Min distance: {np.min(closest_distances):.4f}")
    print(f"Max distance: {np.max(closest_distances):.4f}")
    
    print(f"\nTotal analysis time: {time.time()-start_time:.2f} seconds")

if __name__ == "__main__":
    REAL_DATA_PATH = "/data/yyang409/bowen/imagenet_feature/swin_base/patch4_window7_224/image_features_w_label_train.npz"
    GENERATED_DATA_PATH = "/scratch/bowenxi/dit/data_gen/03_30/class_h5/full_dataset.h5"
    
    analyze_generated_data(
        REAL_DATA_PATH,
        GENERATED_DATA_PATH,
        num_classes=10,       # Number of classes to visualize
        samples_per_class=200 # Samples per class for analysis
    )