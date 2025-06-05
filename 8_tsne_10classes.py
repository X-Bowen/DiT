import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from models import DiT_models
from diffusion import create_diffusion

# Configuration - Adjust These!
selected_classes = [7, 12, 24, 35, 47, 68, 73, 89, 91, 99]  # Example ImageNet class indices
samples_per_class = 100

def load_checkpoint(checkpoint_path, model, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["ema"])
    model.eval()
    return model

def generate_class_latents(model, diffusion, device):
    """Generate 100 samples per class for selected classes"""
    all_latents = []
    
    for class_idx in selected_classes:
        # Generate class-specific samples
        z = torch.randn(samples_per_class, 1, 32, 32).to(device)
        y = torch.full((samples_per_class,), class_idx).to(device)
        
        with torch.no_grad():
            class_latents = diffusion.p_sample_loop(
                model, 
                z.shape, 
                model_kwargs={"y": y}
            ).cpu().numpy()
        
        all_latents.append(class_latents)
    
    return np.concatenate(all_latents)

def load_real_data(data_path):
    """Load 100 samples per class from real data"""
    data = np.load(data_path)
    all_latents = data["features"]
    all_labels = data["labels"]  # Assuming labels exist
    
    real_data = []
    for class_idx in selected_classes:
        class_mask = (all_labels == class_idx)
        real_data.append(all_latents[class_mask][:samples_per_class])
    
    return np.concatenate(real_data)

def visualize_class_tsne(real_data, generated_data):
    """Color-coded visualization by class"""
    combined = np.concatenate([real_data, generated_data])
    labels = np.concatenate([
        np.repeat(selected_classes, samples_per_class),  # Real labels
        np.repeat(selected_classes, samples_per_class)   # Generated labels
    ])
    data_type = np.concatenate([
        np.zeros(len(real_data)),  # 0 = real
        np.ones(len(generated_data))  # 1 = generated
    ])
    
    # Flatten if needed
    if combined.ndim > 2:
        combined = combined.reshape(combined.shape[0], -1)
    
    # Run t-SNE
    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    embeddings = tsne.fit_transform(combined)
    
    # Plot
    plt.figure(figsize=(15, 10))
    for idx, class_id in enumerate(selected_classes):
        # Real samples
        mask_real = (labels == class_id) & (data_type == 0)
        plt.scatter(embeddings[mask_real, 0], embeddings[mask_real, 1],
                    color=plt.cm.tab10(idx), marker='o', alpha=0.7,
                    label=f'Class {class_id} (Real)' if idx == 0 else "")
        
        # Generated samples
        mask_gen = (labels == class_id) & (data_type == 1)
        plt.scatter(embeddings[mask_gen, 0], embeddings[mask_gen, 1],
                    color=plt.cm.tab10(idx), marker='x', alpha=0.7,
                    label=f'Class {class_id} (Gen)' if idx == 0 else "")
    
    plt.title("Class-Wise t-SNE Comparison (Real=○ vs Generated=×)")
    plt.legend(ncol=2)
    plt.show()

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Initialize model
    model = DiT_models["DiT-XL/2"](input_size=32, num_classes=1000).to(device)
    diffusion = create_diffusion(timestep_respacing="ddim100")  # Faster sampling
    
    # Load checkpoint
    model = load_checkpoint("/scratch/bowenxi/dit_result/031-DiT-XL-2/checkpoints/0250000.pt", model, device)
    
    # Generate and load data
    generated = generate_class_latents(model, diffusion, device)
    real = load_real_data("/data/yyang409/bowen/imagenet_feature/swin_base/patch4_window7_224/image_features_w_label_train.npz")
    
    # Visualize
    visualize_class_tsne(real, generated)