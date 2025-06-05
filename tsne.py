import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from models import DiT_models
from diffusion import create_diffusion

def load_checkpoint(checkpoint_path, model, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["ema"])  # Load EMA model
    model.eval()
    return model

def generate_latents(model, diffusion, num_samples, device):
    """Generate latent vectors using the DiT model"""
    z = torch.randn(num_samples, 1, 32, 32).to(device)  # Adjust latent size as needed
    y = torch.randint(0, 1000, (num_samples,)).to(device)  # Random ImageNet labels
    with torch.no_grad():
        return diffusion.p_sample_loop(
            model, 
            z.shape, 
            model_kwargs={"y": y}
        ).cpu().numpy()
    
    # y = None  # Change if using class-conditional model
    
    # Generate samples through the full diffusion process
    # with torch.no_grad():
        # latents = diffusion.p_sample_loop(model, z.shape, model_kwargs={"y": y})
    
    # return latents.cpu().numpy()

def load_real_data(data_path, num_samples):
    """Load real latent vectors from your .npz file"""
    data = np.load(data_path)
    real_latents = data["latents"][:num_samples]  # Adjust key as needed
    return real_latents

def visualize_tsne(real_data, generated_data):
    """Visualize both real and generated data using t-SNE"""
    combined = np.concatenate([real_data, generated_data])
    labels = ["Real"] * len(real_data) + ["Generated"] * len(generated_data)
    
    # Reshape to 2D if needed
    if combined.ndim > 2:
        combined = combined.reshape(combined.shape[0], -1)
    
    # Run t-SNE
    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    embeddings = tsne.fit_transform(combined)
    
    # Plot
    plt.figure(figsize=(10, 8))
    for label in ["Real", "Generated"]:
        mask = np.array(labels) == label
        plt.scatter(embeddings[mask, 0], embeddings[mask, 1], label=label, alpha=0.6)
    
    plt.title("t-SNE Visualization of Real vs Generated Latent Vectors")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    # Config
    device = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint_path = "/scratch/bowenxi/dit_result/031-DiT-XL-2/checkpoints/0250000.pt"  # Update path
    data_path = "/data/yyang409/bowen/imagenet_feature/swin_base/patch4_window7_224/image_features_w_label_train.npz"  # Your real data
    num_samples = 1000  # Number of samples to visualize
    
    # Initialize components
    model = DiT_models["DiT-XL/2"](input_size=32, num_classes=1000).to(device)  # Adjust input_size
    diffusion = create_diffusion(timestep_respacing="")  # Match training config
    
    # Load checkpoint
    model = load_checkpoint(checkpoint_path, model, device)
    
    # Generate samples
    generated_latents = generate_latents(model, diffusion, num_samples, device)
    
    # Load real data
    real_latents = load_real_data(data_path, num_samples)
    
    # Visualize
    visualize_tsne(real_latents, generated_latents)