import time
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from tqdm import tqdm
from models import DiT_models
from diffusion import create_diffusion


# Example: 10-class subset (ImageNet indices)
target_classes = [
    207,  # golden retriever
    281,  # tabby cat
    404,  # sports car
    511,  # espresso
    620,  # mountain bike
    881,  # daisy
    947,  # pizza
    999,  # mosquito
    365,  # strawberry
    22    # bald eagle
]


def load_checkpoint(checkpoint_path, model, device):
    start = time.time()
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["ema"])
    model.eval()
    print(f"Loaded checkpoint in {time.time()-start:.2f}s")
    return model

def generate_latents(model, diffusion, num_samples, device):
    """Generate latent vectors with progress tracking"""
    # Adjust latent dimensions based on your actual model
    z = torch.randn(num_samples, 1, 32, 32).to(device)  # Fixed channel dimension
    y = torch.randint(0, 1000, (num_samples,)).to(device)
    
    # Create a wrapper for the diffusion process with progress bar
    def progress_callback(p_sample_loop):
        return tqdm(p_sample_loop, desc="Generating samples", unit="step")

    with torch.no_grad():
        start = time.time()
        latents = diffusion.p_sample_loop(
            model, 
            z.shape, 
            model_kwargs={"y": y},
            progress=progress_callback  # Requires modified diffusion.py
        ).cpu().numpy()
        print(f"Generated {num_samples} samples in {time.time()-start:.2f}s")
    
    return latents

def load_real_data(data_path, num_samples):
    start = time.time()
    data = np.load(data_path)
    real_latents = data["latents"][:num_samples]
    print(f"Loaded real data in {time.time()-start:.2f}s")
    return real_latents

def visualize_tsne(real_data, generated_data):
    start = time.time()
    combined = np.concatenate([real_data, generated_data])
    
    if combined.ndim > 2:
        combined = combined.reshape(combined.shape[0], -1)
    
    # Reduce sample size for faster t-SNE
    if len(combined) > 2000:
        print("Subsampling to 2000 points for faster t-SNE...")
        rng = np.random.default_rng(42)
        indices = rng.choice(len(combined), 2000, replace=False)
        combined = combined[indices]
        labels = ["Real"]*1000 + ["Generated"]*1000
        labels = [labels[i] for i in indices]
    else:
        labels = ["Real"]*len(real_data) + ["Generated"]*len(generated_data)
    
    # Use faster Barnes-Hut approximation
    tsne = TSNE(n_components=2, perplexity=30, random_state=42, 
                method='barnes_hut', angle=0.5, n_jobs=-1)
    
    print("Running t-SNE...")
    embeddings = tsne.fit_transform(combined)
    
    plt.figure(figsize=(10, 8))
    for label in ["Real", "Generated"]:
        mask = np.array(labels) == label
        plt.scatter(embeddings[mask, 0], embeddings[mask, 1], label=label, alpha=0.6)
    
    plt.title("t-SNE Visualization")
    plt.legend()
    plt.show()
    print(f"Visualization completed in {time.time()-start:.2f}s")

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Configuration
    config = {
        "checkpoint_path": "/scratch/bowenxi/dit_result/031-DiT-XL-2/checkpoints/0250000.pt",
        "data_path": "/data/yyang409/bowen/imagenet_feature/swin_base/patch4_window7_224/image_features_w_label_train.npz",
        "num_samples": 1000,  # Start with 500 for testing
        "input_size": 32,
        "num_classes": 1000
    }
    
    # Initialize components
    start_init = time.time()
    model = DiT_models["DiT-XL/2"](input_size=config["input_size"], 
                                  num_classes=config["num_classes"]).to(device)
    diffusion = create_diffusion(timestep_respacing="ddim100")  # 10x faster sampling
    print(f"Initialized model in {time.time()-start_init:.2f}s")
    
    # Load checkpoint
    model = load_checkpoint(config["checkpoint_path"], model, device)
    
    # Generate samples
    generated_latents = generate_latents(model, diffusion, config["num_samples"], device)
    
    # Load real data
    real_latents = load_real_data(config["data_path"], config["num_samples"])
    
    # Visualize
    visualize_tsne(real_latents, generated_latents)