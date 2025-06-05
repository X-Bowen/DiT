import torch
from torch.utils.data import Dataset, DataLoader
from dit import DiT  # Replace with your actual DiT implementation
import numpy as np
import os

# Configuration
num_classes = 10
samples_per_class = 100
latent_shape = (1, 32, 32)  # Input shape for DiT
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
checkpoint_path = "/scratch/bowenxi/dit_result/031-DiT-XL-2/checkpoints/0250000.pt"
output_dir = "812generated_latents.npz"
os.makedirs(output_dir, exist_ok=True)

# Initialize model (match your architecture)
model = DiT(
    input_size=latent_shape,
    num_classes=num_classes,
    # Add other parameters matching your implementation
).to(device)

# Load checkpoint
checkpoint = torch.load(checkpoint_path, map_location=device)
model.load_state_dict(checkpoint["model"])
model.eval()

def generate_class_samples(class_label, num_samples):
    z = torch.randn(num_samples, *latent_shape, device=device)
    class_conditions = torch.full((num_samples,), class_label, device=device)
    
    # Run through DiT sampling process
    with torch.no_grad():
        generated_latents = model.sample(
            z=z,
            class_labels=class_conditions,
            # Add other sampling parameters (timesteps, guidance scale, etc.)
        )
    
    return generated_latents.cpu().numpy()

for class_idx in range(num_classes):
    print(f"Generating class {class_idx}...")
    
    # Generate latent vectors
    latents = generate_class_samples(class_idx, samples_per_class)
    
    # Reshape and save
    for i in range(samples_per_class):
        latent_vector = latents[i].reshape(-1)  # Flatten to 1024-dim vector
        np.save(
            os.path.join(output_dir, f"class_{class_idx}_sample_{i}.npy"),
            {
                "latent": latent_vector,
                "class": class_idx
            }
        )


# Verify generation
print(f"Generated {num_classes * samples_per_class} samples")
print(f"Sample shape: {latent_vector.shape} (1024-dim vector)")

# For comparison with original data:
# 1. Load original ImageNet latents
# 2. Use same metric space (e.g., LPIPS, FID, or feature-space distances)
# 3. Compare distribution statistics (mean, covariance)