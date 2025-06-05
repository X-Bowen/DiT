import numpy as np
import torch
from tqdm import tqdm
from models import DiT_models
from diffusion import create_diffusion

# Config
SELECTED_CLASSES = [7, 12, 24, 35, 47, 68, 73, 89, 91, 99]
SAMPLES_PER_CLASS = 100
LATENT_SHAPE = (1, 32, 32)
CHECKPOINT_PATH = "/scratch/bowenxi/dit_result/031-DiT-XL-2/checkpoints/0250000.pt"
SAVE_PATH = "generated_latents.npz"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def generate_and_save():
    # Initialize model
    model = DiT_models["DiT-XL/2"](input_size=32, num_classes=1000).to(DEVICE)
    
    # Load checkpoint safely
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
    model.load_state_dict(checkpoint["ema"])
    model.eval()
    
    diffusion = create_diffusion(timestep_respacing="ddim100")
    
    # Generate latents with progress bars
    all_generated, all_labels = [], []
    
    # Wrap class iteration with tqdm
    for class_idx in tqdm(SELECTED_CLASSES, desc="Generating class samples"):
        z = torch.randn(SAMPLES_PER_CLASS, *LATENT_SHAPE, device=DEVICE)
        y = torch.full((SAMPLES_PER_CLASS,), class_idx, device=DEVICE)
        
        with torch.no_grad():
            # Add progress bar for diffusion process
            def update_progress(p_sample_loop):
                return tqdm(p_sample_loop, desc="Diffusion steps", leave=False)
            
            latents = diffusion.p_sample_loop(
                model, 
                z.shape, 
                model_kwargs={"y": y},
                progress=update_progress
            )
            
        all_generated.append(latents.cpu().numpy().reshape(SAMPLES_PER_CLASS, -1))
        all_labels.extend([class_idx] * SAMPLES_PER_CLASS)
    
    # Save to file
    np.savez_compressed(
        SAVE_PATH,
        generated=np.concatenate(all_generated),
        labels=np.array(all_labels)
    )
    print(f"\nSaved generated latents to {SAVE_PATH}")

if __name__ == "__main__":
    generate_and_save()