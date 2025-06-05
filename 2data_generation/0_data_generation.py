import os
import h5py
import numpy as np
import torch
from tqdm import tqdm
from ..models import DiT_models
from diffusion import create_diffusion
from multiprocessing import Pool, Lock, cpu_count

# Configuration
NUM_CLASSES = 1000
SAMPLES_PER_CLASS = 1024
BATCH_SIZE = 1024  # Larger batches for faster generation
LATENT_SHAPE = (1, 32, 32)
CHECKPOINT_PATH = "/scratch/bowenxi/dit_result/DiT-B_4_0327_4_a100/000-DiT-B-4/checkpoints/3000000.pt"
OUTPUT_DIR = "/scratch/bowenxi/dit/data_gen/DiT-B_4"
DEVICES = ["cuda:0", "cuda:1", "cuda:2"]

# Create output directory
os.makedirs(OUTPUT_DIR, exist_ok=True)

def generate_class(class_idx):
    """Generate and save samples for a single class"""
    device = f"cuda:{class_idx % len(DEVICES)}"  # Round-robin device assignment
    h5_path = os.path.join(OUTPUT_DIR, f"class_{class_idx:04d}.h5")
    
    # Skip existing files
    if os.path.exists(h5_path):
        with h5py.File(h5_path, 'r') as f:
            if 'samples' in f and f['samples'].shape[0] == SAMPLES_PER_CLASS:
                return
    
    # Initialize model on target device
    model = DiT_models["DiT-B/4"](input_size=32, num_classes=NUM_CLASSES).to(device)
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
    model.load_state_dict(checkpoint["ema"])
    model.eval()
    
    diffusion = create_diffusion(timestep_respacing="ddim100")
    
    # Generate samples
    latents = []
    for _ in range(0, SAMPLES_PER_CLASS, BATCH_SIZE):
        batch_size = min(BATCH_SIZE, SAMPLES_PER_CLASS - len(latents))
        z = torch.randn(batch_size, *LATENT_SHAPE, device=device)
        y = torch.full((batch_size,), class_idx, device=device)
        
        with torch.no_grad(), torch.cuda.amp.autocast():
            batch_latents = diffusion.p_sample_loop(
                model, z.shape, model_kwargs={"y": y}
            )
        latents.append(batch_latents.cpu().numpy().reshape(batch_size, -1))
    
    # Save to individual HDF5
    with h5py.File(h5_path, 'w') as f:
        f.create_dataset("samples", data=np.concatenate(latents), dtype=np.float32)
        f.create_dataset("labels", data=np.full(SAMPLES_PER_CLASS, class_idx), dtype=np.int32)
    
    torch.cuda.empty_cache()

def merge_h5_files():
    """Combine all class files into single dataset (run when finished)"""
    final_path = os.path.join(OUTPUT_DIR, "full_dataset.h5")
    with h5py.File(final_path, 'w') as h5f:
        samples = h5f.create_dataset("samples", (NUM_CLASSES*SAMPLES_PER_CLASS, 1024), dtype=np.float32)
        labels = h5f.create_dataset("labels", (NUM_CLASSES*SAMPLES_PER_CLASS,), dtype=np.int32)
        
        for class_idx in tqdm(range(NUM_CLASSES), desc="Merging"):
            class_path = os.path.join(OUTPUT_DIR, f"class_{class_idx:04d}.h5")
            with h5py.File(class_path, 'r') as cf:
                start = class_idx * SAMPLES_PER_CLASS
                end = (class_idx + 1) * SAMPLES_PER_CLASS
                samples[start:end] = cf['samples'][:]
                labels[start:end] = cf['labels'][:]

if __name__ == "__main__":
    # Generate in parallel (1 process per GPU)
    with Pool(processes=len(DEVICES)) as pool:
        list(tqdm(pool.imap_unordered(generate_class, range(NUM_CLASSES)), total=NUM_CLASSES))
    
    # Merge files when done
    merge_h5_files()