import os
import h5py
import numpy as np
import torch
from tqdm import tqdm
from models import DiT_models
from diffusion import create_diffusion

# Configuration
NUM_CLASSES = 1000
SAMPLES_PER_CLASS = 1024
BATCH_SIZE = 128  # Adjust based on GPU memory
LATENT_SHAPE = (1, 32, 32)  # Match your latent dimensions
# CHECKPOINT_PATH = "/scratch/bowenxi/dit_result/031-DiT-XL-2/checkpoints/0250000.pt"
CHECKPOINT_PATH = "/scratch/bowenxi/dit_result/DiT-L_8_0401_4_a100/000-DiT-L-8/checkpoints/1000000.pt"
HDF5_PATH = "/scratch/bowenxi/dit/data_gen/0330_2/0404_imagenet_latents.h5"
#DEVICES = ["cuda:0","cuda:1","cuda:2","cuda:3","cuda:4","cuda:5","cuda:6","cuda:7"]  # Available GPUs
DEVICES = ["cuda:0","cuda:1","cuda:2"]
#DEVICES = ["cuda:2"]

def initialize_hdf5():
    """Create an extendable HDF5 dataset"""
    with h5py.File(HDF5_PATH, 'w') as f:
        # Initialize resizable datasets
        f.create_dataset(
            "samples",
            shape=(0, 1024),  # Flattened latent size
            maxshape=(None, 1024),
            dtype=np.float32,
            chunks=(BATCH_SIZE, 1024)  # Optimize for chunked writes
        )
            
        f.create_dataset(
            "labels",
            shape=(0,),
            maxshape=(None,),
            dtype=np.int32,
            chunks=(BATCH_SIZE,)
        )
            
def generate_class_range(device, class_start, class_end):
    """Generate samples for a range of classes on a single GPU"""
    # Initialize model
    model = DiT_models["DiT-L/8"](input_size=32, num_classes=NUM_CLASSES).to(device)
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
    model.load_state_dict(checkpoint["ema"])
    model.eval()
    
    diffusion = create_diffusion(timestep_respacing="ddim100")
    
    # Open HDF5 in append mode
    with h5py.File(HDF5_PATH, 'a') as f, tqdm(
        total=class_end - class_start,
        desc=f"Generating on {device}"
    ) as pbar:
        samples_dset = f["samples"]
        labels_dset = f["labels"]
        
        for class_idx in range(class_start, class_end):
            # Check if class already exists
            if (labels_dset[:] == class_idx).any():
                pbar.update(1)
                continue
                
            # Generate in batches
            class_samples = []
            for _ in range(0, SAMPLES_PER_CLASS, BATCH_SIZE):
                z = torch.randn(BATCH_SIZE, *LATENT_SHAPE, device=device)
                y = torch.full((BATCH_SIZE,), class_idx, device=device)
                
                with torch.no_grad():
                    latents = diffusion.p_sample_loop(
                        model, z.shape, model_kwargs={"y": y}
                    )
                class_samples.append(latents.cpu().numpy().reshape(-1, 1024))
                
            # Concatenate and save
            class_samples = np.concatenate(class_samples)
            
            # Resize datasets
            new_size = samples_dset.shape[0] + SAMPLES_PER_CLASS
            samples_dset.resize(new_size, axis=0)
            labels_dset.resize(new_size, axis=0)
            
            # Append data
            samples_dset[-SAMPLES_PER_CLASS:] = class_samples
            labels_dset[-SAMPLES_PER_CLASS:] = np.full(SAMPLES_PER_CLASS, class_idx)
            
            pbar.update(1)
            torch.cuda.empty_cache()

if __name__ == "__main__":
    # Initialize HDF5 file (only once!)
    if not os.path.exists(HDF5_PATH):
        initialize_hdf5()
    
    # Split classes across GPUs
    classes_per_gpu = NUM_CLASSES // len(DEVICES)
    processes = []
    
    for gpu_id, device in enumerate(DEVICES):
        start = gpu_id * classes_per_gpu
        end = (gpu_id + 1) * classes_per_gpu
        if gpu_id == len(DEVICES) - 1:  # Handle remainder
            end = NUM_CLASSES
        
        generate_class_range(device, start, end)