import os
import numpy as np
import torch
from tqdm import tqdm
import multiprocessing as mp
from models import DiT_models
from diffusion import create_diffusion
import sys

nu = sys.argv[1]
# Configuration
NUM_CLASSES = 1000
SAMPLES_PER_CLASS = 1024
BATCH_SIZE = 1024  # Adjust based on GPU memory
LATENT_SHAPE = (1, 32, 32)  # Match your latent dimensions
CHECKPOINT_PATH = "/scratch/bowenxi/dit_result/DiT-L_8_0401_4_a100/000-DiT-L-8/checkpoints/3500000.pt"
OUTPUT_DIR = f"/scratch/bowenxi/dit/data_gen/L_8/{nu}"
#DEVICES = ["cuda:0","cuda:1","cuda:2","cuda:3","cuda:4","cuda:5","cuda:6","cuda:7"]  # Available GPUs
DEVICES = ["cuda:0","cuda:1","cuda:2","cuda:3"]
#DEVICES = ["cuda:2"]


def ensure_dir(directory):
    """Make sure the output directory exists"""
    if not os.path.exists(directory):
        os.makedirs(directory)

def generate_class_range(device, class_start, class_end):
    """Generate samples for a range of classes on a single GPU and save to NPZ files"""
    # Device-specific output directory
    device_id = device.split(':')[1]
    device_dir = os.path.join(OUTPUT_DIR, f"gpu{device_id}")
    ensure_dir(device_dir)
    
    # Initialize model
    model = DiT_models["DiT-L/8"](input_size=32, num_classes=NUM_CLASSES).to(device)
    # Set weights_only=True for security
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["ema"])
    model.eval()
    
    diffusion = create_diffusion(timestep_respacing="ddim100")
    
    # Process classes
    with tqdm(total=class_end - class_start, desc=f"Generating on {device}") as pbar:
        for class_idx in range(class_start, class_end):
            # Check if this class is already processed
            class_file = os.path.join(device_dir, f"class_{class_idx}.npz")
            if os.path.exists(class_file):
                pbar.update(1)
                continue
            
            # Generate in batches
            class_samples = []
            remaining = SAMPLES_PER_CLASS
            
            while remaining > 0:
                current_batch_size = min(BATCH_SIZE, remaining)
                z = torch.randn(current_batch_size, *LATENT_SHAPE, device=device)
                y = torch.full((current_batch_size,), class_idx, device=device)
                
                with torch.no_grad():
                    latents = diffusion.p_sample_loop(
                        model, z.shape, model_kwargs={"y": y}
                    )
                class_samples.append(latents.cpu().numpy())
                remaining -= current_batch_size
            
            # Concatenate batches and reshape
            class_samples = np.concatenate(class_samples)
            flat_samples = class_samples.reshape(SAMPLES_PER_CLASS, -1)
            
            # Save as NPZ file (one file per class)
            np.savez_compressed(
                class_file,
                samples=flat_samples,
                labels=np.full(SAMPLES_PER_CLASS, class_idx)
            )
            
            pbar.update(1)
            torch.cuda.empty_cache()

def run_on_gpu(device, start, end):
    """Wrapper function for multiprocessing"""
    torch.cuda.set_device(int(device.split(':')[1]))
    generate_class_range(device, start, end)

def merge_npz_files():
    """Optionally merge all NPZ files into a single file"""
    print("Merging NPZ files...")
    
    # Dictionary to store all samples and labels
    all_samples = []
    all_labels = []
    
    # Process each GPU's directory
    for device_id in range(len(DEVICES)):
        device_dir = os.path.join(OUTPUT_DIR, f"gpu{device_id}")
        if not os.path.exists(device_dir):
            print(f"Warning: {device_dir} does not exist. Skipping.")
            continue
        
        # Process each class file
        for filename in tqdm(os.listdir(device_dir), desc=f"Processing GPU {device_id}"):
            if not filename.endswith('.npz'):
                continue
                
            file_path = os.path.join(device_dir, filename)
            data = np.load(file_path)
            
            # Append data
            all_samples.append(data['samples'])
            all_labels.append(data['labels'])
    
    # Concatenate all data
    if all_samples:
        all_samples = np.concatenate(all_samples)
        all_labels = np.concatenate(all_labels)
        
        # Save merged data
        merged_file = os.path.join(OUTPUT_DIR, "imagenet_latents_merged.npz")
        np.savez_compressed(
            merged_file,
            samples=all_samples,
            labels=all_labels
        )
        print(f"Merged data saved to {merged_file}")
    else:
        print("No data to merge.")

if __name__ == "__main__":

    # Create output directory
    ensure_dir(OUTPUT_DIR)
    
    # Split classes across GPUs
    classes_per_gpu = NUM_CLASSES // len(DEVICES)
    processes = []
    
    # Launch processes for each GPU
    for gpu_id, device in enumerate(DEVICES):
        start = gpu_id * classes_per_gpu
        end = (gpu_id + 1) * classes_per_gpu
        if gpu_id == len(DEVICES) - 1:  # Handle remainder
            end = NUM_CLASSES
        
        # Create and start process
        p = mp.Process(target=run_on_gpu, args=(device, start, end))
        p.start()
        processes.append(p)
    
    # Wait for all processes to complete
    for p in processes:
        p.join()
    
    # Optionally merge files - comment this out if you prefer 
    # to keep separate files per class
    merge_npz_files()
    
    print("All done!")