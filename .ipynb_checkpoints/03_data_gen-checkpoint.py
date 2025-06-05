import os
import h5py
import numpy as np
import torch
from tqdm import tqdm
from models import DiT_models
from diffusion import create_diffusion
from multiprocessing import Process, cpu_count

# Configuration
NUM_CLASSES = 1000
SAMPLES_PER_CLASS = 1024
BATCH_SIZE = 1024
LATENT_SHAPE = (1, 32, 32)
CHECKPOINT_PATH = "/scratch/bowenxi/dit_result/031-DiT-XL-2/checkpoints/0250000.pt"
OUTPUT_DIR = "/scratch/bowenxi/dit/data_gen/claude1"
DEVICES = ["cuda:0", "cuda:1", "cuda:2", "cuda:3", "cuda:4", "cuda:5", "cuda:6", "cuda:7"]  # Adjust based on availability

os.makedirs(OUTPUT_DIR, exist_ok=True)

def generate_classes_for_gpu(gpu_id, class_indices):
    """Generate samples for multiple classes on a specific GPU"""
    device = f"cuda:{gpu_id}"
    
    # Load model once per GPU
    model = DiT_models["DiT-XL/2"](input_size=32, num_classes=NUM_CLASSES).to(device)
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
    model.load_state_dict(checkpoint["ema"])
    model.eval()
    diffusion = create_diffusion(timestep_respacing="ddim100")
    
    for class_idx in tqdm(class_indices, desc=f"GPU {gpu_id}"):
        h5_path = os.path.join(OUTPUT_DIR, f"class_{class_idx:04d}.h5")
        if os.path.exists(h5_path):
            with h5py.File(h5_path, 'r') as f:
                if 'samples' in f and f['samples'].shape[0] == SAMPLES_PER_CLASS:
                    continue
        
        # Generate samples
        latents = []
        valid_count = 0
        
        # Keep generating until we have enough valid samples
        while valid_count < SAMPLES_PER_CLASS:
            batch_size = min(BATCH_SIZE, SAMPLES_PER_CLASS - valid_count)
            z = torch.randn(batch_size, *LATENT_SHAPE, device=device)
            y = torch.full((batch_size,), class_idx, device=device)
            
            try:
                with torch.no_grad(), torch.cuda.amp.autocast():
                    batch_latents = diffusion.p_sample_loop(
                        model, z.shape, z, model_kwargs={"y": y}
                    )
                
                # Explicitly move to CPU and convert to numpy to prevent CUDA OOM
                batch_numpy = batch_latents.detach().cpu().numpy()
                
                # Check for NaN values
                if not np.isnan(batch_numpy).any():
                    # Calculate correct flattened dimension
                    flattened_size = np.prod(LATENT_SHAPE)
                    reshaped_batch = batch_numpy.reshape(batch_size, flattened_size)
                    
                    latents.append(reshaped_batch)
                    valid_count += batch_size
                else:
                    print(f"NaN detected in batch for class {class_idx}, retrying...")
                
                # Force garbage collection
                torch.cuda.empty_cache()
                
            except Exception as e:
                print(f"Error in batch generation for class {class_idx}: {str(e)}")
                torch.cuda.empty_cache()
                continue
        
        # Save to HDF5 file
        with h5py.File(h5_path, 'w') as f:
            stacked_latents = np.vstack(latents)[:SAMPLES_PER_CLASS]  # Ensure exact count
            f.create_dataset("samples", data=stacked_latents, dtype=np.float32)
            f.create_dataset("labels", data=np.full(SAMPLES_PER_CLASS, class_idx), dtype=np.int32)
        
    torch.cuda.empty_cache()

def merge_h5_files():
    """Combine all class files into a single dataset"""
    final_path = os.path.join(OUTPUT_DIR, "full_dataset.h5")
    with h5py.File(final_path, 'w') as h5f:
        samples = h5f.create_dataset("samples", (NUM_CLASSES*SAMPLES_PER_CLASS, 1024), dtype=np.float32)
        labels = h5f.create_dataset("labels", (NUM_CLASSES*SAMPLES_PER_CLASS,), dtype=np.int32)
        
        for class_idx in tqdm(range(NUM_CLASSES), desc="Merging"):
            class_path = os.path.join(OUTPUT_DIR, f"class_{class_idx:04d}.h5")
            if os.path.exists(class_path):
                with h5py.File(class_path, 'r') as cf:
                    start = class_idx * SAMPLES_PER_CLASS
                    end = (class_idx + 1) * SAMPLES_PER_CLASS
                    samples[start:end] = cf['samples'][:]
                    labels[start:end] = cf['labels'][:]
            else:
                print(f"Warning: Class {class_idx} file missing!")

def verify_dataset_integrity():
    """Check for NaN values in the generated dataset"""
    total_nans = 0
    for class_idx in tqdm(range(NUM_CLASSES), desc="Verifying"):
        class_path = os.path.join(OUTPUT_DIR, f"class_{class_idx:04d}.h5")
        if os.path.exists(class_path):
            with h5py.File(class_path, 'r') as cf:
                class_samples = cf['samples'][:]
                nan_count = np.isnan(class_samples).sum()
                if nan_count > 0:
                    print(f"Class {class_idx}: {nan_count} NaN values detected")
                    total_nans += nan_count
    
    print(f"Total NaN values across dataset: {total_nans}")
    return total_nans == 0

if __name__ == "__main__":
    num_gpus = torch.cuda.device_count()
    available_gpus = min(len(DEVICES), num_gpus)
    print(f"Using {available_gpus} GPUs out of {num_gpus} available")
    
    # Split classes across available GPUs
    classes_per_gpu = [[] for _ in range(available_gpus)]
    for idx in range(NUM_CLASSES):
        gpu = idx % available_gpus
        classes_per_gpu[gpu].append(idx)
    
    # Start a process for each GPU
    processes = []
    for gpu_id in range(available_gpus):
        p = Process(
            target=generate_classes_for_gpu,
            args=(gpu_id, classes_per_gpu[gpu_id])
        )
        p.start()
        processes.append(p)
    
    for p in processes:
        p.join()
    
    # Verify integrity before merging
    if verify_dataset_integrity():
        print("Dataset integrity verified. Merging files...")
        merge_h5_files()
        print("Dataset generation complete!")
    else:
        print("Dataset contains NaN values. Please check the individual class files.")