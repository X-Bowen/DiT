import os
import h5py
import numpy as np
import torch
from tqdm import tqdm
from ../models import DiT_models
from diffusion import create_diffusion
from multiprocessing import Process, cpu_count

# Configuration
NUM_CLASSES = 1000
SAMPLES_PER_CLASS = 10 * 1024
BATCH_SIZE = 1024
LATENT_SHAPE = (1, 32, 32)
CHECKPOINT_PATH = "/scratch/bowenxi/dit_result/DiT-B_4_0327_4_a100/000-DiT-B-4/checkpoints/3000000.pt"
OUTPUT_DIR = "/scratch/bowenxi/dit/data_gen/0404_all_data"
DEVICES = ["cuda:0", "cuda:1", "cuda:2", "cuda:3","cuda:4", "cuda:5","cuda:6", "cuda:7"]  # Adjust based on torch.cuda.device_count()

os.makedirs(OUTPUT_DIR, exist_ok=True)

def generate_classes_for_gpu(gpu_id, class_indices):
    """Generate samples for multiple classes on a specific GPU"""
    device = f"cuda:{gpu_id}"
    
    # Load model once per GPU
    model = DiT_models["DiT-B/4"](input_size=32, num_classes=NUM_CLASSES).to(device)
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
        for _ in range(0, SAMPLES_PER_CLASS, BATCH_SIZE):
            batch_size = min(BATCH_SIZE, SAMPLES_PER_CLASS - len(latents))
            z = torch.randn(batch_size, *LATENT_SHAPE, device=device)
            y = torch.full((batch_size,), class_idx, device=device)
            
            with torch.no_grad(), torch.cuda.amp.autocast():
                batch_latents = diffusion.p_sample_loop(
                    model, z.shape, model_kwargs={"y": y}
                )
            latents.append(batch_latents.cpu().numpy().reshape(batch_size, -1))
        
        # Save to HDF5
        with h5py.File(h5_path, 'w') as f:
            f.create_dataset("samples", data=np.concatenate(latents), dtype=np.float32)
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
            with h5py.File(class_path, 'r') as cf:
                start = class_idx * SAMPLES_PER_CLASS
                end = (class_idx + 1) * SAMPLES_PER_CLASS
                samples[start:end] = cf['samples'][:]
                labels[start:end] = cf['labels'][:]

if __name__ == "__main__":
    num_gpus = torch.cuda.device_count()
    assert len(DEVICES) <= num_gpus, "More GPUs specified than available!"
    
    # Split classes across GPUs
    classes_per_gpu = [[] for _ in range(len(DEVICES))]
    for idx in range(NUM_CLASSES):
        gpu = idx % len(DEVICES)
        classes_per_gpu[gpu].append(idx)
    
    # Start a process for each GPU
    processes = []
    for gpu_id in range(len(DEVICES)):
        p = Process(
            target=generate_classes_for_gpu,
            args=(gpu_id, classes_per_gpu[gpu_id])
        )
        p.start()
        processes.append(p)
    
    for p in processes:
        p.join()
    
    merge_h5_files()