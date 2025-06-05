 import torch
import time
import numpy as np
import os

def compute_product_with_gpu(F, batch_size=10000, device_ids=None):
    """
    Computes F × F^T × F using GPU acceleration with batching
    
    Args:
        F: Input feature matrix of shape (m, n) as numpy array or torch tensor
        batch_size: Size of batches to process at once
        device_ids: List of GPU device IDs to use (None uses all available)
        
    Returns:
        result: The matrix product F × F^T × F as a numpy array
    """
    start_time = total_time = time.time()
    
    # Determine available GPUs
    if device_ids is None:
        device_ids = list(range(torch.cuda.device_count()))
    
    num_gpus = len(device_ids)
    print(f"Using {num_gpus} GPU(s): {device_ids}")
    
    # Convert to torch tensor if needed
    if isinstance(F, np.ndarray):
        print("Converting input to PyTorch tensor...")
        F = torch.from_numpy(F).float()  # Convert to float32 for better GPU performance
    
    m, n = F.shape
    print(f"Input matrix shape: {m} × {n}")
    
    # First compute A = F^T × F (n × n matrix)
    print("Computing F^T × F...")
    
    # Move to primary GPU for this computation
    primary_device = f"cuda:{device_ids[0]}"
    
    # Compute F^T × F on GPU
    # We'll use the primary GPU for this relatively small computation
    F_t = F.t().to(primary_device)
    A = torch.matmul(F_t, F.to(primary_device))
    
    print(f"A matrix shape: {A.shape}")
    print(f"Time to compute A: {time.time() - start_time:.2f} seconds")
    
    # Create output tensor on CPU initially
    result = torch.zeros((m, n), dtype=F.dtype)
    
    # Distribute batch computation across available GPUs
    num_batches = int(np.ceil(m / batch_size))
    print(f"Computing F × A in {num_batches} batches across {num_gpus} GPU(s)...")
    
    # Keep A on all devices
    device_As = {device_id: A.to(f"cuda:{device_id}") for device_id in device_ids}
    
    for i in range(num_batches):
        batch_start = i * batch_size
        batch_end = min((i + 1) * batch_size, m)
        
        # Select which GPU to use for this batch (simple round-robin)
        gpu_id = device_ids[i % num_gpus]
        device = f"cuda:{gpu_id}"
        
        # Process one batch on the selected GPU
        batch = F[batch_start:batch_end].to(device)
        batch_result = torch.matmul(batch, device_As[gpu_id])
        
        # Move result back to CPU
        result[batch_start:batch_end] = batch_result.cpu()
        
        # Clear GPU cache for this batch
        del batch, batch_result
        torch.cuda.empty_cache()
        
        # Report progress
        if (i + 1) % 5 == 0 or (i + 1) == num_batches:
            batch_time = time.time() - total_time
            total_time = time.time()
            progress = (i + 1) / num_batches * 100
            print(f"Progress: {progress:.1f}% - Batch {i+1}/{num_batches} - " 
                  f"Time: {batch_time:.2f}s - GPU: {gpu_id} - "
                  f"Total elapsed: {time.time() - start_time:.2f}s")
    
    print(f"Total computation time: {time.time() - start_time:.2f} seconds")
    
    # Convert result back to numpy if input was numpy
    if isinstance(F, np.ndarray):
        result = result.numpy()
        
    return result

def process_npz_data_with_gpu(npz_file_path, output_path=None, batch_size=10000, device_ids=None):
    """
    Process the feature matrix from a NPZ file using GPU acceleration
    
    Args:
        npz_file_path: Path to the NPZ file containing 'feature' and 'label' keys
        output_path: Path to save the result (optional)
        batch_size: Size of batches for processing
        device_ids: List of GPU device IDs to use
        
    Returns:
        result: The computed matrix product
        labels: The labels from the input file
    """
    print(f"Loading data from NPZ file: {npz_file_path}...")
    
    # Load the NPZ file which contains 'feature' and 'label' keys
    npz_data = np.load(npz_file_path)
    features = npz_data['features']
    labels = npz_data['labels']
    
    print(f"Features matrix loaded with shape: {features.shape}")
    print(f"Labels array shape: {labels.shape}")
    
    # Compute the product
    result = compute_product_with_gpu(features, batch_size=batch_size, device_ids=device_ids)
    
    # Save the result if output path is provided
    if output_path:
        print(f"Saving result to {output_path}...")
        # Save both the computed result and the original labels
        np.savez(output_path, features=result, labels=labels)
        print(f"Results saved to {output_path}")
    
    return result, labels


# Function to verify that the computation preserves the structure of the data
def verify_data_structure(original_features, transformed_features):
    """
    Verify that the computation preserves the structure of the data
    by checking that the row counts match
    """
    print("Verifying data structure...")
    
    # Check that the number of data points (rows) is the same
    if original_features.shape[0] != transformed_features.shape[0]:
        print("ERROR: Number of data points doesn't match!")
        return False
    
    # Check that we're just transforming the feature space (columns)
    if original_features.shape[1] != transformed_features.shape[1]:
        print("WARNING: Feature dimensions have changed. This is expected if intended.")
    
    print("Data structure verification passed: Row counts match.")
    return True

if __name__ == "__main__":
    # Configuration
    npz_file_path = "/data/yyang409/bowen/imagenet_feature/swin_base/patch4_window7_224/image_features_w_label_val.npz"  # Update this to your file path
    #output_path = "/scratch/bowenxi/dit/neural_tangent_kernel/result.npz"    # Update this to your desired output path
    output_path = "/scratch/bowenxi/dit/neural_tangent_kernel/feature_swin_b/val_tanent.npz"
    # Use all available H100 GPUs (adjust based on your system)
    gpu_ids = list(range(torch.cuda.device_count()))
    print(f"Found {len(gpu_ids)} GPUs")
    
    # Adjust batch size based on your GPU memory 
    # H100 has ~80GB memory, so we can use larger batches
    batch_size = 20000  # Can be increased based on your specific H100 configuration
    
    # Print GPU info
    for gpu_id in gpu_ids:
        free_mem, total_mem = torch.cuda.mem_get_info(gpu_id)
        print(f"GPU {gpu_id} memory: {free_mem/1e9:.2f}GB free / {total_mem/1e9:.2f}GB total")
    
    # Load original data for verification
    print(f"Loading original data from {npz_file_path}...")
    original_data = np.load(npz_file_path)
    original_features = original_data['feature']

    # Process the data
    result, labels = process_npz_data_with_gpu(
        npz_file_path=npz_file_path,
        output_path=output_path,
        batch_size=batch_size,
        device_ids=gpu_ids
    )
    
    # Verify the structure is preserved
    verify_data_structure(original_features, transformed_features)
    
    print("Computation complete!")
    print(f"Original features shape: {original_features.shape}")
    print(f"Transformed features shape: {transformed_features.shape}")
 
    
    # Monitor GPU memory after computation
    for gpu_id in gpu_ids:
        free_mem, total_mem = torch.cuda.mem_get_info(gpu_id)
        print(f"GPU {gpu_id} memory after: {free_mem/1e9:.2f}GB free / {total_mem/1e9:.2f}GB total")

# Optional: Function to check if computation is correct (for debugging with smaller matrices)
def verify_computation(features, computed_result):
    """Verify the computation by comparing with direct calculation"""
    print("Verifying computation...")
    # Direct computation (only use for small matrices!)
    expected = features @ features.T @ features
    
    # Compare
    if isinstance(computed_result, torch.Tensor):
        computed_result = computed_result.numpy()
    if isinstance(expected, torch.Tensor):
        expected = expected.numpy()
        
    diff = np.max(np.abs(computed_result - expected))
    print(f"Maximum absolute difference: {diff}")
    return diff < 1e-5