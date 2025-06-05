import torch
import time
import numpy as np
import os

def compute_product_with_gpu(F, A=None, batch_size=10000, device_ids=None):
    """
    Computes F × A using GPU acceleration with batching, where A can be provided or computed as F^T × F if not provided.
    
    Args:
        F: Input feature matrix of shape (m, n) as numpy array or torch tensor
        A: Precomputed matrix A (optional). If None, computes A = F^T × F.
        batch_size: Size of batches to process at once
        device_ids: List of GPU device IDs to use (None uses all available)
        
    Returns:
        result: The matrix product F × A as a numpy array or torch tensor
        A_computed: The matrix A used in the computation (if computed, else None)
    """
    start_time = total_time = time.time()
    
    # Determine available GPUs
    if device_ids is None:
        device_ids = list(range(torch.cuda.device_count()))
    
    num_gpus = len(device_ids)
    print(f"Using {num_gpus} GPU(s): {device_ids}")
    
    # Convert to torch tensor if needed
    input_is_numpy = isinstance(F, np.ndarray)
    if input_is_numpy:
        print("Converting input to PyTorch tensor...")
        F_tensor = torch.from_numpy(F).float()
    else:
        F_tensor = F.float()
    
    m, n = F_tensor.shape
    print(f"Input matrix shape: {m} × {n}")
    
    primary_device = f"cuda:{device_ids[0]}" if device_ids else "cuda:0"
    
    # Compute A if not provided
    A_computed = None
    if A is None:
        print("Computing A = F^T × F...")
        F_t = F_tensor.t().to(primary_device)
        A_tensor = torch.matmul(F_t, F_tensor.to(primary_device))
        A_computed = A_tensor
    else:
        print("Using provided A matrix...")
        # Convert A to tensor and move to primary device
        if isinstance(A, np.ndarray):
            A_tensor = torch.from_numpy(A).float().to(primary_device)
        else:
            A_tensor = A.float().to(primary_device)
    
    print(f"A matrix shape: {A_tensor.shape}")
    print(f"Time to compute/load A: {time.time() - start_time:.2f} seconds")
    
    # Create output tensor on CPU initially
    result_tensor = torch.zeros((m, n), dtype=F_tensor.dtype)
    
    # Distribute batch computation across available GPUs
    num_batches = int(np.ceil(m / batch_size))
    print(f"Computing F × A in {num_batches} batches across {num_gpus} GPU(s)...")
    
    # Keep A on all devices
    device_As = {device_id: A_tensor.to(f"cuda:{device_id}") for device_id in device_ids}
    
    for i in range(num_batches):
        batch_start = i * batch_size
        batch_end = min((i + 1) * batch_size, m)
        
        # Select GPU using round-robin
        gpu_id = device_ids[i % num_gpus]
        device = f"cuda:{gpu_id}"
        
        # Process batch on selected GPU
        batch = F_tensor[batch_start:batch_end].to(device)
        batch_result = torch.matmul(batch, device_As[gpu_id])
        
        # Move result to CPU
        result_tensor[batch_start:batch_end] = batch_result.cpu()
        
        # Cleanup
        del batch, batch_result
        torch.cuda.empty_cache()
        
        # Progress report
        if (i + 1) % 5 == 0 or (i + 1) == num_batches:
            batch_time = time.time() - total_time
            total_time = time.time()
            progress = (i + 1) / num_batches * 100
            print(f"Progress: {progress:.1f}% - Batch {i+1}/{num_batches} - "
                  f"Time: {batch_time:.2f}s - GPU: {gpu_id} - "
                  f"Total elapsed: {time.time() - start_time:.2f}s")
    
    print(f"Total computation time: {time.time() - start_time:.2f} seconds")
    
    # Convert result back to numpy if input was numpy
    if input_is_numpy:
        result = result_tensor.numpy()
        if A_computed is not None:
            A_computed = A_computed.cpu().numpy()
    else:
        result = result_tensor
    
    return result, A_computed if A is None else None

def process_npz_data_with_gpu(npz_file_path, output_path=None, a_matrix_save_path=None, a_matrix_load_path=None, batch_size=10000, device_ids=None):
    """
    Process the feature matrix from a NPZ file using GPU acceleration
    
    Args:
        npz_file_path: Path to the NPZ file
        output_path: Path to save the result
        a_matrix_save_path: Path to save the computed A matrix (training data)
        a_matrix_load_path: Path to load the A matrix (test data)
        batch_size: Batch size for processing
        device_ids: List of GPU device IDs
    """
    print(f"Loading data from {npz_file_path}...")
    npz_data = np.load(npz_file_path)
    
    feature_key = 'features' if 'features' in npz_data else 'feature'
    label_key = 'labels' if 'labels' in npz_data else 'label'
    
    features = npz_data[feature_key]
    labels = npz_data[label_key]
    original_features = features.copy()
    
    print(f"Features shape: {features.shape}, Labels shape: {labels.shape}")
    
    # Load A if provided
    A = None
    if a_matrix_load_path:
        print(f"Loading A from {a_matrix_load_path}...")
        A = np.load(a_matrix_load_path)
    
    # Compute product and get A if computed
    result, A_computed = compute_product_with_gpu(features, A=A, batch_size=batch_size, device_ids=device_ids)
    
    # Save A if path provided and computed
    if a_matrix_save_path and A_computed is not None:
        print(f"Saving A to {a_matrix_save_path}...")
        np.save(a_matrix_save_path, A_computed)
    
    # Save results
    if output_path:
        print(f"Saving results to {output_path}...")
        np.savez(output_path, **{feature_key: result, label_key: labels})
    
    return result, labels, original_features

# Verification and main script remain mostly the same
def verify_data_structure(original_features, transformed_features):
    if original_features.shape[0] != transformed_features.shape[0]:
        print("ERROR: Row count mismatch!")
        return False
    print("Data structure verified.")
    return True

if __name__ == "__main__":
    # Configuration for Training Data
    train_npz = "/data/yyang409/bowen/imagenet_feature/swin_base/patch4_window7_224/image_features_w_label_train.npz"
    train_output = "/scratch/bowenxi/dit/neural_tangent_kernel/feature_swin_b/train_tanent.npz"
    a_matrix_path = "/scratch/bowenxi/dit/neural_tangent_kernel/feature_swin_b/A_matrix.npy"
    
    # Process Training Data (Compute and Save A)
    print("\nProcessing TRAINING data...")
    train_result, train_labels, train_original = process_npz_data_with_gpu(
        npz_file_path=train_npz,
        output_path=train_output,
        a_matrix_save_path=a_matrix_path,
        batch_size=20000,
        device_ids=list(range(torch.cuda.device_count()))
    )
    verify_data_structure(train_original, train_result)
    
    # Configuration for Test Data
    test_npz = "/data/yyang409/bowen/imagenet_feature/swin_base/patch4_window7_224/image_features_w_label_val.npz"
    test_output = "/scratch/bowenxi/dit/neural_tangent_kernel/feature_swin_b/val_tanent.npz"
    
    # Process Test Data (Load A)
    print("\nProcessing TEST data...")
    test_result, test_labels, test_original = process_npz_data_with_gpu(
        npz_file_path=test_npz,
        output_path=test_output,
        a_matrix_load_path=a_matrix_path,
        batch_size=20000,
        device_ids=list(range(torch.cuda.device_count()))
    )
    verify_data_structure(test_original, test_result)
    
    print("\nProcess completed successfully!")