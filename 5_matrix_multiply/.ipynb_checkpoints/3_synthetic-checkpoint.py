import torch
import time
import numpy as np
import os
import glob

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

def load_synthetic_data(synthetic_file_path):
    """
    Load synthetic data from a NPZ file
    
    Args:
        synthetic_file_path: Path to the synthetic data NPZ file
        
    Returns:
        synthetic_features, synthetic_labels
    """
    print(f"Loading synthetic data from {synthetic_file_path}...")
    synthetic_data = np.load(synthetic_file_path)
    
    feature_key = 'features' if 'features' in synthetic_data else 'feature'
    label_key = 'labels' if 'labels' in synthetic_data else 'label'
    
    synthetic_features = synthetic_data[feature_key]
    synthetic_labels = synthetic_data[label_key]
    
    print(f"Synthetic features shape: {synthetic_features.shape}, Labels shape: {synthetic_labels.shape}")
    
    return synthetic_features, synthetic_labels

def process_combined_data(original_features, original_labels, synthetic_features, synthetic_labels, 
                          output_path, a_matrix_save_path, batch_size=10000, device_ids=None):
    """
    Process combined original and synthetic data
    
    Args:
        original_features: Original training features
        original_labels: Original training labels
        synthetic_features: Synthetic features to be added
        synthetic_labels: Synthetic labels to be added
        output_path: Path to save the result
        a_matrix_save_path: Path to save the computed A matrix
        batch_size: Batch size for processing
        device_ids: List of GPU device IDs
        
    Returns:
        result, a_matrix
    """
    print("Concatenating original and synthetic data...")
    combined_features = np.concatenate([original_features, synthetic_features], axis=0)
    combined_labels = np.concatenate([original_labels, synthetic_labels], axis=0)
    
    print(f"Combined features shape: {combined_features.shape}, Labels shape: {combined_labels.shape}")
    
    # Compute product and get A
    result, a_matrix = compute_product_with_gpu(combined_features, batch_size=batch_size, device_ids=device_ids)
    
    # Save A
    print(f"Saving A matrix to {a_matrix_save_path}...")
    np.save(a_matrix_save_path, a_matrix)
    
    # Save combined results
    print(f"Saving combined results to {output_path}...")
    np.savez(output_path, features=result, labels=combined_labels)
    
    return result, a_matrix

def process_val_data(val_features, val_labels, a_matrix, val_output_path, batch_size=10000, device_ids=None):
    """
    Process validation data using the provided A matrix
    
    Args:
        val_features: Validation features
        val_labels: Validation labels
        a_matrix: A matrix computed from training data
        val_output_path: Path to save the validation results
        batch_size: Batch size for processing
        device_ids: List of GPU device IDs
    """
    print("\nProcessing validation data with the new A matrix...")
    val_result, _ = compute_product_with_gpu(val_features, A=a_matrix, batch_size=batch_size, device_ids=device_ids)
    
    print(f"Saving validation results to {val_output_path}...")
    np.savez(val_output_path, features=val_result, labels=val_labels)
    
    return val_result

def verify_data_structure(original_features, transformed_features):
    """Verify that the data structure is preserved after transformation"""
    if original_features.shape[0] != transformed_features.shape[0]:
        print("ERROR: Row count mismatch!")
        return False
    print("Data structure verified.")
    return True

def incremental_processing_with_synthetic(original_train_npz, synthetic_npz_files, val_npz, 
                                          output_dir, batch_size=20000, device_ids=None):
    """
    Process original training data with synthetic data incrementally
    
    Args:
        original_train_npz: Path to the original training data NPZ file
        synthetic_npz_files: List of paths to synthetic data NPZ files
        val_npz: Path to the validation data NPZ file
        output_dir: Directory to save outputs
        batch_size: Batch size for processing
        device_ids: List of GPU device IDs
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Load original training data
    print(f"Loading original training data from {original_train_npz}...")
    train_data = np.load(original_train_npz)
    feature_key = 'features' if 'features' in train_data else 'feature'
    label_key = 'labels' if 'labels' in train_data else 'label'
    original_features = train_data[feature_key]
    original_labels = train_data[label_key]
    print(f"Original training features shape: {original_features.shape}, Labels shape: {original_labels.shape}")
    
    # Save original training results (without synthetic data)
    original_output_path = os.path.join(output_dir, "original_train_tangent.npz")
    original_a_matrix_path = os.path.join(output_dir, "original_A_matrix.npy")
    
    print("\nProcessing original training data...")
    original_result, original_a_matrix = compute_product_with_gpu(
        original_features, batch_size=batch_size, device_ids=device_ids
    )
    
    print(f"Saving original A matrix to {original_a_matrix_path}...")
    np.save(original_a_matrix_path, original_a_matrix)
    
    print(f"Saving original training results to {original_output_path}...")
    np.savez(original_output_path, features=original_result, labels=original_labels)
    
    # Load validation data
    print(f"\nLoading validation data from {val_npz}...")
    val_data = np.load(val_npz)
    val_features = val_data[feature_key]
    val_labels = val_data[label_key]
    print(f"Validation features shape: {val_features.shape}, Labels shape: {val_labels.shape}")
    
    # Process validation data with original A matrix
    original_val_output_path = os.path.join(output_dir, "original_val_tangent.npz")
    process_val_data(
        val_features, val_labels, original_a_matrix, 
        original_val_output_path, batch_size, device_ids
    )
    
    # Incrementally add synthetic datasets
    current_features = original_features
    current_labels = original_labels
    
    for i, synthetic_file in enumerate(synthetic_npz_files):
        print(f"\n{'='*80}")
        print(f"Processing synthetic dataset {i+1}/{len(synthetic_npz_files)}: {synthetic_file}")
        
        # Load synthetic data
        synthetic_features, synthetic_labels = load_synthetic_data(synthetic_file)
        
        # Create output paths for this increment
        increment_name = f"increment_{i+1}"
        increment_output_path = os.path.join(output_dir, f"{increment_name}_train_tangent.npz")
        increment_a_matrix_path = os.path.join(output_dir, f"{increment_name}_A_matrix.npy")
        increment_val_output_path = os.path.join(output_dir, f"{increment_name}_val_tangent.npz")
        
        # Process combined data
        combined_features = np.concatenate([current_features, synthetic_features], axis=0)
        combined_labels = np.concatenate([current_labels, synthetic_labels], axis=0)
        
        print(f"Combined features shape: {combined_features.shape}, Labels shape: {combined_labels.shape}")
        
        # Compute product and get A
        result, a_matrix = compute_product_with_gpu(
            combined_features, batch_size=batch_size, device_ids=device_ids
        )
        
        # Save A matrix
        print(f"Saving A matrix to {increment_a_matrix_path}...")
        np.save(increment_a_matrix_path, a_matrix)
        
        # Save combined results
        print(f"Saving combined results to {increment_output_path}...")
        np.savez(increment_output_path, features=result, labels=combined_labels)
        
        # Process validation data with this increment's A matrix
        process_val_data(
            val_features, val_labels, a_matrix, 
            increment_val_output_path, batch_size, device_ids
        )
        
        # Update current data for next iteration
        current_features = combined_features
        current_labels = combined_labels
        
        print(f"Completed processing increment {i+1}/{len(synthetic_npz_files)}")
    
    print("\nAll processing completed successfully!")

if __name__ == "__main__":
    # Configuration
    original_train_npz = "/data/yyang409/bowen/imagenet_feature/swin_base/patch4_window7_224/image_features_w_label_train.npz"
    val_npz = "/data/yyang409/bowen/imagenet_feature/swin_base/patch4_window7_224/image_features_w_label_val.npz"
    
    # Synthetic data files - update this path pattern as needed
    synthetic_data_pattern = "/scratch/bowenxi/dit/data_gen/B_4/final_npz_features_labels/renamed_imagenet_latents_*.npz"
    synthetic_files = sorted(glob.glob(synthetic_data_pattern))
    
    if not synthetic_files:
        print(f"ERROR: No synthetic data files found matching pattern {synthetic_data_pattern}")
        exit(1)
    
    print(f"Found {len(synthetic_files)} synthetic data files:")
    for i, f in enumerate(synthetic_files):
        print(f"  {i+1}. {f}")
    
    # Output directory
    output_dir = "/scratch/bowenxi/dit/neural_tangent_kernel/feature_swin_b/incremental_synthetic"
    
    # Process with incremental synthetic data
    incremental_processing_with_synthetic(
        original_train_npz=original_train_npz,
        synthetic_npz_files=synthetic_files,
        val_npz=val_npz,
        output_dir=output_dir,
        batch_size=20000,
        device_ids=list(range(torch.cuda.device_count()))
    )