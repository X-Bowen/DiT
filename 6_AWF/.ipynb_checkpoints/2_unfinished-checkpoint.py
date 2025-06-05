import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import os
import glob
import matplotlib.pyplot as plt
import pandas as pd
import time
from datetime import datetime
import concurrent.futures
import torch.multiprocessing as mp

# Configuration
BASE_DATA_DIR = "/scratch/bowenxi/dit/neural_tangent_kernel/feature_swin_b/auto_ML_CA_1/auto_ML_CA"
RESULTS_DIR = "kernel_polynomial_results_0504"
NUM_CLASSES = 1000
BATCH_SIZE = 2048
EPOCHS = 900
LEARNING_RATES = [1e-6, 5e-6, 1e-5, 5e-5, 1e-4, 5e-4]
WEIGHT_DECAY = 5e-4
MAX_WORKERS = len(LEARNING_RATES)  # Number of parallel processes

# Make sure results directory exists
os.makedirs(RESULTS_DIR, exist_ok=True)

# Path to precomputed matrices
F_PATH = os.path.join(BASE_DATA_DIR, "feature_matrix_F.npz")  # Path to feature matrix F
A_PATH = os.path.join(BASE_DATA_DIR, "kernel_matrix_A.npz")   # Path to A = F^T * F
FA_PATH = os.path.join(BASE_DATA_DIR, "FA_matrix.npz")        # Path to F*A

def load_precomputed_matrices():
    """
    Load the precomputed matrices F, A, and F*A
    
    Returns:
        Dictionary containing the matrices
    """
    matrices = {}
    
    # These paths and loading methods might need to be adjusted based on how you saved the matrices
    print("Loading precomputed matrices...")
    
    # Method 1: Direct loading if matrices are saved as npz files
    try:
        print(f"Loading feature matrix F from {F_PATH}...")
        with np.load(F_PATH) as data:
            matrices['F'] = data['features']
        
        print(f"Loading kernel matrix A from {A_PATH}...")
        with np.load(A_PATH) as data:
            matrices['A'] = data['kernel']
        
        print(f"Loading F*A matrix from {FA_PATH}...")
        with np.load(FA_PATH) as data:
            matrices['FA'] = data['matrix']
            
        print(f"Matrices loaded successfully. F shape: {matrices['F'].shape}")
        return matrices
        
    except Exception as e:
        print(f"Error loading precomputed matrices: {e}")
        print("Falling back to loading from incremental data...")
        
    # Method 2: Fallback - Load F from the original increment and compute the rest
    # This is a simplified fallback and may need adjustment
    try:
        original_path = os.path.join(BASE_DATA_DIR, "original_train_tangent.npz")
        print(f"Loading features from {original_path}...")
        
        with np.load(original_path) as data:
            F = data['features']
            labels = data['labels']
            
        print(f"Computing kernel matrix A = F^T * F...")
        # This may be memory intensive for large matrices
        A = np.matmul(F.T, F)
        
        print(f"Computing F*A matrix...")
        FA = np.matmul(F, A)
        
        matrices = {
            'F': F,
            'A': A,
            'FA': FA,
            'labels': labels
        }
        
        print(f"Matrices computed. F shape: {F.shape}, A shape: {A.shape}")
        return matrices
        
    except Exception as e:
        print(f"Error in fallback loading: {e}")
        raise
        
def load_data_increment(increment_name, matrices=None):
    """
    Load training and validation data for a specific increment
    
    Args:
        increment_name: Name of the increment (e.g., 'original', 'increment_1', etc.)
        matrices: Precomputed matrices (F, A, FA) if available
    
    Returns:
        X_train, y_train, X_val, y_val
    """
    train_path = os.path.join(BASE_DATA_DIR, f"{increment_name}_train_tangent.npz")
    val_path = os.path.join(BASE_DATA_DIR, f"{increment_name}_val_tangent.npz")
    
    print(f"Loading {increment_name} training data from {train_path}...")
    with np.load(train_path) as data:
        X_train = data['features']
        y_train = data['labels']
    
    print(f"Loading {increment_name} validation data from {val_path}...")
    with np.load(val_path) as data:
        X_val = data['features']
        y_val = data['labels']
    
    print(f"{increment_name} training data: {X_train.shape}, validation data: {X_val.shape}")
    
    return X_train, y_train, X_val, y_val

def standardize_features(X_train, X_val):
    """
    Standardize features using training data statistics
    
    Args:
        X_train: Training features
        X_val: Validation features
    
    Returns:
        Standardized X_train, X_val
    """
    print("Standardizing features...")
    mean = np.mean(X_train, axis=0)
    std = np.std(X_train, axis=0) + 1e-8
    
    X_train_std = (X_train - mean) / std
    X_val_std = (X_val - mean) / std
    
    return X_train_std, X_val_std

class KernelPolynomialMLP(nn.Module):
    def __init__(self, input_dim, num_classes, use_polynomial=True):
        super().__init__()
        self.use_polynomial = use_polynomial
        
        # Trainable weights for the polynomial expansion
        if use_polynomial:
            # Initialize with small values to maintain stability
            self.w0 = nn.Parameter(torch.tensor(0.1, dtype=torch.float32))
            self.w1 = nn.Parameter(torch.tensor(0.01, dtype=torch.float32))
            self.w2 = nn.Parameter(torch.tensor(0.001, dtype=torch.float32))
        
        # Classifier layers
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x, F=None, A=None, FA=None, FA2=None, FA3=None):
        """
        Forward pass with optional polynomial kernel transformation
        
        Args:
            x: Input features (can be raw F or already transformed)
            F: Original feature matrix (optional)
            A: Kernel matrix (optional)
            FA: F*A matrix (optional)
            FA2: F*A^2 matrix (optional)
            FA3: F*A^3 matrix (optional)
        
        Returns:
            Model output
        """
        # Apply polynomial transformation if enabled and matrices are provided
        if self.use_polynomial and F is not None and A is not None:
            # Compute the transformation F⋅(I+w0A+w1A²+w2A³) on the fly
            # This is a simplified version; in practice, precompute FA2 and FA3
            if FA is None:
                # Identity term (F*I = F)
                transformed_x = x
                
                # Add w0*F*A term
                if hasattr(self, 'A_tensor') and hasattr(self, 'F_tensor'):
                    FA_term = torch.matmul(self.F_tensor, self.A_tensor)
                    transformed_x = transformed_x + self.w0 * FA_term
                    
                    # Add w1*F*A^2 term
                    FA2_term = torch.matmul(FA_term, self.A_tensor)
                    transformed_x = transformed_x + self.w1 * FA2_term
                    
                    # Add w2*F*A^3 term
                    FA3_term = torch.matmul(FA2_term, self.A_tensor)
                    transformed_x = transformed_x + self.w2 * FA3_term
            else:
                # If FA, FA2, FA3 are provided directly
                transformed_x = x + self.w0 * FA
                if FA2 is not None:
                    transformed_x = transformed_x + self.w1 * FA2
                if FA3 is not None:
                    transformed_x = transformed_x + self.w2 * FA3
                
            x = transformed_x.float()
        
        # Apply the classifier layers
        return self.layers(x)

def precompute_polynomial_terms(F, A, device=None):
    """
    Precompute the terms needed for the polynomial expansion
    
    Args:
        F: Feature matrix
        A: Kernel matrix
        device: PyTorch device
    
    Returns:
        Dictionary with precomputed terms
    """
    print("Precomputing polynomial terms...")
    
    # Convert to PyTorch tensors if needed
    if not isinstance(F, torch.Tensor):
        F_tensor = torch.FloatTensor(F)
    else:
        F_tensor = F
        
    if not isinstance(A, torch.Tensor):
        A_tensor = torch.FloatTensor(A)
    else:
        A_tensor = A
    
    # Move to device if specified
    if device is not None:
        F_tensor = F_tensor.to(device)
        A_tensor = A_tensor.to(device)
    
    # Compute F*A
    print("Computing F*A...")
    FA = torch.matmul(F_tensor, A_tensor)
    
    # Compute F*A^2
    print("Computing F*A^2...")
    FA2 = torch.matmul(FA, A_tensor)
    
    # Compute F*A^3
    print("Computing F*A^3...")
    FA3 = torch.matmul(FA2, A_tensor)
    
    return {
        'F': F_tensor,
        'A': A_tensor,
        'FA': FA,
        'FA2': FA2,
        'FA3': FA3
    }

def get_available_gpu():
    """
    Get the GPU with the most available memory
    
    Returns:
        GPU device ID with most available memory
    """
    if not torch.cuda.is_available():
        return None
    
    # Get the number of available GPUs
    num_gpus = torch.cuda.device_count()
    
    if num_gpus == 1:
        return 0  # Only one GPU available
    
    # Find the GPU with the most available memory
    available_memory = []
    for i in range(num_gpus):
        # Get free memory in bytes
        free_memory = torch.cuda.get_device_properties(i).total_memory - torch.cuda.memory_allocated(i)
        available_memory.append((i, free_memory))
    
    # Sort by available memory (descending)
    available_memory.sort(key=lambda x: x[1], reverse=True)
    
    # Return the device ID with the most available memory
    return available_memory[0][0]

def train_model_worker(params):
    """
    Worker function for parallel training
    
    Args:
        params: Dictionary with parameters for training
    
    Returns:
        Dictionary with training results
    """
    # Extract parameters
    X_train = params['X_train']
    y_train = params['y_train']
    X_val = params['X_val']
    y_val = params['y_val']
    increment_name = params['increment_name']
    learning_rate = params['learning_rate']
    gpu_id = params.get('gpu_id', get_available_gpu())
    precomputed_terms = params.get('precomputed_terms', None)
    use_polynomial = params.get('use_polynomial', True)
    
    # Set device
    device = torch.device(f"cuda:{gpu_id}" if gpu_id is not None else "cpu")
    
    print(f"Training {increment_name} with LR={learning_rate:.1e} on {device}")
    print(f"Using polynomial transformation: {use_polynomial}")
    
    # Convert to PyTorch tensors
    train_dataset = TensorDataset(
        torch.FloatTensor(X_train),
        torch.LongTensor(y_train)
    )
    val_dataset = TensorDataset(
        torch.FloatTensor(X_val),
        torch.LongTensor(y_val)
    )
    
    # Create loaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
    
    # Initialize model
    model = KernelPolynomialMLP(X_train.shape[1], NUM_CLASSES, use_polynomial=use_polynomial)
    model = model.to(device)
    
    # If using polynomial transformation, prepare matrices
    if use_polynomial and precomputed_terms is not None:
        # Move precomputed terms to the right device
        F_device = precomputed_terms['F'].to(device) if torch.is_tensor(precomputed_terms['F']) else torch.FloatTensor(precomputed_terms['F']).to(device)
        A_device = precomputed_terms['A'].to(device) if torch.is_tensor(precomputed_terms['A']) else torch.FloatTensor(precomputed_terms['A']).to(device)
        FA_device = precomputed_terms['FA'].to(device) if torch.is_tensor(precomputed_terms['FA']) else torch.FloatTensor(precomputed_terms['FA']).to(device)
        FA2_device = precomputed_terms.get('FA2', None)
        if FA2_device is not None and not torch.is_tensor(FA2_device):
            FA2_device = torch.FloatTensor(FA2_device).to(device)
        FA3_device = precomputed_terms.get('FA3', None)
        if FA3_device is not None and not torch.is_tensor(FA3_device):
            FA3_device = torch.FloatTensor(FA3_device).to(device)
        
        # Store references to tensors in model for use during training
        model.F_tensor = F_device
        model.A_tensor = A_device
        model.FA_tensor = FA_device
        model.FA2_tensor = FA2_device
        model.FA3_tensor = FA3_device
    
    # Training setup
    criterion = nn.CrossEntropyLoss()
    
    # Use different parameter groups for kernel weights and network weights
    if use_polynomial:
        optimizer = torch.optim.Adam([
            {'params': [model.w0, model.w1, model.w2], 'lr': learning_rate * 0.1},  # Lower LR for polynomial weights
            {'params': model.layers.parameters(), 'lr': learning_rate}
        ], weight_decay=WEIGHT_DECAY)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=WEIGHT_DECAY)
    
    # Use CosineAnnealingLR scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=0)
    
    # Tracking metrics
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
        'polynomial_weights': [] if use_polynomial else None
    }
    
    # Training loop
    best_acc = 0
    best_epoch = 0
    start_time = time.time()
    
    for epoch in range(EPOCHS):
        # Training
        model.train()
        train_loss = 0
        correct = 0
        total = 0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            
            # Forward pass with precomputed terms if available
            if use_polynomial and hasattr(model, 'FA_tensor'):
                outputs = model(inputs, 
                               F=model.F_tensor if hasattr(model, 'F_tensor') else None,
                               A=model.A_tensor if hasattr(model, 'A_tensor') else None, 
                               FA=model.FA_tensor if hasattr(model, 'FA_tensor') else None,
                               FA2=model.FA2_tensor if hasattr(model, 'FA2_tensor') else None,
                               FA3=model.FA3_tensor if hasattr(model, 'FA3_tensor') else None)
            else:
                outputs = model(inputs)
                
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
        
        train_acc = correct / total
        avg_train_loss = train_loss / len(train_loader)
        history['train_loss'].append(avg_train_loss)
        history['train_acc'].append(train_acc)
        
        # Save polynomial weights if using them
        if use_polynomial:
            history['polynomial_weights'].append({
                'w0': model.w0.item(),
                'w1': model.w1.item(),
                'w2': model.w2.item()
            })
        
        if epoch % 50 == 0:  # Print every 50 epochs to reduce output
            print(f"LR {learning_rate:.1e} - Epoch {epoch+1}: Train Loss: {avg_train_loss:.4f} | Train Acc: {train_acc:.2%}")
            if use_polynomial:
                print(f"   Polynomial weights: w0={model.w0.item():.4f}, w1={model.w1.item():.4f}, w2={model.w2.item():.4f}")
        
        # Validation
        model.eval()
        val_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                
                # Forward pass with precomputed terms if available
                if use_polynomial and hasattr(model, 'FA_tensor'):
                    outputs = model(inputs, 
                                  F=model.F_tensor if hasattr(model, 'F_tensor') else None,
                                  A=model.A_tensor if hasattr(model, 'A_tensor') else None, 
                                  FA=model.FA_tensor if hasattr(model, 'FA_tensor') else None,
                                  FA2=model.FA2_tensor if hasattr(model, 'FA2_tensor') else None,
                                  FA3=model.FA3_tensor if hasattr(model, 'FA3_tensor') else None)
                else:
                    outputs = model(inputs)
                    
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        
        val_acc = correct / total
        avg_val_loss = val_loss / len(val_loader)
        history['val_loss'].append(avg_val_loss)
        history['val_acc'].append(val_acc)
        
        if epoch % 50 == 0:  # Print every 50 epochs to reduce output
            print(f"LR {learning_rate:.1e} - Epoch {epoch+1}: Val Loss: {avg_val_loss:.4f} | Val Acc: {val_acc:.2%}")
        
        # Step the scheduler
        scheduler.step()
        
        # Save best model
        if val_acc > best_acc:
            best_acc = val_acc
            best_epoch = epoch
            model_path = os.path.join(RESULTS_DIR, f"best_model_{increment_name}_lr_{learning_rate:.1e}.pt")
            torch.save(model.state_dict(), model_path)
            
            # If using polynomial, also save the best weights separately
            if use_polynomial:
                best_weights = {
                    'w0': model.w0.item(),
                    'w1': model.w1.item(),
                    'w2': model.w2.item(),
                }
                weight_path = os.path.join(RESULTS_DIR, f"best_weights_{increment_name}_lr_{learning_rate:.1e}.json")
                with open(weight_path, 'w') as f:
                    import json
                    json.dump(best_weights, f, indent=2)
    
    training_time = time.time() - start_time
    print(f"Training completed for LR={learning_rate:.1e} in {training_time:.2f} seconds.")
    print(f"Best validation accuracy: {best_acc:.4f} at epoch {best_epoch+1}")
    
    # If using polynomial, print final weights
    if use_polynomial:
        print(f"Final polynomial weights: w0={model.w0.item():.4f}, w1={model.w1.item():.4f}, w2={model.w2.item():.4f}")
    
    # Plot history after training
    plot_training_history(history, increment_name, learning_rate, use_polynomial)
    
    # Clean up GPU memory
    model = model.cpu()
    torch.cuda.empty_cache()
    
    # Return results
    result = {
        'increment': increment_name,
        'learning_rate': learning_rate,
        'use_polynomial': use_polynomial,
        'history': history,
        'best_acc': best_acc,
        'best_epoch': best_epoch,
        'training_time': training_time
    }
    
    # Add polynomial weights if used
    if use_polynomial:
        result['final_weights'] = {
            'w0': model.w0.item(),
            'w1': model.w1.item(),
            'w2': model.w2.item()
        }
    
    return result

def plot_training_history(history, increment_name, learning_rate, use_polynomial=True):
    """
    Plot training and validation metrics
    
    Args:
        history: Dictionary with training history
        increment_name: Name of the increment for saving plot
        learning_rate: Learning rate used for training
        use_polynomial: Whether polynomial weights were used
    """
    epochs = range(1, len(history['train_loss']) + 1)
    
    # Create a figure with 2 or 3 subplots depending on whether we have polynomial weights
    n_plots = 3 if use_polynomial else 2
    plt.figure(figsize=(12, 5*n_plots))
    
    # Plot loss
    plt.subplot(n_plots, 1, 1)
    plt.plot(epochs, history['train_loss'], 'b-', label='Training Loss')
    plt.plot(epochs, history['val_loss'], 'r-', label='Validation Loss')
    plt.title(f'Training and Validation Loss ({increment_name}, LR={learning_rate:.1e})')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    # Plot accuracy
    plt.subplot(n_plots, 1, 2)
    plt.plot(epochs, history['train_acc'], 'b-', label='Training Accuracy')
    plt.plot(epochs, history['val_acc'], 'r-', label='Validation Accuracy')
    plt.title(f'Training and Validation Accuracy ({increment_name}, LR={learning_rate:.1e})')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    
    # Plot polynomial weights if used
    if use_polynomial and history['polynomial_weights'] is not None:
        plt.subplot(n_plots, 1, 3)
        w0_values = [w['w0'] for w in history['polynomial_weights']]
        w1_values = [w['w1'] for w in history['polynomial_weights']]
        w2_values = [w['w2'] for w in history['polynomial_weights']]
        
        plt.plot(epochs, w0_values, 'g-', label='w0 (A)')
        plt.plot(epochs, w1_values, 'b-', label='w1 (A²)')
        plt.plot(epochs, w2_values, 'r-', label='w2 (A³)')
        plt.title(f'Polynomial Weights Evolution ({increment_name}, LR={learning_rate:.1e})')
        plt.xlabel('Epochs')
        plt.ylabel('Weight Value')
        plt.legend()
        plt.grid(True)
    
    plt.tight_layout()
    poly_tag = "poly" if use_polynomial else "nopoly"
    plt.savefig(os.path.join(RESULTS_DIR, f'training_history_{increment_name}_lr_{learning_rate:.1e}_{poly_tag}.png'), dpi=300)
    plt.close()

def get_increment_size(increment_name):
    """
    Get the size of an increment (number of classes and samples)
    
    Args:
        increment_name: Name of the increment
    
    Returns:
        Dictionary with increment statistics
    """
    if increment_name == 'original':
        synthetic_classes = 0
        synthetic_samples_per_class = 0
    else:
        increment_num = int(increment_name.split('_')[1])
        synthetic_classes = NUM_CLASSES
        synthetic_samples_per_class = 1000 * increment_num  # Assuming 1000 samples per class per increment
    
    return {
        'name': increment_name,
        'synthetic_classes': synthetic_classes,
        'synthetic_samples_per_class': synthetic_samples_per_class,
        'total_synthetic': synthetic_classes * synthetic_samples_per_class
    }

def run_incremental_experiment_parallel():
    """
    Run experiment for each incremental dataset with different learning rates in parallel
    
    Returns:
        DataFrame with results
    """
    # Find all increment files
    train_files = sorted(glob.glob(os.path.join(BASE_DATA_DIR, "*_train_tangent.npz")))
    increment_names = [os.path.basename(f).replace("_train_tangent.npz", "") for f in train_files]
    
    print(f"Found {len(increment_names)} increments: {increment_names}")
    
    # Load precomputed matrices
    try:
        matrices = load_precomputed_matrices()
    except Exception as e:
        print(f"Warning: Could not load precomputed matrices: {e}")
        print("Continuing without polynomial transformation")
        matrices = None
    
    # Initialize torch multiprocessing
    mp.set_start_method('spawn', force=True)
    
    # Store results
    all_results = []
    
    # Process each increment
    for increment_name in increment_names:
        print(f"\n{'='*80}")
        print(f"Processing increment: {increment_name}")
        print(f"{'='*80}")
        
        # Load data
        X_train, y_train, X_val, y_val = load_data_increment(increment_name, matrices)
        
        # Standardize features
        X_train_std, X_val_std = standardize_features(X_train, X_val)
        
        # Precompute polynomial terms if matrices are available
        precomputed_terms = None
        if matrices is not None:
            # This is a placeholder - actual implementation would depend on how matrices are stored
            precomputed_terms = precompute_polynomial_terms(
                matrices.get('F', X_train),
                matrices.get('A'),
                device=None  # Will move to device in worker function
            )
        
        # Free up memory for original features
        del X_train, X_val
        
        # Get increment statistics
        increment_stats = get_increment_size(increment_name)
        
        # Prepare parameters for parallel training
        training_params = []
        
        # Train with polynomial transformation
        for i, lr in enumerate(LEARNING_RATES):
            # Assign GPUs in a round-robin fashion if multiple are available
            gpu_id = i % torch.cuda.device_count() if torch.cuda.is_available() else None
            
            training_params.append({
                'X_train': X_train_std,
                'y_train': y_train,
                'X_val': X_val_std,
                'y_val': y_val,
                'increment_name': increment_name,
                'learning_rate': lr,
                'gpu_id': gpu_id,
                'precomputed_terms': precomputed_terms,
                'use_polynomial': True
            })
        
        # Also try without polynomial transformation as baseline
        for i, lr in enumerate(LEARNING_RATES[:3]):  # Use fewer LRs for baseline to save time
            gpu_id = (i + len(LEARNING_RATES)) % torch.cuda.device_count() if torch.cuda.is_available() else None
            
            training_params.append({
                'X_train': X_train_std,
                'y_train': y_train,
                'X_val': X_val_std,
                'y_val': y_val,
                'increment_name': increment_name,
                'learning_rate': lr,
                'gpu_id': gpu_id,
                'precomputed_terms': None,
                'use_polynomial': False
            })
        
        # Train models with different learning rates in parallel
        results = []
        
        # Using ProcessPoolExecutor for parallel processing
        with concurrent.futures.ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
            future_to_params = {executor.submit(train_model_worker, params): params 
                              for params in training_params}
            
            for future in concurrent.futures.as_completed(future_to_params):
                params = future_to_params[future]
                try:
                    result = future.result()
                    poly_str = "with polynomial" if params['use_polynomial'] else "baseline"
                    print(f"Completed training for {increment_name} with LR={params['learning_rate']:.1e} {poly_str}")
                    results.append(result)
                except Exception as e:
                    print(f"Training failed for {increment_name} with LR={params['learning_rate']:.1e}: {e}")
        
        # Format and store results
        for result in results:
            result_dict = {
                'increment': result['increment'],
                'learning_rate': result['learning_rate'],
                'use_polynomial': result['use_polynomial']