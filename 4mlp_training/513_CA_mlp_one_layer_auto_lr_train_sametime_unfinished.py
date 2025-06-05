import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, TensorDataset, DistributedSampler
from tqdm import tqdm
import os
import glob
import matplotlib.pyplot as plt
import pandas as pd
import time
import argparse
from datetime import datetime

# Configuration
BASE_DATA_DIR = "/scratch/bowenxi/dit/neural_tangent_kernel/feature_swin_b/incremental_synthetic"
RESULTS_DIR = "incremental_results"
NUM_CLASSES = 1000
BATCH_SIZE = 1024
EPOCHS = 400
LEARNING_RATES = [1e-4, 5e-4, 1e-3, 5e-3]  # Multiple learning rates to test
WEIGHT_DECAY = 1e-4

# Make sure results directory exists
os.makedirs(RESULTS_DIR, exist_ok=True)

def parse_args():
    parser = argparse.ArgumentParser(description='Incremental learning with multiple GPUs')
    parser.add_argument('--distributed', action='store_true', 
                        help='Enable distributed training across multiple GPUs')
    parser.add_argument('--gpu_ids', type=str, default='0,1,2,3',
                        help='Comma-separated list of GPU IDs to use (e.g., "0,1,2,3")')
    parser.add_argument('--world_size', type=int, default=None,
                        help='Number of processes for distributed training')
    parser.add_argument('--rank', type=int, default=None,
                        help='Rank of the current process in distributed training')
    parser.add_argument('--dist_url', default='tcp://127.0.0.1:23456', type=str,
                        help='URL used to set up distributed training')
    parser.add_argument('--dist_backend', default='nccl', type=str,
                        help='Distributed backend')
    parser.add_argument('--split_jobs', action='store_true',
                        help='Split jobs across GPUs without distributed training')
    return parser.parse_args()

def setup_distributed(args):
    """
    Initialize the distributed environment
    """
    if args.distributed:
        print(f"Setting up distributed training with world_size={args.world_size}, rank={args.rank}")
        
        # Initialize the process group
        dist.init_process_group(
            backend=args.dist_backend,
            init_method=args.dist_url,
            world_size=args.world_size,
            rank=args.rank
        )
        
        # Set the device
        torch.cuda.set_device(args.rank % torch.cuda.device_count())
        device = torch.device(f"cuda:{args.rank % torch.cuda.device_count()}")
        return device
    else:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        return device

def load_data_increment(increment_name):
    """
    Load training and validation data for a specific increment
    
    Args:
        increment_name: Name of the increment (e.g., 'original', 'increment_1', etc.)
    
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

class AdaptiveMLP(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.classifier = nn.Linear(input_dim, num_classes)
        
    def forward(self, x):
        return self.classifier(x)


def train_model(X_train, y_train, X_val, y_val, increment_name, learning_rate, device, args=None):
    """
    Train and evaluate MLP with specific learning rate
    
    Args:
        X_train: Training features
        y_train: Training labels
        X_val: Validation features
        y_val: Validation labels
        increment_name: Name of the increment for saving model
        learning_rate: Learning rate to use for training
        device: Device to run training on
        args: Command line arguments
    
    Returns:
        Dictionary with training history and best validation accuracy
    """
    # Convert to PyTorch tensors
    train_dataset = TensorDataset(
        torch.FloatTensor(X_train),
        torch.LongTensor(y_train)
    )
    val_dataset = TensorDataset(
        torch.FloatTensor(X_val),
        torch.LongTensor(y_val)
    )
    
    # Create loaders with distributed sampler if needed
    if args and args.distributed:
        train_sampler = DistributedSampler(
            train_dataset, 
            num_replicas=args.world_size,
            rank=args.rank
        )
        val_sampler = DistributedSampler(
            val_dataset,
            num_replicas=args.world_size,
            rank=args.rank
        )
        
        train_loader = DataLoader(
            train_dataset, 
            batch_size=BATCH_SIZE, 
            sampler=train_sampler
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=BATCH_SIZE,
            sampler=val_sampler
        )
    else:
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
    
    # Initialize model
    model = AdaptiveMLP(X_train.shape[1], NUM_CLASSES)
    model = model.to(device)
    
    # Wrap model in DDP if using distributed training
    if args and args.distributed:
        model = DDP(model, device_ids=[args.rank % torch.cuda.device_count()])
    
    # Training setup
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=WEIGHT_DECAY)
    # Use CosineAnnealingLR scheduler
    T_max = EPOCHS  # Maximum number of iterations
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_max, eta_min=0)
    
    # Tracking metrics
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }
    
    # Training loop
    best_acc = 0
    best_epoch = 0
    start_time = time.time()
    
    for epoch in range(EPOCHS):
        # Set epoch for distributed sampler
        if args and args.distributed:
            train_sampler.set_epoch(epoch)
        
        # Training
        model.train()
        train_loss = 0
        correct = 0
        total = 0
        
        for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}", disable=args and args.distributed and args.rank != 0):
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
        
        # Normalize by world_size in distributed setting
        if args and args.distributed:
            train_loss_tensor = torch.tensor([train_loss, correct, total], device=device)
            dist.all_reduce(train_loss_tensor)
            train_loss, correct, total = train_loss_tensor.tolist()
        
        train_acc = correct / total
        avg_train_loss = train_loss / len(train_loader)
        history['train_loss'].append(avg_train_loss)
        history['train_acc'].append(train_acc)
        
        if not args or args.rank == 0 or not args.distributed:
            print(f"Epoch {epoch+1}: Train Loss: {avg_train_loss:.4f} | Train Acc: {train_acc:.2%}")
        
        # Validation
        model.eval()
        val_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        
        # Normalize by world_size in distributed setting
        if args and args.distributed:
            val_loss_tensor = torch.tensor([val_loss, correct, total], device=device)
            dist.all_reduce(val_loss_tensor)
            val_loss, correct, total = val_loss_tensor.tolist()
        
        val_acc = correct / total
        avg_val_loss = val_loss / len(val_loader)
        history['val_loss'].append(avg_val_loss)
        history['val_acc'].append(val_acc)
        
        if not args or args.rank == 0 or not args.distributed:
            print(f"Epoch {epoch+1}: Val Loss: {avg_val_loss:.4f} | Val Acc: {val_acc:.2%}")
        
        # Step the scheduler
        scheduler.step()
        
        # Save best model (only in rank 0 if distributed)
        if val_acc > best_acc and (not args or args.rank == 0 or not args.distributed):
            best_acc = val_acc
            best_epoch = epoch
            model_path = os.path.join(RESULTS_DIR, f"best_model_{increment_name}_lr_{learning_rate:.1e}.pt")
            
            # Save the model state dict without DDP wrapper
            if args and args.distributed:
                torch.save(model.module.state_dict(), model_path)
            else:
                torch.save(model.state_dict(), model_path)
                
            if not args or args.rank == 0 or not args.distributed:
                print(f"Best model saved with validation accuracy: {val_acc:.2%}")
    
    training_time = time.time() - start_time
    if not args or args.rank == 0 or not args.distributed:
        print(f"Training completed in {training_time:.2f} seconds.")
        print(f"Best validation accuracy: {best_acc:.4f} at epoch {best_epoch+1}")
    
    return {
        'history': history,
        'best_acc': best_acc,
        'best_epoch': best_epoch,
        'training_time': training_time
    }

def plot_training_history(history, increment_name, learning_rate):
    """
    Plot training and validation metrics
    
    Args:
        history: Dictionary with training history
        increment_name: Name of the increment for saving plot
        learning_rate: Learning rate used for training
    """
    epochs = range(1, len(history['train_loss']) + 1)
    
    plt.figure(figsize=(12, 10))
    
    # Plot loss
    plt.subplot(2, 1, 1)
    plt.plot(epochs, history['train_loss'], 'b-', label='Training Loss')
    plt.plot(epochs, history['val_loss'], 'r-', label='Validation Loss')
    plt.title(f'Training and Validation Loss ({increment_name}, LR={learning_rate:.1e})')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    # Plot accuracy
    plt.subplot(2, 1, 2)
    plt.plot(epochs, history['train_acc'], 'b-', label='Training Accuracy')
    plt.plot(epochs, history['val_acc'], 'r-', label='Validation Accuracy')
    plt.title(f'Training and Validation Accuracy ({increment_name}, LR={learning_rate:.1e})')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, f'training_history_{increment_name}_lr_{learning_rate:.1e}.png'), dpi=300)
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

def run_distributed_experiment(local_rank, args):
    """
    Run experiment in distributed mode
    
    Args:
        local_rank: Local rank of current process
        args: Command line arguments
    """
    # Set rank and initialize distributed process
    if args.rank is None:
        args.rank = local_rank
    
    device = setup_distributed(args)
    
    # Process only relevant increments/learning rates for this GPU
    train_files = sorted(glob.glob(os.path.join(BASE_DATA_DIR, "*_train_tangent.npz")))
    increment_names = [os.path.basename(f).replace("_train_tangent.npz", "") for f in train_files]
    
    if args.rank == 0:
        print(f"Found {len(increment_names)} increments: {increment_names}")
    
    # Store results
    results = []
    
    # Run experiments
    for increment_name in increment_names:
        if args.rank == 0:
            print(f"\n{'='*80}")
            print(f"Processing increment: {increment_name}")
            print(f"{'='*80}")
        
        # Load data
        X_train, y_train, X_val, y_val = load_data_increment(increment_name)
        
        # Standardize features
        X_train_std, X_val_std = standardize_features(X_train, X_val)
        
        # Free up memory
        del X_train, X_val
        
        # Get increment statistics
        increment_stats = get_increment_size(increment_name)
        
        # Train models with different learning rates
        for lr in LEARNING_RATES:
            if args.rank == 0:
                print(f"\n{'-'*60}")
                print(f"Training with learning rate: {lr:.1e}")
                print(f"{'-'*60}")
            
            # Train model with specific learning rate
            training_results = train_model(X_train_std, y_train, X_val_std, y_val, 
                                          increment_name, lr, device, args)
            
            # Only rank 0 plots and saves results
            if args.rank == 0:
                # Plot history
                plot_training_history(training_results['history'], increment_name, lr)
                
                # Store results
                results.append({
                    'increment': increment_name,
                    'learning_rate': lr,
                    'train_samples': len(X_train_std),
                    'val_accuracy': training_results['best_acc'],
                    'best_epoch': training_results['best_epoch'] + 1,
                    'training_time': training_results['training_time'],
                    'synthetic_classes': increment_stats['synthetic_classes'],
                    'synthetic_samples_per_class': increment_stats['synthetic_samples_per_class'],
                    'total_synthetic': increment_stats['total_synthetic']
                })
        
        # Free up memory
        del X_train_std, X_val_std, y_train, y_val
        torch.cuda.empty_cache()
    
    # Only rank 0 saves results
    if args.rank == 0:
        # Convert to DataFrame
        results_df = pd.DataFrame(results)
        
        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_df.to_csv(os.path.join(RESULTS_DIR, f'incremental_results_{timestamp}.csv'), index=False)
        
        # Display results in sorted order
        sorted_results = results_df.sort_values(['increment', 'learning_rate'])
        print("\nIncremental Results:")
        print("=" * 120)
        print(sorted_results.to_string(index=False))
        print("=" * 120)
        
        # Plot final results
        plot_learning_rate_comparison(results_df)
    
    if args.distributed:
        dist.destroy_process_group()

def run_split_experiment(args):
    """
    Run experiment by splitting work across GPUs (each GPU handles different jobs)
    
    Args:
        args: Command line arguments
    """
    # Parse GPU IDs
    gpu_ids = [int(id) for id in args.gpu_ids.split(',')]
    num_gpus = len(gpu_ids)
    
    # Find all increment files
    train_files = sorted(glob.glob(os.path.join(BASE_DATA_DIR, "*_train_tangent.npz")))
    increment_names = [os.path.basename(f).replace("_train_tangent.npz", "") for f in train_files]
    
    print(f"Found {len(increment_names)} increments: {increment_names}")
    
    # Create job list (increment + learning rate combinations)
    jobs = []
    for increment_name in increment_names:
        for lr in LEARNING_RATES:
            jobs.append((increment_name, lr))
    
    print(f"Total jobs: {len(jobs)}")
    
    # Split jobs across GPUs
    gpu_to_jobs = {}
    for i, job in enumerate(jobs):
        gpu_id = gpu_ids[i % num_gpus]
        if gpu_id not in gpu_to_jobs:
            gpu_to_jobs[gpu_id] = []
        gpu_to_jobs[gpu_id].append(job)
    
    # Store results
    all_results = []
    
    # Process each increment and learning rate pair
    for gpu_id, job_list in gpu_to_jobs.items():
        device = torch.device(f"cuda:{gpu_id}")
        print(f"\nProcessing {len(job_list)} jobs on GPU {gpu_id}")
        
        for increment_name, lr in job_list:
            print(f"\n{'='*80}")
            print(f"Processing increment: {increment_name} with learning rate: {lr:.1e} on GPU {gpu_id}")
            print(f"{'='*80}")
            
            # Load data
            X_train, y_train, X_val, y_val = load_data_increment(increment_name)
            
            # Standardize features
            X_train_std, X_val_std = standardize_features(X_train, X_val)
            
            # Free up memory
            del X_train, X_val
            
            # Get increment statistics
            increment_stats = get_increment_size(increment_name)
            
            # Train model with specific learning rate
            training_results = train_model(X_train_std, y_train, X_val_std, y_val, 
                                          increment_name, lr, device)
            
            # Plot history
            plot_training_history(training_results['history'], increment_name, lr)
            
            # Store results
            all_results.append({
                'increment': increment_name,
                'learning_rate': lr,
                'train_samples': len(X_train_std),
                'val_accuracy': training_results['best_acc'],
                'best_epoch': training_results['best_epoch'] + 1,
                'training_time': training_results['training_time'],
                'gpu_id': gpu_id,
                'synthetic_classes': increment_stats['synthetic_classes'],
                'synthetic_samples_per_class': increment_stats['synthetic_samples_per_class'],
                'total_synthetic': increment_stats['total_synthetic']
            })
            
            # Free up memory
            del X_train_std, X_val_std, y_train, y_val
            torch.cuda.empty_cache()
    
    # Convert to DataFrame
    results_df = pd.DataFrame(all_results)
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_df.to_csv(os.path.join(RESULTS_DIR, f'incremental_results_{timestamp}.csv'), index=False)
    
    # Display results in sorted order
    sorted_results = results_df.sort_values(['increment', 'learning_rate'])
    print("\nIncremental Results:")
    print("=" * 120)
    print(sorted_results.to_string(index=False))
    print("=" * 120)
    
    # Plot final results
    return plot_learning_rate_comparison(results_df)

def run_incremental_experiment(args=None):
    """
    Run experiment for each incremental dataset with different learning rates
    
    Args:
        args: Command line arguments
    
    Returns:
        DataFrame with results
    """
    if args and args.distributed:
        # For distributed training, this should not be called directly
        print("For distributed training, use distributed.launch instead")
        return None
    elif args and args.split_jobs:
        # Split jobs across GPUs
        return run_split_experiment(args)
    
    # Standard single-GPU training
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Find all increment files
    train_files = sorted(glob.glob(os.path.join(BASE_DATA_DIR, "*_train_tangent.npz")))
    increment_names = [os.path.basename(f).replace("_train_tangent.npz", "") for f in train_files]
    
    print(f"Found {len(increment_names)} increments: {increment_names}")
    
    # Store results
    results = []
    
    # Run experiment for each increment and learning rate
    for increment_name in increment_names:
        print(f"\n{'='*80}")
        print(f"Processing increment: {increment_name}")
        print(f"{'='*80}")
        
        # Load data
        X_train, y_train, X_val, y_val = load_data_increment(increment_name)
        
        # Standardize features
        X_train_std, X_val_std = standardize_features(X_train, X_val)
        
        # Free up memory
        del X_train, X_val
        
        # Get increment statistics
        increment_stats = get_increment_size(increment_name)
        
        # Train models with different learning rates
        for lr in LEARNING_RATES:
            print(f"\n{'-'*60}")
            print(f"Training with learning rate: {lr:.1e}")
            print(f"{'-'*60}")
            
            # Train model with specific learning rate
            training_results = train_model(X_train_std, y_train, X_val_std, y_val, increment_name, lr, device)
            
            # Plot history
            plot_training_history(training_results['history'], increment_name, lr)
            
            # Store results
            results.append({
                'increment': increment_name,
                'learning_rate': lr,
                'train_samples': len(X_train_std),
                'val_accuracy': training_results['best_acc'],
                'best_epoch': training_results['best_epoch'] + 1,
                'training_time': training_results['training_time'],
                'synthetic_classes': increment_stats['synthetic_classes'],
                'synthetic_samples_per_class': increment_stats['synthetic_samples_per_class'],
                'total_synthetic': increment_stats['total_synthetic']
            })
        
        # Free up memory
        del X_train_std, X_val_std, y_train, y_val
        torch.cuda.empty_cache()
    
    # Convert to DataFrame
    results_df = pd.DataFrame(results)
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_df.to_csv(os.path.join(RESULTS_DIR, f'incremental_results_{timestamp}.csv'), index=False)
    
    # Display results in sorted order
    sorted_results = results_df.sort_values(['increment', 'learning_rate'])
    print("\nIncremental Results:")
    print("=" * 120)
    print(sorted_results.to_string(index=False))
    print("=" * 120)
    
    return results_df

def plot_learning_rate_comparison(results_df):
    """
    Create plots comparing different learning rates
    
    Args:
        results_df: DataFrame with results
    """
    # Process the increment names to get proper ordering
    results_df['increment_order'] = results_df['increment'].apply(
        lambda x: 0 if x == 'original' else int(x.split('_')[1])
    )
    
    # Plot validation accuracy vs learning rate for each increment
    increments = sorted(results_df['increment'].unique(), 
                        key=lambda x: 0 if x == 'original' else int(x.split('_')[1]))
    
    plt.figure(figsize=(14, 8))
    
    for increment in increments:
        subset = results_df[results_df['increment'] == increment]
        plt.plot(subset['learning_rate'], subset['val_accuracy'], 'o-', label=increment, markersize=8)
    
    plt.xscale('log')
    plt.xlabel('Learning Rate (log scale)', fontsize=14)
    plt.ylabel('Validation Accuracy', fontsize=14)
    plt.title('Validation Accuracy vs Learning Rate by Increment', fontsize=16)
    plt.grid(True)
    plt.legend(title='Increment')
    
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'accuracy_vs_learning_rate.png'), dpi=300)
    
    # Find best learning rate for each increment
    best_params = results_df.loc[results_df.groupby('increment')['val_accuracy'].idxmax()]
    
    # Plot best accuracy vs increment with learning rate annotation
    plt.figure(figsize=(14, 8))
    plt.plot(best_params['increment_order'], best_params['val_accuracy'], 'bo-', markersize=8)
    plt.xlabel('Increment Number (0=original)', fontsize=14)
    plt.ylabel('Best Validation Accuracy', fontsize=14)
    plt.title('Best ImageNet Classification Accuracy vs Increment Level', fontsize=16)
    plt.grid(True)
    plt.xticks(best_params['increment_order'])
    
    # Add text labels with learning rate info
    for i, row in best_params.iterrows():
        plt.annotate(f"{row['val_accuracy']:.2%}\nLR={row['learning_rate']:.1e}", 
                     (row['increment_order'], row['val_accuracy']), 
                     textcoords="offset points",
                     xytext=(0, 10), 
                     ha='center')
    
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'best_accuracy_vs_increment.png'), dpi=300)
    
    # Create heatmap of learning rate vs increment
    pivot_data = results_df.pivot_table(
        index='increment_order', 
        columns='learning_rate', 
        values='val_accuracy',
        aggfunc='mean'
    )
    
    plt.figure(figsize=(14, 10))
    heatmap = plt.imshow(pivot_data, cmap='viridis', aspect='auto')
    plt.colorbar(heatmap, label='Validation Accuracy')
    
    # Set labels
    plt.yticks(range(len(pivot_data.index)), 
               [f"{i}" if i > 0 else "original" for i in pivot_data.index])
    plt.xticks(range(len(pivot_data.columns)), 
               [f"{lr:.1e}" for lr in pivot_data.columns])
    
    plt.xlabel('Learning Rate', fontsize=14)
    plt.ylabel('Increment', fontsize=14)
    plt.title('Validation Accuracy Heatmap: Increment vs Learning Rate', fontsize=16)
    
    # Add text annotations
    for i in range(len(pivot_data.index)):
        for j in range(len(pivot_data.columns)):
            try:
                value = pivot_data.iloc[i, j]
                plt.text(j, i, f"{value:.2%}", ha='center', va='center', 
                         color='white' if value < pivot_data.values.mean() else 'black')
            except:
                pass
    
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'accuracy_heatmap.png'), dpi=300)
    
    # Create a comprehensive summary table as an image
    fig, ax = plt.subplots(figsize=(16, len(best_params