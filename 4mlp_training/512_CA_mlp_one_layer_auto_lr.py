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
import multiprocessing
from functools import partial
import torch.multiprocessing as mp

# Configuration
BASE_DATA_DIR = "/scratch/bowenxi/dit/neural_tangent_kernel/feature_swin_b/auto_ML_CA"
RESULTS_DIR = "incremental_results"
NUM_CLASSES = 1000
BATCH_SIZE = 1024
EPOCHS = 50
LEARNING_RATES = [1e-4, 5e-4, 1e-3, 5e-3, 1e-2]  # Multiple learning rates to test
WEIGHT_DECAY = 1e-4

# Make sure results directory exists
os.makedirs(RESULTS_DIR, exist_ok=True)

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


def train_model(X_train, y_train, X_val, y_val, increment_name, learning_rate, gpu_id):
    """
    Train and evaluate MLP with specific learning rate on specified GPU
    
    Args:
        X_train: Training features
        y_train: Training labels
        X_val: Validation features
        y_val: Validation labels
        increment_name: Name of the increment for saving model
        learning_rate: Learning rate to use for training
        gpu_id: GPU ID to use for training
    
    Returns:
        Dictionary with training history and best validation accuracy
    """
    # Set device
    device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device} with LR={learning_rate:.1e}")
    
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
    model = AdaptiveMLP(X_train.shape[1], NUM_CLASSES)
    model = model.to(device)
    
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
        # Training
        model.train()
        train_loss = 0
        correct = 0
        total = 0
        
        for inputs, labels in tqdm(train_loader, desc=f"GPU {gpu_id}, LR {learning_rate:.1e}, Epoch {epoch+1}"):
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
        
        train_acc = correct / total
        avg_train_loss = train_loss / len(train_loader)
        history['train_loss'].append(avg_train_loss)
        history['train_acc'].append(train_acc)
        
        print(f"GPU {gpu_id}, LR {learning_rate:.1e}, Epoch {epoch+1}: Train Loss: {avg_train_loss:.4f} | Train Acc: {train_acc:.2%}")
        
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
        
        val_acc = correct / total
        avg_val_loss = val_loss / len(val_loader)
        history['val_loss'].append(avg_val_loss)
        history['val_acc'].append(val_acc)
        
        print(f"GPU {gpu_id}, LR {learning_rate:.1e}, Epoch {epoch+1}: Val Loss: {avg_val_loss:.4f} | Val Acc: {val_acc:.2%}")
        
        # Step the scheduler
        scheduler.step()
        
        # Save best model
        if val_acc > best_acc:
            best_acc = val_acc
            best_epoch = epoch
            model_path = os.path.join(RESULTS_DIR, f"best_model_{increment_name}_lr_{learning_rate:.1e}.pt")
            torch.save(model.state_dict(), model_path)
            print(f"GPU {gpu_id}: Best model saved with validation accuracy: {val_acc:.2%}")
    
    training_time = time.time() - start_time
    print(f"GPU {gpu_id}: Training completed in {training_time:.2f} seconds.")
    print(f"GPU {gpu_id}: Best validation accuracy: {best_acc:.4f} at epoch {best_epoch+1}")
    
    # Plot history
    plot_training_history(history, increment_name, learning_rate, gpu_id)
    
    return {
        'increment': increment_name,
        'learning_rate': learning_rate,
        'train_samples': len(X_train),
        'val_accuracy': best_acc,
        'best_epoch': best_epoch + 1,
        'training_time': training_time,
        'history': history,
    }

def plot_training_history(history, increment_name, learning_rate, gpu_id):
    """
    Plot training and validation metrics
    
    Args:
        history: Dictionary with training history
        increment_name: Name of the increment for saving plot
        learning_rate: Learning rate used for training
        gpu_id: GPU ID used for training
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

def process_increment(increment_name, available_gpus):
    """
    Process a single increment with multiple learning rates in parallel
    
    Args:
        increment_name: Name of the increment
        available_gpus: List of available GPU IDs
    
    Returns:
        List of results for each learning rate
    """
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
    
    # Create a pool of worker processes for parallel training
    results = []
    
    # Parallel training with multiprocessing
    with mp.Pool(processes=len(available_gpus)) as pool:
        # Create tasks for each learning rate
        tasks = []
        for i, lr in enumerate(LEARNING_RATES):
            gpu_id = available_gpus[i % len(available_gpus)]
            tasks.append((X_train_std, y_train, X_val_std, y_val, increment_name, lr, gpu_id))
        
        # Run tasks in parallel
        # We need to use starmap because we have multiple arguments
        worker_results = pool.starmap(train_model_wrapper, tasks)
        
        # Process results
        for result in worker_results:
            result.update({
                'synthetic_classes': increment_stats['synthetic_classes'],
                'synthetic_samples_per_class': increment_stats['synthetic_samples_per_class'],
                'total_synthetic': increment_stats['total_synthetic']
            })
            results.append(result)
    
    # Free up memory
    del X_train_std, X_val_std, y_train, y_val
    torch.cuda.empty_cache()
    
    return results

def train_model_wrapper(X_train, y_train, X_val, y_val, increment_name, learning_rate, gpu_id):
    """
    Wrapper function for train_model to handle multiprocessing
    """
    try:
        torch.cuda.set_device(gpu_id)
        return train_model(X_train, y_train, X_val, y_val, increment_name, learning_rate, gpu_id)
    except Exception as e:
        print(f"Error in training on GPU {gpu_id} with LR {learning_rate}: {str(e)}")
        import traceback
        traceback.print_exc()
        return {
            'increment': increment_name,
            'learning_rate': learning_rate,
            'train_samples': len(X_train),
            'val_accuracy': 0,
            'best_epoch': 0,
            'training_time': 0,
            'error': str(e)
        }

def run_incremental_experiment():
    """
    Run experiment for each incremental dataset with different learning rates in parallel
    
    Returns:
        DataFrame with results
    """
    # Find all increment files
    train_files = sorted(glob.glob(os.path.join(BASE_DATA_DIR, "*_train_tangent.npz")))
    increment_names = [os.path.basename(f).replace("_train_tangent.npz", "") for f in train_files]
    
    print(f"Found {len(increment_names)} increments: {increment_names}")
    
    # Get available GPUs
    available_gpus = list(range(torch.cuda.device_count()))
    print(f"Available GPUs: {available_gpus}")
    
    if not available_gpus:
        print("No GPUs available, exiting.")
        return
    
    # Store results
    all_results = []
    
    # Process each increment sequentially, but learning rates in parallel
    for increment_name in increment_names:
        increment_results = process_increment(increment_name, available_gpus)
        all_results.extend(increment_results)
    
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
    fig, ax = plt.subplots(figsize=(16, len(best_params) * 0.8))
    ax.axis('tight')
    ax.axis('off')
    
    # Format the data for the table
    table_data = [
        ['Increment', 'Best LR', 'Val Accuracy', 'Train Samples', 'Best Epoch', 'Training Time (s)']
    ]
    
    for i, row in best_params.iterrows():
        increment = row['increment']
        lr = f"{row['learning_rate']:.1e}"
        accuracy = f"{row['val_accuracy']:.4f} ({row['val_accuracy']:.2%})"
        samples = f"{row['train_samples']:,}"
        epoch = f"{row['best_epoch']}"
        train_time = f"{row['training_time']:.1f}"
        
        table_data.append([increment, lr, accuracy, samples, epoch, train_time])
    
    table = ax.table(cellText=table_data, loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1, 1.5)
    
    plt.savefig(os.path.join(RESULTS_DIR, 'best_lr_summary_table.png'), dpi=300, bbox_inches='tight')
    
    return best_params

if __name__ == "__main__":
    # Set environment variables for multiprocessing
    os.environ['PYTHONUNBUFFERED'] = '1'  # Ensures output is unbuffered
    mp.set_start_method('spawn', force=True)  # More stable for CUDA in multiprocessing
    
    print(f"Starting parallel incremental tangent feature experiment at {datetime.now()}")
    print(f"Testing learning rates: {LEARNING_RATES}")
    print(f"Number of available GPUs: {torch.cuda.device_count()}")
    print(f"Results will be saved to {RESULTS_DIR}")
    
    # Run the experiment
    results = run_incremental_experiment()
    
    # Plot learning rate comparison
    best_params = plot_learning_rate_comparison(results)
    
    print(f"\nExperiment completed at {datetime.now()}")
    print(f"Best learning rates for each increment:")
    print(best_params[['increment', 'learning_rate', 'val_accuracy']].to_string(index=False))
    print(f"\nAll results saved to {RESULTS_DIR}")