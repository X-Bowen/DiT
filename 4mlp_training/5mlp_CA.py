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

# Configuration
BASE_DATA_DIR = "/scratch/bowenxi/dit/neural_tangent_kernel/feature_swin_b/incremental_synthetic"
RESULTS_DIR = "incremental_results"
NUM_CLASSES = 1000
BATCH_SIZE = 1024
EPOCHS = 0
LEARNING_RATE = 1e-3
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
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            
            nn.Linear(1024, 768),
            nn.BatchNorm1d(768),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(768, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.1),
            
            nn.Linear(512, num_classes)
        )
        
    def forward(self, x):
        return self.layers(x)

def train_model(X_train, y_train, X_val, y_val, increment_name):
    """
    Train and evaluate MLP
    
    Args:
        X_train: Training features
        y_train: Training labels
        X_val: Validation features
        y_val: Validation labels
        increment_name: Name of the increment for saving model
    
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
    
    # Create loaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
    
    # Initialize model
    model = AdaptiveMLP(X_train.shape[1], NUM_CLASSES)
    model = model.cuda()
    
    # Training setup
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=3)
    
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
        
        for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            inputs, labels = inputs.cuda(), labels.cuda()
            
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
        
        print(f"Epoch {epoch+1}: Train Loss: {avg_train_loss:.4f} | Train Acc: {train_acc:.2%}")
        
        # Validation
        model.eval()
        val_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.cuda(), labels.cuda()
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
        
        print(f"Epoch {epoch+1}: Val Loss: {avg_val_loss:.4f} | Val Acc: {val_acc:.2%}")
        
        scheduler.step(val_acc)
        
        # Save best model
        if val_acc > best_acc:
            best_acc = val_acc
            best_epoch = epoch
            model_path = os.path.join(RESULTS_DIR, f"best_model_{increment_name}.pt")
            torch.save(model.state_dict(), model_path)
            print(f"Best model saved with validation accuracy: {val_acc:.2%}")
    
    training_time = time.time() - start_time
    print(f"Training completed in {training_time:.2f} seconds.")
    print(f"Best validation accuracy: {best_acc:.4f} at epoch {best_epoch+1}")
    
    return {
        'history': history,
        'best_acc': best_acc,
        'best_epoch': best_epoch,
        'training_time': training_time
    }

def plot_training_history(history, increment_name):
    """
    Plot training and validation metrics
    
    Args:
        history: Dictionary with training history
        increment_name: Name of the increment for saving plot
    """
    epochs = range(1, len(history['train_loss']) + 1)
    
    plt.figure(figsize=(12, 10))
    
    # Plot loss
    plt.subplot(2, 1, 1)
    plt.plot(epochs, history['train_loss'], 'b-', label='Training Loss')
    plt.plot(epochs, history['val_loss'], 'r-', label='Validation Loss')
    plt.title(f'Training and Validation Loss ({increment_name})')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    # Plot accuracy
    plt.subplot(2, 1, 2)
    plt.plot(epochs, history['train_acc'], 'b-', label='Training Accuracy')
    plt.plot(epochs, history['val_acc'], 'r-', label='Validation Accuracy')
    plt.title(f'Training and Validation Accuracy ({increment_name})')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, f'training_history_{increment_name}.png'), dpi=300)
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

def run_incremental_experiment():
    """
    Run experiment for each incremental dataset
    
    Returns:
        DataFrame with results
    """
    # Find all increment files
    train_files = sorted(glob.glob(os.path.join(BASE_DATA_DIR, "*_train_tangent.npz")))
    increment_names = [os.path.basename(f).replace("_train_tangent.npz", "") for f in train_files]
    
    print(f"Found {len(increment_names)} increments: {increment_names}")
    
    # Store results
    results = []
    
    # Run experiment for each increment
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
        
        # Train model
        training_results = train_model(X_train_std, y_train, X_val_std, y_val, increment_name)
        
        # Plot history
        plot_training_history(training_results['history'], increment_name)
        
        # Collect statistics
        increment_stats = get_increment_size(increment_name)
        
        # Store results
        results.append({
            'increment': increment_name,
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
    sorted_results = results_df.sort_values('increment')
    print("\nIncremental Results:")
    print("=" * 100)
    print(sorted_results.to_string(index=False))
    print("=" * 100)
    
    return results_df

def plot_final_results(results_df):
    """
    Create summary plots for all increments
    
    Args:
        results_df: DataFrame with results
    """
    # Process the increment names to get proper ordering
    results_df['increment_order'] = results_df['increment'].apply(
        lambda x: 0 if x == 'original' else int(x.split('_')[1])
    )
    sorted_df = results_df.sort_values('increment_order')
    
    # Plot validation accuracy vs increment
    plt.figure(figsize=(14, 8))
    plt.plot(sorted_df['increment_order'], sorted_df['val_accuracy'], 'bo-', markersize=8)
    plt.xlabel('Increment Number (0=original)', fontsize=14)
    plt.ylabel('Validation Accuracy', fontsize=14)
    plt.title('ImageNet Classification Accuracy vs Increment Level', fontsize=16)
    plt.grid(True)
    plt.xticks(sorted_df['increment_order'])
    
    # Add text labels
    for i, row in sorted_df.iterrows():
        plt.annotate(f"{row['val_accuracy']:.2%}", 
                     (row['increment_order'], row['val_accuracy']), 
                     textcoords="offset points",
                     xytext=(0, 10), 
                     ha='center')
    
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'accuracy_vs_increment.png'), dpi=300)
    
    # Plot validation accuracy vs total samples
    plt.figure(figsize=(14, 8))
    plt.plot(sorted_df['train_samples'], sorted_df['val_accuracy'], 'ro-', markersize=8)
    plt.xlabel('Number of Training Samples', fontsize=14)
    plt.ylabel('Validation Accuracy', fontsize=14)
    plt.title('ImageNet Classification Accuracy vs Training Data Size', fontsize=16)
    plt.grid(True)
    plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
    
    # Add text labels
    for i, row in sorted_df.iterrows():
        plt.annotate(f"{row['increment']}: {row['val_accuracy']:.2%}", 
                     (row['train_samples'], row['val_accuracy']), 
                     textcoords="offset points",
                     xytext=(5, 0), 
                     ha='left')
    
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'accuracy_vs_samples.png'), dpi=300)
    
    # Create a comprehensive summary table as an image
    fig, ax = plt.subplots(figsize=(16, len(sorted_df) * 0.8))
    ax.axis('tight')
    ax.axis('off')
    
    # Format the data for the table
    table_data = [
        ['Increment', 'Train Samples', 'Synthetic per Class', 'Val Accuracy', 'Training Time (s)']
    ]
    
    for i, row in sorted_df.iterrows():
        increment = row['increment']
        samples = f"{row['train_samples']:,}"
        synthetic = f"{row['synthetic_samples_per_class']:,}"
        accuracy = f"{row['val_accuracy']:.4f} ({row['val_accuracy']:.2%})"
        train_time = f"{row['training_time']:.1f}"
        
        table_data.append([increment, samples, synthetic, accuracy, train_time])
    
    table = ax.table(cellText=table_data, loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1, 1.5)
    
    plt.savefig(os.path.join(RESULTS_DIR, 'results_summary_table.png'), dpi=300, bbox_inches='tight')
    
    return sorted_df

if __name__ == "__main__":
    print(f"Starting incremental tangent feature experiment at {datetime.now()}")
    print(f"Results will be saved to {RESULTS_DIR}")
    
    # Run the experiment
    results = run_incremental_experiment()
    
    # Plot summary results
    final_results = plot_final_results(results)
    
    print(f"\nExperiment completed at {datetime.now()}")
    print(f"All results saved to {RESULTS_DIR}")