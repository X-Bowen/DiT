import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import os
import glob

# Configuration
#REAL_TRAIN_DATA_PATH = "/data/yyang409/bowen/imagenet_feature/swin_base/patch4_window7_224/image_features_w_label_train.npz"
REAL_VAL_DATA_PATH = "/data/yyang409/bowen/imagenet_feature/swin_base/patch4_window7_224/image_features_w_label_val.npz"

REAL_TRAIN_DATA_PATH = "/scratch/bowenxi/dit/neural_tangent_kernel/feature_swin_b/train_tanent.npz"
#REAL_VAL_DATA_PATH = "/scratch/bowenxi/dit/neural_tangent_kernel/feature_swin_b/val_tanent.npz"


#SYNTH_DATA_DIR = "/scratch/bowenxi/dit/data_gen/L_8/final_data/"  # Directory containing the NPZ files

#SYNTH_DATA_DIR = "/scratch/bowenxi/dit/data_gen/B_4/final_npz_features_labels/"
#SYNTH_DATA_PATTERN = "renamed_imagenet_latents_*.npz"  # Pattern to match your NPZ files

SYNTH_DATA_DIR = "/scratch/bowenxi/dit/neural_tangent_kernel/feature_swin_b/"
SYNTH_DATA_PATTERN = "train_tanent.npz"
NUM_CLASSES = 1000
SAMPLES_PER_CLASS = 1024  # Number of synthetic samples per class in each increment
MAX_MULTIPLES = 10  # Maximum number of multiples to test (10x)
FILE_NUMBERS = 0
BATCH_SIZE = 1024
EPOCHS = 50

def load_and_prepare_data():
    """Load and preprocess real/synthetic features"""
    # Load real training data
    print("Loading real training data...")
    with np.load(REAL_TRAIN_DATA_PATH) as data:
        real_train_features = data['features']
        real_train_labels = data['labels']
    
    # Load real validation data
    print("Loading real validation data...")
    with np.load(REAL_VAL_DATA_PATH) as data:
        real_val_features = data['features']
        real_val_labels = data['labels']
    
    # Load synthetic data from multiple NPZ files
    print("Loading synthetic data from multiple NPZ files...")
    synth_features_list = []
    synth_labels_list = []
    
    # Get list of all NPZ files matching the pattern
    npz_files = sorted(glob.glob(os.path.join(SYNTH_DATA_DIR, SYNTH_DATA_PATTERN)))
    print(f"Found {len(npz_files)} synthetic data files: {[os.path.basename(f) for f in npz_files]}")
    
    # Load each NPZ file and append to our lists
    for npz_file in npz_files:
        print(f"Loading {os.path.basename(npz_file)}...")
        with np.load(npz_file) as data:
            # Assuming each NPZ file has 'features' and 'labels' keys - adjust as needed
            features = data['features']
            labels = data['labels']
            
            synth_features_list.append(features)
            synth_labels_list.append(labels)
    
    # Combine all synthetic data
    synth_features = np.vstack(synth_features_list)
    synth_labels = np.concatenate(synth_labels_list)
    
    # Check for NaN values in synthetic data
    if np.isnan(synth_features).any():
        print(f"WARNING: Found {np.isnan(synth_features).sum()} NaN values in synthetic features")
        print("Replacing NaN values with 0...")
        synth_features = np.nan_to_num(synth_features, nan=0.0)
    
    # Print dataset statistics
    print(f"Real training data: {real_train_features.shape[0]} samples")
    print(f"Real validation data: {real_val_features.shape[0]} samples")
    print(f"Synthetic data: {synth_features.shape[0]} samples")
    
    # Verify label distributions
    real_class_counts = np.bincount(real_train_labels, minlength=NUM_CLASSES)
    synth_class_counts = np.bincount(synth_labels, minlength=NUM_CLASSES)
    
    print(f"Real data class distribution: min={real_class_counts.min()}, max={real_class_counts.max()}, avg={real_class_counts.mean():.1f}")
    print(f"Synthetic data class distribution: min={synth_class_counts.min()}, max={synth_class_counts.max()}, avg={synth_class_counts.mean():.1f}")
    
    # Standardize features using training data statistics
    print("Standardizing features...")
    mean = np.mean(real_train_features, axis=0)
    std = np.std(real_train_features, axis=0) + 1e-8
    
    # Apply standardization to all datasets
    real_train_features = (real_train_features - mean) / std
    real_val_features = (real_val_features - mean) / std
    synth_features = (synth_features - mean) / std
    
    return (real_train_features, real_train_labels,
            real_val_features, real_val_labels,
            synth_features, synth_labels)

def prepare_synthetic_data_by_ratio(synth_features, synth_labels, ratio):
    """Prepare synthetic data for a given ratio - take samples per class based on ratio"""
    samples_to_take = SAMPLES_PER_CLASS * ratio
    
    # Initialize arrays to store selected data
    selected_features = []
    selected_labels = []
    
    # For each class, take the appropriate number of samples
    for class_idx in range(NUM_CLASSES):
        # Find indices for this class
        class_indices = np.where(synth_labels == class_idx)[0]
        
        # If we have enough samples, take the first N
        if len(class_indices) >= samples_to_take:
            idx_to_use = class_indices[:samples_to_take]
        else:
            # Raise an error if there are not enough samples for the class.
            raise ValueError(f"Not enough samples for class {class_idx}: required {samples_to_take} but found {len(class_indices)}.")
            # If not enough samples, take all available and repeat as needed
            # idx_to_use = np.concatenate([class_indices] * (samples_to_take // len(class_indices) + 1))[:samples_to_take]
        
        # Add selected samples to our arrays
        selected_features.append(synth_features[idx_to_use])
        selected_labels.append(synth_labels[idx_to_use])
    
    # Combine all selected samples
    X_synth = np.vstack(selected_features)
    y_synth = np.concatenate(selected_labels)
    
    return X_synth, y_synth

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

def train_model(X_train, y_train, X_val, y_val, ratio):
    """Train and evaluate MLP"""
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
    # optimizer = optim.AdamW(model.parameters(), lr=2e-3)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=3)
    
    best_acc = 0
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
        print(f"Epoch {epoch+1}: Train Loss: {train_loss/len(train_loader):.4f} | Train Acc: {train_acc:.2%}")
        
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
        print(f"Epoch {epoch+1}: Val Loss: {val_loss/len(val_loader):.4f} | Val Acc: {val_acc:.2%}")
        
        scheduler.step(val_acc)
        
        # Save best model
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), f"best_model_ratio_{ratio}x.pt")
            print(f"Best model saved with validation accuracy: {val_acc:.2%}")
    
    return best_acc

def run_experiment():
    # Load data
    (real_train_features, real_train_labels,
     real_val_features, real_val_labels,
     synth_features, synth_labels) = load_and_prepare_data()
    
    # Track results
    results = {}
    
    # First, train with real data only (0x synthetic)
    print("\n=== Training with real data only (0x synthetic) ===")
    val_acc = train_model(real_train_features, real_train_labels, 
                          real_val_features, real_val_labels, 0)
    results[0] = val_acc
    print(f"Real data only | Final Val Acc: {val_acc:.2%}")
    
    # Calculate the maximum ratio we can support with the available synthetic data
    # We have 3 files, each with 1024 samples per class

    available_samples_per_class = FILE_NUMBERS * 1024  # 3 files x 1024 samples per class
    max_ratio = min(MAX_MULTIPLES, available_samples_per_class // SAMPLES_PER_CLASS)
    
    print(f"\nBased on available synthetic data, we can support up to {max_ratio}x ratios")
    
    # Then progressively add more synthetic data
    for ratio in range(1, max_ratio + 1):
        print(f"\n=== Training with {ratio}x synthetic data ===")
        
        # Prepare synthetic data for this ratio (SAMPLES_PER_CLASS * ratio per class)
        X_synth, y_synth = prepare_synthetic_data_by_ratio(synth_features, synth_labels, ratio)
        
        print(f"Using {len(X_synth)} synthetic samples ({len(X_synth)//NUM_CLASSES} per class)")
        
        # Combine real and synthetic data
        X_train = np.concatenate([real_train_features, X_synth])
        y_train = np.concatenate([real_train_labels, y_synth])
        
        # Train and evaluate
        val_acc = train_model(X_train, y_train, real_val_features, real_val_labels, ratio)
        results[ratio] = val_acc
        print(f"Ratio {ratio}x | Final Val Acc: {val_acc:.2%}")
    
    return results

def analyze_results(results):
    import matplotlib.pyplot as plt
    
    ratios = sorted(results.keys())
    accuracies = [results[r] for r in ratios]
    
    plt.figure(figsize=(12, 7))
    plt.plot(ratios, accuracies, 'bo-', markersize=8)
    plt.xlabel('Synthetic Data Ratio (x real data size per class)', fontsize=14)
    plt.ylabel('Validation Accuracy', fontsize=14)
    plt.title('ImageNet Classification Accuracy vs Synthetic Data Ratio', fontsize=16)
    plt.grid(True)
    plt.xticks(ratios)
    
    # Add text labels for each point
    for x, y in zip(ratios, accuracies):
        plt.annotate(f"{y:.2%}", 
                    (x, y), 
                    textcoords="offset points",
                    xytext=(0,10), 
                    ha='center')
    
    plt.tight_layout()
    plt.savefig('synthetic_data_impact.png', dpi=300)
    plt.show()
    
    # Print numeric results for reference
    print("\nNumeric Results:")
    print("Ratio | Validation Accuracy")
    print("-" * 30)
    for r, acc in zip(ratios, accuracies):
        print(f"{r}x    | {acc:.4f} ({acc:.2%})")

if __name__ == "__main__":
    experiment_results = run_experiment()
    analyze_results(experiment_results)