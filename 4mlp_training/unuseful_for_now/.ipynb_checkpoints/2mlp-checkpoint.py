import numpy as np
import h5py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

# Configuration
REAL_TRAIN_DATA_PATH = "/data/yyang409/bowen/imagenet_feature/swin_base/patch4_window7_224/image_features_w_label_train.npz"
REAL_VAL_DATA_PATH = "/data/yyang409/bowen/imagenet_feature/swin_base/patch4_window7_224/image_features_w_label_val.npz"  # Path to validation features
SYNTH_DATA_PATH = "/scratch/bowenxi/dit/data_gen/03_30/class_h5/full_dataset.h5"
NUM_CLASSES = 1000
SYNTH_RATIOS = [0, 1, 2, 5, 10]  # 0 = real data only
BATCH_SIZE = 512
EPOCHS = 50

def load_and_prepare_data():
    """Load and preprocess real/synthetic features"""
    # Load real training data
    with np.load(REAL_TRAIN_DATA_PATH) as data:
        real_train_features = data['features']
        real_train_labels = data['labels']
    
    # Load real validation data
    with np.load(REAL_VAL_DATA_PATH) as data:
        real_val_features = data['features']
        real_val_labels = data['labels']
    
    # Load synthetic data
    with h5py.File(SYNTH_DATA_PATH, 'r') as f:
        synth_features = f['samples'][:]
        synth_labels = f['labels'][:]

    # Add this after loading the synthetic data
    print(f"Synthetic data shape: {synth_features.shape}")
    print(f"Synthetic labels shape: {synth_labels.shape}")
    print(f"Synthetic label range: {synth_labels.min()} to {synth_labels.max()}")
    print(f"Real train label range: {real_train_labels.min()} to {real_train_labels.max()}")
    print(f"Sample synthetic labels: {synth_labels[:2000]}")
    print(f"Sample real labels: {real_train_labels[:2000]}")

    # Check for NaN values
    print(f"NaN in synthetic features: {np.isnan(synth_features).any()}")
    print(f"NaN in synthetic labels: {np.isnan(synth_labels).any()}")
    
    # Standardize features using training data statistics
    mean = np.mean(real_train_features, axis=0)
    std = np.std(real_train_features, axis=0) + 1e-8
    
    # Apply standardization to all datasets
    real_train_features = (real_train_features - mean) / std
    real_val_features = (real_val_features - mean) / std
    synth_features = (synth_features - mean) / std
    
    return (real_train_features, real_train_labels,
            real_val_features, real_val_labels,
            synth_features, synth_labels)

class AdaptiveMLP(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            
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
    optimizer = optim.AdamW(model.parameters(), lr=1e-3)
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
        print(f"Train Loss: {train_loss/len(train_loader):.4f} | Train Acc: {train_acc:.2%}")
        
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
        print(f"Val Loss: {val_loss/len(val_loader):.4f} | Val Acc: {val_acc:.2%}")
        
        scheduler.step(val_acc)
        
        # Save best model
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), f"best_model_ratio_{ratio}.pt")
            print(f"Best model saved with validation accuracy: {val_acc:.2%}")
    
    return best_acc

def run_experiment():
    # Load data
    (real_train_features, real_train_labels,
     real_val_features, real_val_labels,
     synth_features, synth_labels) = load_and_prepare_data()
    
    results = {}
    for ratio in SYNTH_RATIOS:
        print(f"\n=== Training with {ratio}x synthetic data ===")
        
        # Calculate synthetic samples to add
        num_real = len(real_train_features)
        num_synth = int(num_real * ratio)
        
        if ratio > 0:
            # Select synthetic samples
            synth_indices = np.random.choice(len(synth_features), num_synth, replace=(num_synth > len(synth_features)))
            X_synth = synth_features[synth_indices]
            y_synth = synth_labels[synth_indices]
            
            # Combine datasets
            X_train = np.concatenate([real_train_features, X_synth])
            y_train = np.concatenate([real_train_labels, y_synth])
        else:
            # Use real data only
            X_train = real_train_features
            y_train = real_train_labels
        
        # Train and evaluate
        val_acc = train_model(X_train, y_train, real_val_features, real_val_labels, ratio)
        results[ratio] = val_acc
        print(f"Ratio {ratio}x | Final Val Acc: {val_acc:.2%}")
    
    return results

def analyze_results(results):
    import matplotlib.pyplot as plt
    
    ratios = sorted(results.keys())
    accuracies = [results[r] for r in ratios]
    
    plt.figure(figsize=(10, 6))
    plt.plot(ratios, accuracies, 'bo-', markersize=8)
    plt.xlabel('Synthetic Data Ratio (x real data size)')
    plt.ylabel('Validation Accuracy')
    plt.title('ImageNet Classification Accuracy vs Synthetic Data Ratio')
    plt.grid(True)
    plt.savefig('synthetic_data_impact.png')
    plt.show()
    
    # Print numeric results for reference
    print("\nNumeric Results:")
    for r, acc in zip(ratios, accuracies):
        print(f"Ratio {r}x: {acc:.4f}")

if __name__ == "__main__":
    experiment_results = run_experiment()
    analyze_results(experiment_results)