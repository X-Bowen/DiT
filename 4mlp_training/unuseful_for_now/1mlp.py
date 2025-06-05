import numpy as np
import h5py
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

# Configuration
REAL_DATA_PATH = "/data/yyang409/bowen/imagenet_feature/swin_base/patch4_window7_224/image_features_w_label_train.npz"
SYNTH_DATA_PATH = "/scratch/bowenxi/dit/data_gen/04_31/1/full_dataset.h5"
NUM_CLASSES = 1000
SYNTH_RATIOS = [0]  # [0, 1, 2, 5, 10]  # 0 = real data only
BATCH_SIZE = 512
EPOCHS = 50

def load_and_prepare_data():
    """Load and preprocess real/synthetic features"""
    # Load real data
    with np.load(REAL_DATA_PATH) as data:
        real_features = data['features']
        real_labels = data['labels']
    
    # Load synthetic data
    with h5py.File(SYNTH_DATA_PATH, 'r') as f:
        synth_features = f['samples'][:]
        synth_labels = f['labels'][:]
    
    # Standardize features
    mean = np.mean(real_features, axis=0)
    std = np.std(real_features, axis=0) + 1e-8
    real_features = (real_features - mean) / std
    synth_features = (synth_features - mean) / std
    
    # Split real data for validation
    X_real_train, X_real_val, y_real_train, y_real_val = train_test_split(
        real_features, real_labels, test_size=0.2, stratify=real_labels
    )
    
    return (X_real_train, y_real_train,
            X_real_val, y_real_val,
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
        for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            inputs, labels = inputs.cuda(), labels.cuda()
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        # Validation
        model.eval()
        val_acc = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.cuda(), labels.cuda()
                outputs = model(inputs)
                val_acc += (outputs.argmax(1) == labels).sum().item()
        
        val_acc /= len(val_dataset)
        scheduler.step(val_acc)
        
        # Save best model
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), f"best_model_ratio_{ratio}.pt")
    
    return best_acc


def run_experiment():
    # Load data
    (X_real_train, y_real_train,
     X_real_val, y_real_val,
     synth_features, synth_labels) = load_and_prepare_data()
    
    results = {}
    for ratio in SYNTH_RATIOS:
        print(f"\n=== Training with {ratio}x synthetic data ===")
        
        # Calculate synthetic samples to add
        num_real = len(X_real_train)
        num_synth = int(num_real * ratio)
        
        # Select synthetic samples
        synth_indices = np.random.choice(len(synth_features), num_synth, replace=False)
        X_synth = synth_features[synth_indices]
        y_synth = synth_labels[synth_indices]
        
        # Combine datasets
        X_combined = np.concatenate([X_real_train, X_synth])
        y_combined = np.concatenate([y_real_train, y_synth])
        
        # Train and evaluate
        val_acc = train_model(X_combined, y_combined, X_real_val, y_real_val, ratio)
        results[ratio] = val_acc
        print(f"Ratio {ratio}x | Val Acc: {val_acc:.2%}")
    
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

if __name__ == "__main__":
    experiment_results = run_experiment()
    analyze_results(experiment_results)
