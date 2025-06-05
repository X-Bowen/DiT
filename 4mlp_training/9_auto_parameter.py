import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import os
import glob
import optuna
from torch.cuda.amp import autocast, GradScaler
import matplotlib.pyplot as plt

# Configuration
REAL_TRAIN_DATA_PATH = "/scratch/bowenxi/dit/neural_tangent_kernel/feature_swin_b/train_tanent.npz"
REAL_VAL_DATA_PATH = "/scratch/bowenxi/dit/neural_tangent_kernel/feature_swin_b/val_tanent.npz"
#SYNTH_DATA_DIR = "/scratch/bowenxi/dit/data_gen/B_4/final_data/"
SYNTH_DATA_DIR = "/scratch/bowenxi/dit/data_gen/b4l8/"

SYNTH_DATA_PATTERN = "imagenet_latents_*.npz"
NUM_CLASSES = 1000
SAMPLES_PER_CLASS = 1024
MAX_MULTIPLES = 10
BATCH_SIZE = 1024  # Default, will be optimized
EPOCHS = 50
OPTUNA_TRIALS = 20  # Number of hyperparameter search trials per ratio

def load_and_prepare_data():
    """Load and preprocess data (same as original)"""
    # [Keep the original implementation here]
    # ... (same code as provided) ...

class DynamicMLP(nn.Module):
    def __init__(self, input_dim, num_classes, hidden_dims, dropouts):
        super().__init__()
        layers = []
        for i, (h_dim, dropout) in enumerate(zip(hidden_dims, dropouts)):
            layers.append(nn.Linear(input_dim if i == 0 else hidden_dims[i-1], h_dim))
            layers.append(nn.BatchNorm1d(h_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(hidden_dims[-1], num_classes))
        self.layers = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.layers(x)

def train_model_with_params(X_train, y_train, X_val, y_val, params, ratio):
    """Train model with optimized hyperparameters"""
    # Convert to PyTorch datasets
    train_dataset = TensorDataset(torch.FloatTensor(X_train), torch.LongTensor(y_train))
    val_dataset = TensorDataset(torch.FloatTensor(X_val), torch.LongTensor(y_val))
    
    # Create loaders with optimized batch size
    train_loader = DataLoader(train_dataset, batch_size=params['batch_size'], shuffle=True, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=params['batch_size'], pin_memory=True)
    
    # Initialize model with optimized architecture
    model = DynamicMLP(
        input_dim=X_train.shape[1],
        num_classes=NUM_CLASSES,
        hidden_dims=params['hidden_dims'],
        dropouts=params['dropouts']
    ).cuda()
    
    # Optimizer and scheduler
    optimizer = getattr(optim, params['optimizer'])(
        model.parameters(),
        lr=params['lr'],
        weight_decay=params['weight_decay']
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=2)
    
    # Training setup
    criterion = nn.CrossEntropyLoss()
    scaler = GradScaler()
    best_acc = 0
    patience = 5
    no_improve = 0
    
    for epoch in range(params['max_epochs']):
        # Training
        model.train()
        train_loss = 0
        correct = 0
        total = 0
        
        for inputs, labels in tqdm(train_loader, desc=f"Ratio {ratio}x Epoch {epoch+1}"):
            inputs, labels = inputs.cuda(), labels.cuda()
            
            optimizer.zero_grad()
            with autocast():
                outputs = model(inputs)
                loss = criterion(outputs, labels)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
        
        # Validation
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.cuda(), labels.cuda()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()
        
        val_acc = val_correct / val_total
        scheduler.step(val_acc)
        
        # Early stopping
        if val_acc > best_acc:
            best_acc = val_acc
            no_improve = 0
            torch.save(model.state_dict(), f"best_model_ratio_{ratio}x.pt")
        else:
            no_improve += 1
            
        if no_improve >= patience:
            break
    
    return best_acc

def objective(trial, X_train, y_train, X_val, y_val, ratio):
    """Optuna optimization objective"""
    params = {
        'lr': trial.suggest_float('lr', 1e-5, 1e-3, log=True),
        'weight_decay': trial.suggest_float('weight_decay', 1e-6, 1e-3),
        'optimizer': trial.suggest_categorical('optimizer', ['AdamW', 'Adam', 'SGD']),
        'batch_size': trial.suggest_categorical('batch_size', [512, 1024, 2048]),
        'max_epochs': trial.suggest_int('max_epochs', 30, 60),
        'hidden_dims': [
            trial.suggest_int('hidden_dim1', 512, 1536, step=256),
            trial.suggest_int('hidden_dim2', 384, 1024, step=128)
        ],
        'dropouts': [
            trial.suggest_float('dropout1', 0.1, 0.5),
            trial.suggest_float('dropout2', 0.0, 0.3)
        ]
    }
    
    try:
        acc = train_model_with_params(X_train, y_train, X_val, y_val, params, ratio)
    except RuntimeError as e:  # Handle CUDA OOM
        print(f"OOM error: {str(e)}")
        acc = 0.0
    return acc

def run_experiment():
    # Load data once
    (real_train_features, real_train_labels,
     real_val_features, real_val_labels,
     synth_features, synth_labels) = load_and_prepare_data()
    
    results = {}
    
    # Baseline with real data only
    print("\n=== Optimizing Real Data Only ===")
    study = optuna.create_study(direction='maximize')
    study.optimize(lambda trial: objective(trial, real_train_features, real_train_labels,
                                         real_val_features, real_val_labels, 0), 
                 n_trials=OPTUNA_TRIALS)
    results[0] = study.best_value
    print(f"Real data only | Best Val Acc: {study.best_value:.2%}")
    
    # Synthetic data ratios
    available_samples = 10 * 1024  # Update based on your data
    max_ratio = min(MAX_MULTIPLES, available_samples // SAMPLES_PER_CLASS)
    
    for ratio in range(1, max_ratio + 1):
        print(f"\n=== Optimizing {ratio}x Synthetic Data ===")
        
        # Prepare synthetic data
        X_synth, y_synth = prepare_synthetic_data_by_ratio(synth_features, synth_labels, ratio)
        X_train = np.concatenate([real_train_features, X_synth])
        y_train = np.concatenate([real_train_labels, y_synth])
        
        # Hyperparameter search
        study = optuna.create_study(direction='maximize')
        study.optimize(lambda trial: objective(trial, X_train, y_train,
                                              real_val_features, real_val_labels, ratio), 
                      n_trials=OPTUNA_TRIALS)
        
        # Store best result
        results[ratio] = study.best_value
        print(f"Ratio {ratio}x | Best Val Acc: {study.best_value:.2%}")
        
        # Save best params
        with open(f"best_params_ratio_{ratio}x.txt", "w") as f:
            f.write(str(study.best_params))
    
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