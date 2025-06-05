import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import os
import glob
import matplotlib.pyplot as plt
import optuna
from sklearn.model_selection import StratifiedKFold
from ray import tune
from ray.tune.schedulers import ASHAScheduler
import ray
from functools import partial

# Configuration
REAL_TRAIN_DATA_PATH = "/data/yyang409/bowen/imagenet_feature/swin_base/patch4_window7_224/image_features_w_label_train.npz"
REAL_VAL_DATA_PATH = "/data/yyang409/bowen/imagenet_feature/swin_base/patch4_window7_224/image_features_w_label_val.npz"
SYNTH_DATA_DIR = "/scratch/bowenxi/dit/data_gen/B_4/final_data/"  # Directory containing the NPZ files
SYNTH_DATA_PATTERN = "imagenet_latents_*.npz"  # Pattern to match your NPZ files
NUM_CLASSES = 1000
SAMPLES_PER_CLASS = 1024  # Number of synthetic samples per class in each increment
MAX_MULTIPLES = 10  # Maximum number of multiples to test (10x)
BATCH_SIZE = 1024
EPOCHS = 50

# For hyperparameter optimization
NUM_TRIALS = 20  # Number of hyperparameter combinations to try
NUM_CV_FOLDS = 3  # Number of cross-validation folds
USE_RAY = True  # Set to True to use Ray Tune, False to use Optuna

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
            features = data['samples']
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
            # If not enough samples, take all available and repeat as needed
            idx_to_use = np.concatenate([class_indices] * (samples_to_take // len(class_indices) + 1))[:samples_to_take]
        
        # Add selected samples to our arrays
        selected_features.append(synth_features[idx_to_use])
        selected_labels.append(synth_labels[idx_to_use])
    
    # Combine all selected samples
    X_synth = np.vstack(selected_features)
    y_synth = np.concatenate(selected_labels)
    
    return X_synth, y_synth

class DynamicMLP(nn.Module):
    def __init__(self, input_dim, num_classes, hidden_layers, dropout_rates, use_batch_norm=True, activation='relu'):
        super().__init__()
        
        layers = []
        prev_dim = input_dim
        
        # Create hidden layers dynamically
        for i, (hidden_dim, dropout_rate) in enumerate(zip(hidden_layers, dropout_rates)):
            # Linear layer
            layers.append(nn.Linear(prev_dim, hidden_dim))
            
            # Batch normalization
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            
            # Activation function
            if activation == 'relu':
                layers.append(nn.ReLU())
            elif activation == 'leaky_relu':
                layers.append(nn.LeakyReLU(0.1))
            elif activation == 'elu':
                layers.append(nn.ELU())
            elif activation == 'gelu':
                layers.append(nn.GELU())
            
            # Dropout
            if dropout_rate > 0:
                layers.append(nn.Dropout(dropout_rate))
            
            prev_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(prev_dim, num_classes))
        
        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.model(x)

def train_val_model(model, train_loader, val_loader, config):
    """Train and evaluate model with given configuration"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # Get optimizer
    if config['optimizer'] == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])
    elif config['optimizer'] == 'adamw':
        optimizer = optim.AdamW(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])
    elif config['optimizer'] == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=config['lr'], momentum=0.9, weight_decay=config['weight_decay'])
    
    # Get learning rate scheduler
    if config['scheduler'] == 'reduce_on_plateau':
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=config['scheduler_patience'], factor=0.5)
    elif config['scheduler'] == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config['epochs'])
    elif config['scheduler'] == 'none':
        scheduler = None
    
    # Loss function
    criterion = nn.CrossEntropyLoss()
    
    # Training loop
    best_val_acc = 0
    early_stopping_counter = 0
    
    for epoch in range(config['epochs']):
        # Training
        model.train()
        train_loss = 0
        correct = 0
        total = 0
        
        for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}", leave=False):
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            
            # Gradient clipping
            if config.get('clip_grad', False):
                torch.nn.utils.clip_grad_norm_(model.parameters(), config['clip_value'])
                
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
        
        train_acc = correct / total
        
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
        
        if scheduler is not None and config['scheduler'] == 'reduce_on_plateau':
            scheduler.step(val_acc)
        elif scheduler is not None:
            scheduler.step()
        
        # Print stats periodically
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"Epoch {epoch+1}: Train Loss: {train_loss/len(train_loader):.4f} | Train Acc: {train_acc:.2%} | Val Acc: {val_acc:.2%}")
        
        # Check for new best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            early_stopping_counter = 0
        else:
            early_stopping_counter += 1
            
        # Early stopping
        if early_stopping_counter >= config['early_stopping_patience']:
            print(f"Early stopping triggered after epoch {epoch+1}")
            break
            
    return best_val_acc

def optuna_objective(trial, X_train, y_train, X_val, y_val, input_dim):
    """Objective function for Optuna optimization"""
    # Define hyperparameters to search
    config = {
        'hidden_layers': [
            trial.suggest_int(f'layer1', 256, 2048),
            trial.suggest_int(f'layer2', 256, 1024),
            trial.suggest_int(f'layer3', 128, 512),
        ],
        'dropout_rates': [
            trial.suggest_float(f'dropout1', 0.1, 0.7),
            trial.suggest_float(f'dropout2', 0.1, 0.5),
            trial.suggest_float(f'dropout3', 0.0, 0.3),
        ],
        'use_batch_norm': trial.suggest_categorical('use_batch_norm', [True, False]),
        'activation': trial.suggest_categorical('activation', ['relu', 'leaky_relu', 'elu', 'gelu']),
        'optimizer': trial.suggest_categorical('optimizer', ['adam', 'adamw', 'sgd']),
        'lr': trial.suggest_float('lr', 1e-4, 1e-2, log=True),
        'weight_decay': trial.suggest_float('weight_decay', 1e-5, 1e-3, log=True),
        'scheduler': trial.suggest_categorical('scheduler', ['reduce_on_plateau', 'cosine', 'none']),
        'scheduler_patience': trial.suggest_int('scheduler_patience', 2, 5),
        'batch_size': trial.suggest_categorical('batch_size', [512, 1024, 2048]),
        'early_stopping_patience': 7,
        'epochs': EPOCHS,
        'clip_grad': trial.suggest_categorical('clip_grad', [True, False]),
        'clip_value': trial.suggest_float('clip_value', 0.5, 5.0)
    }
    
    # Convert data to PyTorch datasets
    train_dataset = TensorDataset(
        torch.FloatTensor(X_train),
        torch.LongTensor(y_train)
    )
    val_dataset = TensorDataset(
        torch.FloatTensor(X_val),
        torch.LongTensor(y_val)
    )
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'])
    
    # Create model with the suggested hyperparameters
    model = DynamicMLP(
        input_dim=input_dim,
        num_classes=NUM_CLASSES,
        hidden_layers=config['hidden_layers'],
        dropout_rates=config['dropout_rates'],
        use_batch_norm=config['use_batch_norm'],
        activation=config['activation']
    )
    
    # Train and evaluate
    val_acc = train_val_model(model, train_loader, val_loader, config)
    
    return val_acc

def ray_objective(config, X_train, y_train, X_val, y_val, input_dim):
    """Objective function for Ray Tune optimization"""
    # Convert data to PyTorch datasets
    train_dataset = TensorDataset(
        torch.FloatTensor(X_train),
        torch.LongTensor(y_train)
    )
    val_dataset = TensorDataset(
        torch.FloatTensor(X_val),
        torch.LongTensor(y_val)
    )
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=int(config['batch_size']), shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=int(config['batch_size']))
    
    # Create model with the given hyperparameters
    model = DynamicMLP(
        input_dim=input_dim,
        num_classes=NUM_CLASSES,
        hidden_layers=[int(config['layer1']), int(config['layer2']), int(config['layer3'])],
        dropout_rates=[config['dropout1'], config['dropout2'], config['dropout3']],
        use_batch_norm=config['use_batch_norm'],
        activation=config['activation']
    )
    
    # Add early stopping patience to config
    config['early_stopping_patience'] = 7
    
    # Train and evaluate
    val_acc = train_val_model(model, train_loader, val_loader, config)
    
    # Report metrics to Ray Tune
    tune.report(val_accuracy=val_acc)

def optimize_hyperparameters(X_train, y_train, X_val, y_val):
    """Find optimal hyperparameters using either Optuna or Ray Tune"""
    input_dim = X_train.shape[1]
    
    if USE_RAY:
        # Ray Tune configuration
        ray.init(num_cpus=8, num_gpus=1)
        
        config = {
            'layer1': tune.randint(256, 2048),
            'layer2': tune.randint(256, 1024),
            'layer3': tune.randint(128, 512),
            'dropout1': tune.uniform(0.1, 0.7),
            'dropout2': tune.uniform(0.1, 0.5),
            'dropout3': tune.uniform(0.0, 0.3),
            'use_batch_norm': tune.choice([True, False]),
            'activation': tune.choice(['relu', 'leaky_relu', 'elu', 'gelu']),
            'optimizer': tune.choice(['adam', 'adamw', 'sgd']),
            'lr': tune.loguniform(1e-4, 1e-2),
            'weight_decay': tune.loguniform(1e-5, 1e-3),
            'scheduler': tune.choice(['reduce_on_plateau', 'cosine', 'none']),
            'scheduler_patience': tune.randint(2, 6),
            'batch_size': tune.choice([512, 1024, 2048]),
            'epochs': EPOCHS,
            'clip_grad': tune.choice([True, False]),
            'clip_value': tune.uniform(0.5, 5.0)
        }
        
        scheduler = ASHAScheduler(
            max_t=EPOCHS,
            grace_period=10,
            reduction_factor=2
        )
        
        # Create a partial function with fixed arguments
        objective_with_data = partial(
            ray_objective,
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            input_dim=input_dim
        )
        
        # Run hyperparameter search
        result = tune.run(
            objective_with_data,
            config=config,
            num_samples=NUM_TRIALS,
            scheduler=scheduler,
            resources_per_trial={"cpu": 2, "gpu": 0.5},
            progress_reporter=tune.CLIReporter(
                parameter_columns=["layer1", "layer2", "layer3", "optimizer", "lr", "activation"],
                metric_columns=["val_accuracy", "training_iteration"]
            )
        )
        
        # Get best trial and config
        best_trial = result.get_best_trial("val_accuracy", "max", "last")
        best_config = best_trial.config
        best_accuracy = best_trial.last_result["val_accuracy"]
        
        # Format the hidden layers and dropout rates for model creation
        best_config['hidden_layers'] = [int(best_config['layer1']), int(best_config['layer2']), int(best_config['layer3'])]
        best_config['dropout_rates'] = [best_config['dropout1'], best_config['dropout2'], best_config['dropout3']]
        
        ray.shutdown()
        
    else:
        # Optuna optimization
        study = optuna.create_study(direction="maximize")
        
        # Create a partial function with fixed arguments
        objective_with_data = partial(
            optuna_objective,
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            input_dim=input_dim
        )
        
        # Run optimization
        study.optimize(objective_with_data, n_trials=NUM_TRIALS)
        
        # Get best parameters
        best_config = study.best_params
        best_accuracy = study.best_value
        
        # Format the hidden layers and dropout rates for consistency
        best_config['hidden_layers'] = [best_config[f'layer{i+1}'] for i in range(3)]
        best_config['dropout_rates'] = [best_config[f'dropout{i+1}'] for i in range(3)]
    
    print("\n----- Best Hyperparameters -----")
    for key, value in best_config.items():
        if key not in ['hidden_layers', 'dropout_rates', 'layer1', 'layer2', 'layer3', 'dropout1', 'dropout2', 'dropout3']:
            print(f"{key}: {value}")
    print(f"Hidden layers: {best_config['hidden_layers']}")
    print(f"Dropout rates: {best_config['dropout_rates']}")
    print(f"Best validation accuracy: {best_accuracy:.4f}")
    
    return best_config, best_accuracy

def train_model_with_best_params(X_train, y_train, X_val, y_val, best_config, ratio):
    """Train the model with the best hyperparameters"""
    input_dim = X_train.shape[1]
    
    # Convert data to PyTorch datasets
    train_dataset = TensorDataset(
        torch.FloatTensor(X_train),
        torch.LongTensor(y_train)
    )
    val_dataset = TensorDataset(
        torch.FloatTensor(X_val),
        torch.LongTensor(y_val)
    )
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=int(best_config.get('batch_size', 1024)), shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=int(best_config.get('batch_size', 1024)))
    
    # Create model with best hyperparameters
    model = DynamicMLP(
        input_dim=input_dim,
        num_classes=NUM_CLASSES,
        hidden_layers=best_config['hidden_layers'],
        dropout_rates=best_config['dropout_rates'],
        use_batch_norm=best_config.get('use_batch_norm', True),
        activation=best_config.get('activation', 'relu')
    )
    
    # Set early stopping patience
    best_config['early_stopping_patience'] = 10
    best_config['epochs'] = EPOCHS
    
    # Train and evaluate
    print(f"\nTraining final model with best hyperparameters for ratio {ratio}x...")
    val_acc = train_val_model(model, train_loader, val_loader, best_config)
    
    # Save the model
    torch.save(model.state_dict(), f"best_model_automl_ratio_{ratio}x.pt")
    torch.save(best_config, f"best_config_automl_ratio_{ratio}x.pt")
    
    return val_acc

def run_experiment():
    # Load data
    (real_train_features, real_train_labels,
     real_val_features, real_val_labels,
     synth_features, synth_labels) = load_and_prepare_data()
    
    # Track results
    results = {}
    
    # First, train with real data only (0x synthetic)
    print("\n=== Optimizing hyperparameters with real data only (0x synthetic) ===")
    best_config, _ = optimize_hyperparameters(
        real_train_features, real_train_labels,
        real_val_features, real_val_labels
    )
    
    # Train final model with best hyperparameters
    val_acc = train_model_with_best_params(
        real_train_features, real_train_labels,
        real_val_features, real_val_labels,
        best_config, 0
    )
    
    results[0] = val_acc
    print(f"Real data only | Final Val Acc: {val_acc:.2%}")
    
    # Calculate the maximum ratio we can support with the available synthetic data
    available_samples_per_class = 10 * 1024  # Assuming 10 files x 1024 samples per class
    max_ratio = min(MAX_MULTIPLES, available_samples_per_class // SAMPLES_PER_CLASS)
    
    print(f"\nBased on available synthetic data, we can support up to {max_ratio}x ratios")
    
    # Then progressively add more synthetic data and optimize
    for ratio in range(1, max_ratio + 1):
        print(f"\n=== Training with {ratio}x synthetic data ===")
        
        # Prepare synthetic data for this ratio (SAMPLES_PER_CLASS * ratio per class)
        X_synth, y_synth = prepare_synthetic_data_by_ratio(synth_features, synth_labels, ratio)
        
        print(f"Using {len(X_synth)} synthetic samples ({len(X_synth)//NUM_CLASSES} per class)")
        
        # Combine real and synthetic data
        X_train = np.concatenate([real_train_features, X_synth])
        y_train = np.concatenate([real_train_labels, y_synth])
        
        # Optimize hyperparameters for this ratio if it's a milestone (1x, 5x, 10x)
        if ratio in [1, 5, max_ratio] or ratio % 3 == 0:
            print(f"Optimizing hyperparameters for ratio {ratio}x...")
            best_config, _ = optimize_hyperparameters(
                X_train, y_train,
                real_val_features, real_val_labels
            )
        else:
            # Use previous best config for intermediate ratios
            print(f"Using previously optimized hyperparameters for ratio {ratio}x...")
        
        # Train and evaluate with best hyperparameters
        val_acc = train_model_with_best_params(
            X_train, y_train,
            real_val_features, real_val_labels,
            best_config, ratio
        )
        
        results[ratio] = val_acc
        print(f"Ratio {ratio}x | Final Val Acc: {val_acc:.2%}")
    
    return results

def cross_validate_best_model(X_train, y_train, best_config, ratio, n_folds=5):
    """Perform cross-validation on the best model to assess its stability"""
    input_dim = X_train.shape[1]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Set up k-fold cross-validation
    kfold = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    
    fold_scores = []
    
    for fold, (train_idx, val_idx) in enumerate(kfold.split(X_train, y_train)):
        print(f"\nFold {fold+1}/{n_folds}")
        
        # Split data for this fold
        X_fold_train, X_fold_val = X_train[train_idx], X_train[val_idx]
        y_fold_train, y_fold_val = y_train[train_idx], y_train[val_idx]
        
        # Convert to PyTorch datasets
        train_dataset = TensorDataset(
            torch.FloatTensor(X_fold_train),
            torch.LongTensor(y_fold_train)
        )
        val_dataset = TensorDataset(
            torch.FloatTensor(X_fold_val),
            torch.LongTensor(y_fold_val)
        )
        
        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=int(best_config.get('batch_size', 1024)), shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=int(best_config.get('batch_size', 1024)))
        
        # Create model with best hyperparameters
        model = DynamicMLP(
            input_dim=input_dim,
            num_classes=NUM_CLASSES,
            hidden_layers=best_config['hidden_layers'],
            dropout_rates=best_config['dropout_rates'],
            use_batch_norm=best_config.get('use_batch_norm', True),
            activation=best_config.get('activation', 'relu')
        )
        
        # Train and evaluate
        fold_acc = train_val_model(model, train_loader, val_loader, best_config)
        fold_scores.append(fold_acc)
        
        print(f"Fold {fold+1} accuracy: {fold_acc:.4f}")
    
    mean_acc = np.mean(fold_scores)
    std_acc = np.std(fold_scores)
    
    print(f"\nCross-validation results for ratio {ratio}x:")
    print(f"Mean accuracy: {mean_acc:.4f} ± {std_acc:.4f}")
    
    return mean_acc, std_acc

def analyze_results(results, cv_results=None):
    """Analyze and visualize the results"""
    import matplotlib.pyplot as plt
    
    ratios = sorted(results.keys())
    accuracies = [results[r] for r in ratios]
    
    plt.figure(figsize=(12, 7))
    
    # Plot the accuracy vs. ratio curve
    plt.plot(ratios, accuracies, 'bo-', markersize=8, label='Test Accuracy')
    
    # Plot cross-validation results if available
    if cv_results is not None:
        cv_means = [cv_results[r]['mean'] for r in ratios if r in cv_results]
        cv_stds = [cv_results[r]['std'] for r in ratios if r in cv_results]
        cv_ratios = [r for r in ratios if r in cv_results]
        
        plt.errorbar(cv_ratios, cv_means, yerr=cv_stds, fmt='ro-', capsize=5, 
                     markersize=8, label='Cross-Validation Accuracy')
    
    plt.xlabel('Synthetic Data Ratio (x real data size per class)', fontsize=14)
    plt.ylabel('Validation Accuracy', fontsize=14)
    plt.title('AutoML ImageNet Classification Accuracy vs Synthetic Data Ratio', fontsize=16)
    plt.grid(True)
    plt.xticks(ratios)
    
    # Add text labels for each point
    for x, y in zip(ratios, accuracies):
        plt.annotate(f"{y:.2%}", 
                    (x, y), 
                    textcoords="offset points",
                    xytext=(0,10), 
                    ha='center')
    
    if cv_results is not None:
        plt.legend()
        
    plt.tight_layout