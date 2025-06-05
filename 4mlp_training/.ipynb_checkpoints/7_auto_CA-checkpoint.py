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
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif
import optuna
from optuna.trial import Trial
from functools import partial

# Configuration
BASE_DATA_DIR = "/scratch/bowenxi/dit/neural_tangent_kernel/feature_swin_b/incremental_synthetic"
# BASE_DATA_DIR = "/scratch/bowenxi/dit/neural_tangent_kernel/feature_swin_b/auto_ML_CA"
RESULTS_DIR = "incremental_results_automl"
NUM_CLASSES = 1000
BATCH_SIZE = 1024
EPOCHS = 50  # Reduced for quicker trials
NUM_TRIALS = 30  # Number of hyperparameter optimization trials
OPTUNA_STUDY_NAME = "automl_ntk_study"
FEATURE_REDUCTION_METHODS = ["none", "pca", "select_k_best"]

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

def preprocess_features(X_train, X_val, method="standardize", n_components=None, k_best=None):
    """
    Preprocess features using various methods
    
    Args:
        X_train: Training features
        X_val: Validation features
        method: Preprocessing method ("standardize", "normalize", "none")
        n_components: Number of PCA components (if method="pca")
        k_best: Number of features to select (if method="select_k_best")
    
    Returns:
        Processed X_train, X_val and potentially the input dimension for model
    """
    if method == "none":
        return X_train, X_val, X_train.shape[1]
    
    if method == "standardize":
        print("Standardizing features...")
        mean = np.mean(X_train, axis=0)
        std = np.std(X_train, axis=0) + 1e-8
        
        X_train_proc = (X_train - mean) / std
        X_val_proc = (X_val - mean) / std
        return X_train_proc, X_val_proc, X_train.shape[1]
    
    elif method == "normalize":
        print("Normalizing features...")
        norm = np.linalg.norm(X_train, axis=1, keepdims=True) + 1e-8
        X_train_proc = X_train / norm
        
        norm_val = np.linalg.norm(X_val, axis=1, keepdims=True) + 1e-8
        X_val_proc = X_val / norm_val
        return X_train_proc, X_val_proc, X_train.shape[1]
    
    elif method == "pca":
        if n_components is None:
            n_components = min(X_train.shape[0], X_train.shape[1] // 2)
        
        print(f"Applying PCA with {n_components} components...")
        pca = PCA(n_components=n_components, random_state=42)
        X_train_proc = pca.fit_transform(X_train)
        X_val_proc = pca.transform(X_val)
        print(f"Explained variance ratio: {sum(pca.explained_variance_ratio_):.4f}")
        return X_train_proc, X_val_proc, n_components
    
    elif method == "select_k_best":
        if k_best is None:
            k_best = min(X_train.shape[1], 1000)
        
        print(f"Selecting {k_best} best features...")
        selector = SelectKBest(f_classif, k=k_best)
        X_train_proc = selector.fit_transform(X_train, y_train)
        X_val_proc = selector.transform(X_val)
        return X_train_proc, X_val_proc, k_best
    
    else:
        raise ValueError(f"Unknown preprocessing method: {method}")

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dims, num_classes, dropout_rate=0.0):
        super().__init__()
        
        layers = []
        prev_dim = input_dim
        
        # Hidden layers
        for dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, dim))
            layers.append(nn.ReLU())
            if dropout_rate > 0:
                layers.append(nn.Dropout(dropout_rate))
            prev_dim = dim
        
        # Output layer
        layers.append(nn.Linear(prev_dim, num_classes))
        
        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.model(x)

class ResidualMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes, num_blocks=2, dropout_rate=0.0):
        super().__init__()
        
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.blocks = nn.ModuleList()
        
        for _ in range(num_blocks):
            block = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate),
                nn.Linear(hidden_dim, hidden_dim),
                nn.Dropout(dropout_rate)
            )
            self.blocks.append(block)
        
        self.output = nn.Linear(hidden_dim, num_classes)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.input_proj(x)
        
        for block in self.blocks:
            residual = x
            x = block(x)
            x = self.relu(x + residual)  # Residual connection
        
        return self.output(x)

def create_model(model_type, input_dim, num_classes, params):
    """
    Create a model based on the type and parameters
    
    Args:
        model_type: Type of model ("mlp" or "residual_mlp")
        input_dim: Input dimension
        num_classes: Number of output classes
        params: Dictionary of model parameters
    
    Returns:
        Initialized model
    """
    if model_type == "mlp":
        hidden_dims = params["hidden_dims"]
        dropout_rate = params.get("dropout_rate", 0.0)
        return MLP(input_dim, hidden_dims, num_classes, dropout_rate)
    
    elif model_type == "residual_mlp":
        hidden_dim = params["hidden_dim"]
        num_blocks = params.get("num_blocks", 2)
        dropout_rate = params.get("dropout_rate", 0.0)
        return ResidualMLP(input_dim, hidden_dim, num_classes, num_blocks, dropout_rate)
    
    else:
        raise ValueError(f"Unknown model type: {model_type}")

def objective(trial, X_train, y_train, X_val, y_val, input_dim):
    """
    Optuna objective function for hyperparameter optimization
    
    Args:
        trial: Optuna trial object
        X_train, y_train, X_val, y_val: Training and validation data
        input_dim: Input dimension after preprocessing
    
    Returns:
        Validation accuracy
    """
    # Hyperparameter search space
    model_type = trial.suggest_categorical("model_type", ["mlp", "residual_mlp"])
    
    if model_type == "mlp":
        n_layers = trial.suggest_int("n_layers", 1, 3)
        hidden_dims = []
        for i in range(n_layers):
            hidden_dims.append(trial.suggest_int(f"hidden_dim_{i}", 128, 1024, step=128))
        dropout_rate = trial.suggest_float("dropout_rate", 0.0, 0.5)
        
        model_params = {
            "hidden_dims": hidden_dims,
            "dropout_rate": dropout_rate
        }
    
    elif model_type == "residual_mlp":
        hidden_dim = trial.suggest_int("hidden_dim", 128, 1024, step=128)
        num_blocks = trial.suggest_int("num_blocks", 1, 4)
        dropout_rate = trial.suggest_float("dropout_rate", 0.0, 0.5)
        
        model_params = {
            "hidden_dim": hidden_dim,
            "num_blocks": num_blocks,
            "dropout_rate": dropout_rate
        }
    
    # Optimization hyperparameters
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)
    batch_size = trial.suggest_categorical("batch_size", [256, 512, 1024, 2048])
    
    # Create datasets and loaders
    train_dataset = TensorDataset(
        torch.FloatTensor(X_train),
        torch.LongTensor(y_train)
    )
    val_dataset = TensorDataset(
        torch.FloatTensor(X_val),
        torch.LongTensor(y_val)
    )
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    # Create model
    model = create_model(model_type, input_dim, NUM_CLASSES, model_params)
    model = model.cuda()
    
    # Training setup
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    
    # Training loop
    best_acc = 0
    patience = 5
    patience_counter = 0
    max_epochs = min(30, EPOCHS)  # Limit to 30 epochs for quicker trials
    
    for epoch in range(max_epochs):
        # Training
        model.train()
        train_loss = 0
        correct = 0
        total = 0
        
        for inputs, labels in train_loader:
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
        
        # Report intermediate results
        trial.report(val_acc, epoch)
        
        # Handle pruning based on intermediate results
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()
        
        # Early stopping
        if val_acc > best_acc:
            best_acc = val_acc
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break
        
        scheduler.step()
    
    return best_acc

def run_automl(increment_name):
    """
    Run AutoML for a specific increment
    
    Args:
        increment_name: Name of the increment
    
    Returns:
        Dictionary with best model, parameters and results
    """
    print(f"\n{'='*80}")
    print(f"AutoML for increment: {increment_name}")
    print(f"{'='*80}")
    
    # Load data
    X_train, y_train, X_val, y_val = load_data_increment(increment_name)
    
    # First, optimize the preprocessing method
    preproc_study = optuna.create_study(
        direction="maximize",
        study_name=f"{OPTUNA_STUDY_NAME}_preproc_{increment_name}"
    )
    
    # Simple preprocessing optimization
    def preproc_objective(trial):
        method = trial.suggest_categorical("preproc_method", FEATURE_REDUCTION_METHODS)
        
        if method == "pca":
            n_components = trial.suggest_int("n_components", 
                                           min(100, X_train.shape[1] // 10), 
                                           min(1000, X_train.shape[1] // 2))
            X_train_proc, X_val_proc, input_dim = preprocess_features(
                X_train, X_val, method=method, n_components=n_components
            )
        elif method == "select_k_best":
            k_best = trial.suggest_int("k_best", 
                                      min(100, X_train.shape[1] // 10), 
                                      min(1000, X_train.shape[1] // 2))
            X_train_proc, X_val_proc, input_dim = preprocess_features(
                X_train, X_val, method=method, k_best=k_best
            )
        else:  # "none" or "standardize" or "normalize"
            X_train_proc, X_val_proc, input_dim = preprocess_features(
                X_train, X_val, method="standardize"
            )
        
        # Simple evaluation with a basic model
        model = MLP(input_dim, [512], NUM_CLASSES)
        model = model.cuda()
        
        # Training setup
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        
        # Create datasets and loaders
        train_dataset = TensorDataset(
            torch.FloatTensor(X_train_proc),
            torch.LongTensor(y_train)
        )
        val_dataset = TensorDataset(
            torch.FloatTensor(X_val_proc),
            torch.LongTensor(y_val)
        )
        
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
        
        # Quick training for 5 epochs
        for epoch in range(5):
            model.train()
            for inputs, labels in train_loader:
                inputs, labels = inputs.cuda(), labels.cuda()
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
        
        # Validation
        model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.cuda(), labels.cuda()
                outputs = model(inputs)
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        
        val_acc = correct / total
        return val_acc
    
    # Run preprocessing optimization
    preproc_study.optimize(preproc_objective, n_trials=10)
    
    # Get best preprocessing method
    best_preproc = preproc_study.best_params
    print(f"Best preprocessing: {best_preproc}")
    
    # Apply the best preprocessing
    if best_preproc["preproc_method"] == "pca":
        X_train_proc, X_val_proc, input_dim = preprocess_features(
            X_train, X_val, method="pca", n_components=best_preproc["n_components"]
        )
    elif best_preproc["preproc_method"] == "select_k_best":
        X_train_proc, X_val_proc, input_dim = preprocess_features(
            X_train, X_val, method="select_k_best", k_best=best_preproc["k_best"]
        )
    else:
        X_train_proc, X_val_proc, input_dim = preprocess_features(
            X_train, X_val, method="standardize"
        )
    
    # Now optimize the model architecture and training hyperparameters
    study = optuna.create_study(
        direction="maximize", 
        study_name=f"{OPTUNA_STUDY_NAME}_{increment_name}",
        pruner=optuna.pruners.MedianPruner()
    )
    
    # Create the objective function with data
    obj_func = partial(
        objective, 
        X_train=X_train_proc, 
        y_train=y_train, 
        X_val=X_val_proc, 
        y_val=y_val,
        input_dim=input_dim
    )
    
    # Run hyperparameter optimization
    study.optimize(obj_func, n_trials=NUM_TRIALS)
    
    print(f"Best trial:")
    trial = study.best_trial
    print(f"  Value: {trial.value}")
    print(f"  Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")
    
    # Create and train the best model
    model_type = trial.params["model_type"]
    
    if model_type == "mlp":
        n_layers = trial.params["n_layers"]
        hidden_dims = [trial.params[f"hidden_dim_{i}"] for i in range(n_layers)]
        dropout_rate = trial.params["dropout_rate"]
        
        model_params = {
            "hidden_dims": hidden_dims,
            "dropout_rate": dropout_rate
        }
    
    elif model_type == "residual_mlp":
        hidden_dim = trial.params["hidden_dim"]
        num_blocks = trial.params["num_blocks"]
        dropout_rate = trial.params["dropout_rate"]
        
        model_params = {
            "hidden_dim": hidden_dim,
            "num_blocks": num_blocks,
            "dropout_rate": dropout_rate
        }
    
    learning_rate = trial.params["learning_rate"]
    weight_decay = trial.params["weight_decay"]
    batch_size = trial.params["batch_size"]
    
    # Final training with the best model and hyperparameters
    best_model = create_model(model_type, input_dim, NUM_CLASSES, model_params)
    best_model = best_model.cuda()
    
    # Train the model with all epochs
    train_dataset = TensorDataset(
        torch.FloatTensor(X_train_proc),
        torch.LongTensor(y_train)
    )
    val_dataset = TensorDataset(
        torch.FloatTensor(X_val_proc),
        torch.LongTensor(y_val)
    )
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(best_model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }
    
    best_acc = 0
    best_epoch = 0
    start_time = time.time()
    
    for epoch in range(EPOCHS):
        # Training
        best_model.train()
        train_loss = 0
        correct = 0
        total = 0
        
        for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            inputs, labels = inputs.cuda(), labels.cuda()
            
            optimizer.zero_grad()
            outputs = best_model(inputs)
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
        best_model.eval()
        val_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.cuda(), labels.cuda()
                outputs = best_model(inputs)
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
        
        scheduler.step()
        
        # Save best model
        if val_acc > best_acc:
            best_acc = val_acc
            best_epoch = epoch
            model_path = os.path.join(RESULTS_DIR, f"best_model_{increment_name}_automl.pt")
            torch.save(best_model.state_dict(), model_path)
            print(f"Best model saved with validation accuracy: {val_acc:.2%}")
    
    training_time = time.time() - start_time
    
    # Save automl results
    automl_results = {
        "increment": increment_name,
        "best_preprocessing": best_preproc,
        "best_model_type": model_type,
        "best_model_params": model_params,
        "best_optimizer_params": {
            "learning_rate": learning_rate,
            "weight_decay": weight_decay,
            "batch_size": batch_size
        },
        "best_val_accuracy": best_acc,
        "best_epoch": best_epoch,
        "training_time": training_time,
        "history": history
    }
    
    # Save study and results
    optuna_results_path = os.path.join(RESULTS_DIR, f"optuna_results_{increment_name}.pkl")
    pd.to_pickle(automl_results, optuna_results_path)
    
    # Plot training history
    plot_training_history(history, f"{increment_name}_automl")
    
    # Save the optuna visualization
    try:
        fig1 = optuna.visualization.plot_optimization_history(study)
        fig1.write_image(os.path.join(RESULTS_DIR, f"optuna_history_{increment_name}.png"))
        
        fig2 = optuna.visualization.plot_param_importances(study)
        fig2.write_image(os.path.join(RESULTS_DIR, f"optuna_importance_{increment_name}.png"))
        
        fig3 = optuna.visualization.plot_slice(study)
        fig3.write_image(os.path.join(RESULTS_DIR, f"optuna_slice_{increment_name}.png"))
    except:
        print("Skipping optuna visualization (requires plotly)")
    
    return automl_results

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

def run_incremental_automl():
    """
    Run AutoML experiment for each incremental dataset
    
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
        
        # Run AutoML
        automl_results = run_automl(increment_name)
        
        # Get increment statistics
        increment_stats = get_increment_size(increment_name)
        
        # Extract key metrics
        X_train, y_train, _, _ = load_data_increment(increment_name)
        
        # Store results
        results.append({
            'increment': increment_name,
            'train_samples': len(X_train),
            'val_accuracy': automl_results['best_val_accuracy'],
            'best_epoch': automl_results['best_epoch'] + 1,
            'training_time': automl_results['training_time'],
            'best_model_type': automl_results['best_model_type'],
            'best_preprocessing': str(automl_results['best_preprocessing']),
            'synthetic_classes': increment_stats['synthetic_classes'],
            'synthetic_samples_per_class': increment_stats['synthetic_samples_per_class'],
            'total_synthetic': increment_stats['total_synthetic']
        })
        
        # Free up memory
        torch.cuda.empty_cache()
    
    # Convert to DataFrame
    results_df = pd.DataFrame(results)
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_df.to_csv(os.path.join(RESULTS_DIR, f'incremental_results_automl_{timestamp}.csv'), index=False)
    
    # Display results in sorted order
    sorted_results = results_df.sort_values('increment')
    print("\nAutoML Incremental Results:")
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
    plt.title('ImageNet Classification Accuracy vs Increment Level (AutoML)', fontsize=16)
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
    plt.savefig(os.path.join(RESULTS_DIR, 'accuracy_vs_increment_automl.png'), dpi=300)
    
    # Plot validation accuracy vs total samples
    plt.figure(figsize=(14, 8))
    plt.plot(sorted_df['train_samples'], sorted_df['val_accuracy'], 'ro-', markersize=8)
    plt.xlabel('Number of Training Samples', fontsize=14)
    plt.ylabel('Validation Accuracy', fontsize=14)
    plt.title('ImageNet Classification Accuracy vs Training Data Size (AutoML)', fontsize=16)
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
    plt.savefig(os.path.join(RESULTS_DIR, 'accuracy_vs_samples_automl.png'), dpi=300)
    
    # Create a comprehensive summary table as an image
    fig, ax = plt.subplots(figsize=(18, len(sorted_df) * 0.8))
    ax.axis('tight')
    ax.axis('off')
    
    # Format the data for the table
    table_data = [
        ['Increment', 'Train Samples', 'Model Type', 'Preprocessing', 'Val Accuracy', 'Time (s)']
    ]
    
    for i, row in sorted_df.iterrows():
        increment = row['increment']
        samples = f"{row['train_samples']:,}"
        model_type = row['best_model_type']
        preproc = row['best_preprocessing']
        accuracy = f"{row['val_accuracy']:.4f} ({row['val_accuracy']:.2%})"
        train_time = f"{row['training_time']:.1f}"
        
        table_data.append([increment, samples, model_type, preproc, accuracy, train_time])
    
    table = ax.table(cellText=table_data, loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1, 1.5)
    
    plt.savefig(os.path.join(RESULTS_DIR, 'results_summary_table_automl.png'), dpi=300, bbox_inches='tight')
    
    return sorted_df

def compare_with_baseline(automl_results_df, baseline_results_path=None):
    """
    Compare AutoML results with baseline results
    
    Args:
        automl_results_df: DataFrame with AutoML results
        baseline_results_path: Path to baseline results CSV (optional)
    """
    # If baseline results are not provided, check if they exist in the default location
    if baseline_results_path is None:
        baseline_files = glob.glob(os.path.join("incremental_results", "incremental_results_*.csv"))
        if baseline_files:
            baseline_results_path = sorted(baseline_files)[-1]  # Get the most recent
        else:
            print("No baseline results found. Skipping comparison.")
            return None
    
    # Load baseline results
    try:
        baseline_df = pd.read_csv(baseline_results_path)
        print(f"Loaded baseline results from {baseline_results_path}")
    except:
        print(f"Could not load baseline results from {baseline_results_path}. Skipping comparison.")
        return None
    
    # Process increment order for sorting
    baseline_df['increment_order'] = baseline_df['increment'].apply(
        lambda x: 0 if x == 'original' else int(x.split('_')[1])
    )
    baseline_df = baseline_df.sort_values('increment_order')
    
    automl_df = automl_results_df.copy()
    automl_df['increment_order'] = automl_df['increment'].apply(
        lambda x: 0 if x == 'original' else int(x.split('_')[1])
    )
    automl_df = automl_df.sort_values('increment_order')
    
    # Merge datasets
    comparison_df = pd.merge(
        baseline_df[['increment', 'increment_order', 'val_accuracy', 'training_time']],
        automl_df[['increment', 'val_accuracy', 'training_time', 'best_model_type']],
        on='increment',
        suffixes=('_baseline', '_automl')
    )
    
    # Calculate improvements
    comparison_df['accuracy_diff'] = comparison_df['val_accuracy_automl'] - comparison_df['val_accuracy_baseline']
    comparison_df['accuracy_improvement'] = (comparison_df['accuracy_diff'] / comparison_df['val_accuracy_baseline']) * 100
    comparison_df['time_ratio'] = comparison_df['training_time_automl'] / comparison_df['training_time_baseline']
    
    # Save comparison to CSV
    comparison_df.to_csv(os.path.join(RESULTS_DIR, 'automl_vs_baseline_comparison.csv'), index=False)
    
    # Create comparison plots
    plt.figure(figsize=(14, 8))
    
    increments = comparison_df['increment_order']
    baseline_acc = comparison_df['val_accuracy_baseline']
    automl_acc = comparison_df['val_accuracy_automl']
    
    bar_width = 0.35
    index = np.arange(len(increments))
    
    plt.bar(index, baseline_acc, bar_width, label='Baseline')
    plt.bar(index + bar_width, automl_acc, bar_width, label='AutoML')
    
    plt.xlabel('Increment Number (0=original)', fontsize=14)
    plt.ylabel('Validation Accuracy', fontsize=14)
    plt.title('Baseline vs AutoML: Validation Accuracy by Increment', fontsize=16)
    plt.xticks(index + bar_width / 2, comparison_df['increment'])
    plt.legend()
    plt.grid(True, axis='y')
    
    # Add text labels
    for i, (base_acc, auto_acc) in enumerate(zip(baseline_acc, automl_acc)):
        plt.annotate(f"{base_acc:.2%}", 
                     (i, base_acc), 
                     textcoords="offset points",
                     xytext=(0, 5), 
                     ha='center')
        plt.annotate(f"{auto_acc:.2%}", 
                     (i + bar_width, auto_acc), 
                     textcoords="offset points",
                     xytext=(0, 5), 
                     ha='center')
    
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'baseline_vs_automl_accuracy.png'), dpi=300)
    
    # Plot accuracy improvement
    plt.figure(figsize=(14, 8))
    plt.bar(comparison_df['increment'], comparison_df['accuracy_improvement'], color='green')
    plt.axhline(y=0, color='r', linestyle='-')
    plt.xlabel('Increment', fontsize=14)
    plt.ylabel('Accuracy Improvement (%)', fontsize=14)
    plt.title('AutoML Improvement Over Baseline (%)', fontsize=16)
    plt.grid(True, axis='y')
    
    # Add text labels
    for i, improvement in enumerate(comparison_df['accuracy_improvement']):
        plt.annotate(f"{improvement:.2f}%", 
                     (i, improvement), 
                     textcoords="offset points",
                     xytext=(0, 5 if improvement > 0 else -15), 
                     ha='center')
    
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'automl_improvement_percentage.png'), dpi=300)
    
    # Create comparison summary table
    fig, ax = plt.subplots(figsize=(18, len(comparison_df) * 0.8 + 1))
    ax.axis('tight')
    ax.axis('off')
    
    table_data = [
        ['Increment', 'Baseline Acc', 'AutoML Acc', 'Improvement', 'Best Model', 'Time Ratio']
    ]
    
    for i, row in comparison_df.iterrows():
        increment = row['increment']
        baseline_acc = f"{row['val_accuracy_baseline']:.2%}"
        automl_acc = f"{row['val_accuracy_automl']:.2%}"
        improvement = f"{row['accuracy_improvement']:.2f}%"
        model_type = row['best_model_type']
        time_ratio = f"{row['time_ratio']:.2f}x"
        
        table_data.append([increment, baseline_acc, automl_acc, improvement, model_type, time_ratio])
    
    table = ax.table(cellText=table_data, loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1, 1.5)
    
    # Color cells based on improvement
    for i in range(1, len(table_data)):
        improvement_val = comparison_df.iloc[i-1]['accuracy_improvement']
        if improvement_val > 0:
            table[(i, 3)].set_facecolor('#d8f3dc')  # Light green
        elif improvement_val < 0:
            table[(i, 3)].set_facecolor('#ffccd5')  # Light red
    
    plt.savefig(os.path.join(RESULTS_DIR, 'comparison_summary_table.png'), dpi=300, bbox_inches='tight')
    
    print("\nComparison Analysis:")
    print("=" * 100)
    print(comparison_df.to_string(index=False))
    print("=" * 100)
    
    avg_improvement = comparison_df['accuracy_improvement'].mean()
    print(f"Average accuracy improvement: {avg_improvement:.2f}%")
    
    return comparison_df

def analyze_model_choices(results_df):
    """
    Analyze which model types and preprocessing methods work best
    
    Args:
        results_df: DataFrame with AutoML results
    """
    # Count model types
    model_counts = results_df['best_model_type'].value_counts()
    
    # Find which preprocessing methods appear most often
    preproc_counts = results_df['best_preprocessing'].apply(lambda x: x.split("'")[3] if "'preproc_method': '" in x else "unknown").value_counts()
    
    # Create a grouped analysis
    grouped_results = results_df.groupby('best_model_type')['val_accuracy'].agg(['mean', 'min', 'max', 'count'])
    
    # Create visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Model type pie chart
    ax1.pie(model_counts, labels=model_counts.index, autopct='%1.1f%%', startangle=90)
    ax1.set_title('Model Type Distribution')
    
    # Preprocessing method pie chart
    ax2.pie(preproc_counts, labels=preproc_counts.index, autopct='%1.1f%%', startangle=90)
    ax2.set_title('Preprocessing Method Distribution')
    
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'model_choices_analysis.png'), dpi=300)
    
    # Create a table of results grouped by model type
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.axis('tight')
    ax.axis('off')
    
    table_data = [['Model Type', 'Count', 'Avg Accuracy', 'Min Accuracy', 'Max Accuracy']]
    
    for model_type, stats in grouped_results.iterrows():
        count = int(stats['count'])
        avg_acc = f"{stats['mean']:.2%}"
        min_acc = f"{stats['min']:.2%}"
        max_acc = f"{stats['max']:.2%}"
        
        table_data.append([model_type, count, avg_acc, min_acc, max_acc])
    
    table = ax.table(cellText=table_data, loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1, 1.5)
    
    plt.savefig(os.path.join(RESULTS_DIR, 'model_performance_table.png'), dpi=300, bbox_inches='tight')
    
    return {
        'model_counts': model_counts,
        'preproc_counts': preproc_counts,
        'grouped_results': grouped_results
    }

if __name__ == "__main__":
    print(f"Starting AutoML incremental tangent feature experiment at {datetime.now()}")
    print(f"Results will be saved to {RESULTS_DIR}")
    
    # Create results directory
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    # Run the AutoML experiment
    results = run_incremental_automl()
    
    # Plot summary results
    final_results = plot_final_results(results)
    
    # Compare with baseline if available
    comparison = compare_with_baseline(results)
    
    # Analyze model choices
    model_analysis = analyze_model_choices(results)
    
    print(f"\nAutoML experiment completed at {datetime.now()}")
    print(f"All results saved to {RESULTS_DIR}")