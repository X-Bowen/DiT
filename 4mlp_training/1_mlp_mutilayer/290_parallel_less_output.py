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
import itertools
import sys
from multiprocessing import Queue
from threading import Thread

# Configuration
BASE_DATA_DIR = "/scratch/bowenxi/dit/neural_tangent_kernel/feature_swin_b/auto_ML_CA_1/auto_ML_CA"
NUM_CLASSES = 1000
BATCH_SIZE = 2048
EPOCHS = 100

# Hyperparameters to test - Learning Rates and Weight Decay
LEARNING_RATES = [5e-5, 1e-4, 5e-4, 1e-3, 5e-3]
WEIGHT_DECAYS = [1e-5, 5e-5, 1e-4, 5e-4, 1e-3]

# Get all combinations of learning rates and weight decays
HYPERPARAMETER_COMBINATIONS = list(itertools.product(LEARNING_RATES, WEIGHT_DECAYS))

# Create unique RESULTS_DIR with timestamp
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
RESULTS_DIR = os.path.join("incremental_results", f"experiment_{timestamp}")
os.makedirs(RESULTS_DIR, exist_ok=True)

# Set up logging
log_file = os.path.join(RESULTS_DIR, 'experiment.log')
log_queue = Queue()

def log_writer(queue, log_file):
    with open(log_file, 'w') as f:
        while True:
            message = queue.get()
            if message is None:
                break
            f.write(message + '\n')
            f.flush()
            print(message)

log_thread = Thread(target=log_writer, args=(log_queue, log_file))
log_thread.start()

def log(message, queue):
    queue.put(message)

# Maximum number of parallel processes
MAX_PARALLEL_PROCESSES = min(
    len(HYPERPARAMETER_COMBINATIONS),
    torch.cuda.device_count() if torch.cuda.is_available() else mp.cpu_count()
)

def load_data_increment(increment_name):
    train_path = os.path.join(BASE_DATA_DIR, f"{increment_name}_train_tangent.npz")
    val_path = os.path.join(BASE_DATA_DIR, f"{increment_name}_val_tangent.npz")
    
    log(f"Loading {increment_name} training data from {train_path}...", log_queue)
    with np.load(train_path) as data:
        X_train = data['features']
        y_train = data['labels']
    
    log(f"Loading {increment_name} validation data from {val_path}...", log_queue)
    with np.load(val_path) as data:
        X_val = data['features']
        y_val = data['labels']
    
    log(f"{increment_name} training data: {X_train.shape}, validation data: {X_val.shape}", log_queue)
    
    return X_train, y_train, X_val, y_val

def standardize_features(X_train, X_val):
    log("Standardizing features...", log_queue)
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
            nn.Dropout(0.2),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, num_classes)
        )
        
    def forward(self, x):
        return self.layers(x)
        
def get_available_gpu():
    if not torch.cuda.is_available():
        return None
    num_gpus = torch.cuda.device_count()
    if num_gpus == 1:
        return 0
    available_memory = []
    for i in range(num_gpus):
        free_memory = torch.cuda.get_device_properties(i).total_memory - torch.cuda.memory_allocated(i)
        available_memory.append((i, free_memory))
    available_memory.sort(key=lambda x: x[1], reverse=True)
    return available_memory[0][0]

def train_model_worker(params):
    X_train = params['X_train']
    y_train = params['y_train']
    X_val = params['X_val']
    y_val = params['y_val']
    increment_name = params['increment_name']
    learning_rate = params['learning_rate']
    weight_decay = params['weight_decay']
    gpu_id = params.get('gpu_id', get_available_gpu())
    log_queue = params['log_queue']
    
    device = torch.device(f"cuda:{gpu_id}" if gpu_id is not None else "cpu")
    
    log(f"Training {increment_name} with LR={learning_rate:.1e}, WD={weight_decay:.1e} on {device}", log_queue)
    
    train_dataset = TensorDataset(torch.FloatTensor(X_train), torch.LongTensor(y_train))
    val_dataset = TensorDataset(torch.FloatTensor(X_val), torch.LongTensor(y_val))
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
    
    model = AdaptiveMLP(X_train.shape[1], NUM_CLASSES).to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=0)
    
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    best_acc = 0
    best_epoch = 0
    start_time = time.time()
    
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0
        correct = 0
        total = 0
        for inputs, labels in train_loader:
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
        
        if epoch % 10 == 0:
            log(f"LR {learning_rate:.1e}, WD {weight_decay:.1e} - Epoch {epoch+1}: Train Loss: {avg_train_loss:.4f} | Train Acc: {train_acc:.2%}", log_queue)
        
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
        
        if epoch % 10 == 0:
            log(f"LR {learning_rate:.1e}, WD {weight_decay:.1e} - Epoch {epoch+1}: Val Loss: {avg_val_loss:.4f} | Val Acc: {val_acc:.2%}", log_queue)
        
        scheduler.step()
        
        if val_acc > best_acc:
            best_acc = val_acc
            best_epoch = epoch
            model_path = os.path.join(RESULTS_DIR, f"best_model_{increment_name}_lr_{learning_rate:.1e}_wd_{weight_decay:.1e}.pt")
            torch.save(model.state_dict(), model_path)
    
    training_time = time.time() - start_time
    log(f"Training completed for LR={learning_rate:.1e}, WD={weight_decay:.1e} in {training_time:.2f} seconds.", log_queue)
    log(f"Best validation accuracy: {best_acc:.4f} at epoch {best_epoch+1}", log_queue)
    
    plot_training_history(history, increment_name, learning_rate, weight_decay)
    
    model = model.cpu()
    torch.cuda.empty_cache()
    
    return {
        'increment': increment_name,
        'learning_rate': learning_rate,
        'weight_decay': weight_decay,
        'history': history,
        'best_acc': best_acc,
        'best_epoch': best_epoch,
        'training_time': training_time
    }

def plot_training_history(history, increment_name, learning_rate, weight_decay):
    epochs = range(1, len(history['train_loss']) + 1)
    plt.figure(figsize=(12, 10))
    plt.subplot(2, 1, 1)
    plt.plot(epochs, history['train_loss'], 'b-', label='Training Loss')
    plt.plot(epochs, history['val_loss'], 'r-', label='Validation Loss')
    plt.title(f'Training and Validation Loss ({increment_name}, LR={learning_rate:.1e}, WD={weight_decay:.1e})')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.subplot(2, 1, 2)
    plt.plot(epochs, history['train_acc'], 'b-', label='Training Accuracy')
    plt.plot(epochs, history['val_acc'], 'r-', label='Validation Accuracy')
    plt.title(f'Training and Validation Accuracy ({increment_name}, LR={learning_rate:.1e}, WD={weight_decay:.1e})')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, f'training_history_{increment_name}_lr_{learning_rate:.1e}_wd_{weight_decay:.1e}.png'), dpi=300)
    plt.close()

def get_increment_size(increment_name):
    if increment_name == 'original':
        synthetic_classes = 0
        synthetic_samples_per_class = 0
    else:
        increment_num = int(increment_name.split('_')[1])
        synthetic_classes = NUM_CLASSES
        synthetic_samples_per_class = 1000 * increment_num
    return {
        'name': increment_name,
        'synthetic_classes': synthetic_classes,
        'synthetic_samples_per_class': synthetic_samples_per_class,
        'total_synthetic': synthetic_classes * synthetic_samples_per_class
    }

def run_incremental_experiment_parallel():
    train_files = sorted(glob.glob(os.path.join(BASE_DATA_DIR, "*_train_tangent.npz")))
    increment_names = [os.path.basename(f).replace("_train_tangent.npz", "") for f in train_files]
    
    log(f"Found {len(increment_names)} increments: {increment_names}", log_queue)
    
    mp.set_start_method('spawn', force=True)
    all_results = []
    
    for increment_name in tqdm(increment_names, desc="Processing increments"):
        log(f"\n{'='*80}", log_queue)
        log(f"Processing increment: {increment_name}", log_queue)
        log(f"{'='*80}", log_queue)
        
        X_train, y_train, X_val, y_val = load_data_increment(increment_name)
        X_train_std, X_val_std = standardize_features(X_train, X_val)
        del X_train, X_val
        increment_stats = get_increment_size(increment_name)
        
        training_params = []
        for i, (lr, wd) in enumerate(HYPERPARAMETER_COMBINATIONS):
            gpu_id = i % torch.cuda.device_count() if torch.cuda.is_available() else None
            training_params.append({
                'X_train': X_train_std,
                'y_train': y_train,
                'X_val': X_val_std,
                'y_val': y_val,
                'increment_name': increment_name,
                'learning_rate': lr,
                'weight_decay': wd,
                'gpu_id': gpu_id,
                'log_queue': log_queue
            })
        
        results = []
        for i in range(0, len(training_params), MAX_PARALLEL_PROCESSES):
            batch_params = training_params[i:i + MAX_PARALLEL_PROCESSES]
            with concurrent.futures.ProcessPoolExecutor(max_workers=len(batch_params)) as executor:
                future_to_params = {executor.submit(train_model_worker, params): params for params in batch_params}
                for future in concurrent.futures.as_completed(future_to_params):
                    params = future_to_params[future]
                    lr, wd = params['learning_rate'], params['weight_decay']
                    try:
                        result = future.result()
                        log(f"Completed training for {increment_name} with LR={lr:.1e}, WD={wd:.1e}", log_queue)
                        results.append(result)
                    except Exception as e:
                        log(f"Training failed for {increment_name} with LR={lr:.1e}, WD={wd:.1e}: {e}", log_queue)
        
        for result in results:
            all_results.append({
                'increment': result['increment'],
                'learning_rate': result['learning_rate'],
                'weight_decay': result['weight_decay'],
                'train_samples': len(X_train_std),
                'val_accuracy': result['best_acc'],
                'best_epoch': result['best_epoch'] + 1,
                'training_time': result['training_time'],
                'synthetic_classes': increment_stats['synthetic_classes'],
                'synthetic_samples_per_class': increment_stats['synthetic_samples_per_class'],
                'total_synthetic': increment_stats['total_synthetic']
            })
        
        del X_train_std, X_val_std, y_train, y_val
        torch.cuda.empty_cache()
    
    results_df = pd.DataFrame(all_results)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_df.to_csv(os.path.join(RESULTS_DIR, f'incremental_results_{timestamp}.csv'), index=False)
    
    sorted_results = results_df.sort_values(['increment', 'learning_rate', 'weight_decay'])
    log("\nIncremental Results:", log_queue)
    log("=" * 140, log_queue)
    log(sorted_results.to_string(index=False), log_queue)
    log("=" * 140, log_queue)
    
    return results_df

def plot_hyperparameter_comparison(results_df):
    results_df['increment_order'] = results_df['increment'].apply(lambda x: 0 if x == 'original' else int(x.split('_')[1]))
    increments = sorted(results_df['increment'].unique(), key=lambda x: 0 if x == 'original' else int(x.split('_')[1]))
    
    for increment in increments:
        subset = results_df[results_df['increment'] == increment]
        pivot_data = subset.pivot_table(index='weight_decay', columns='learning_rate', values='val_accuracy')
        plt.figure(figsize=(12, 10))
        heatmap = plt.imshow(pivot_data, cmap='viridis', aspect='auto')
        plt.colorbar(heatmap, label='Validation Accuracy')
        plt.yticks(range(len(pivot_data.index)), [f"{wd:.1e}" for wd in pivot_data.index])
        plt.xticks(range(len(pivot_data.columns)), [f"{lr:.1e}" for lr in pivot_data.columns])
        plt.xlabel('Learning Rate', fontsize=14)
        plt.ylabel('Weight Decay', fontsize=14)
        plt.title(f'Validation Accuracy Heatmap for {increment}', fontsize=16)
        for i in range(len(pivot_data.index)):
            for j in range(len(pivot_data.columns)):
                try:
                    value = pivot_data.iloc[i, j]
                    plt.text(j, i, f"{value:.2%}", ha='center', va='center', color='white' if value < pivot_data.values.mean() else 'black')
                except:
                    pass
        plt.tight_layout()
        plt.savefig(os.path.join(RESULTS_DIR, f'accuracy_heatmap_{increment}.png'), dpi=300)
        plt.close()
    
    best_params = results_df.loc[results_df.groupby('increment')['val_accuracy'].idxmax()]
    plt.figure(figsize=(14, 8))
    plt.plot(best_params['increment_order'], best_params['val_accuracy'], 'bo-', markersize=8)
    plt.xlabel('Increment Number (0=original)', fontsize=14)
    plt.ylabel('Best Validation Accuracy', fontsize=14)
    plt.title('Best ImageNet Classification Accuracy vs Increment Level', fontsize=16)
    plt.grid(True)
    plt.xticks(best_params['increment_order'])
    for i, row in best_params.iterrows():
        plt.annotate(f"{row['val_accuracy']:.2%}\nLR={row['learning_rate']:.1e}\nWD={row['weight_decay']:.1e}", 
                     (row['increment_order'], row['val_accuracy']), textcoords="offset points", xytext=(0, 10), ha='center')
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'best_accuracy_vs_increment.png'), dpi=300)
    plt.close()
    
    fig, ax = plt.subplots(figsize=(18, len(best_params) * 0.8))
    ax.axis('tight')
    ax.axis('off')
    table_data = [['Increment', 'Best LR', 'Best WD', 'Val Accuracy', 'Train Samples', 'Best Epoch', 'Training Time (s)']]
    for i, row in best_params.iterrows():
        increment = row['increment']
        lr = f"{row['learning_rate']:.1e}"
        wd = f"{row['weight_decay']:.1e}"
        accuracy = f"{row['val_accuracy']:.4f} ({row['val_accuracy']:.2%})"
        samples = f"{row['train_samples']:,}"
        epoch = f"{row['best_epoch']}"
        train_time = f"{row['training_time']:.1f}"
        table_data.append([increment, lr, wd, accuracy, samples, epoch, train_time])
    table = ax.table(cellText=table_data, loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1, 1.5)
    plt.savefig(os.path.join(RESULTS_DIR, 'best_hyperparams_summary_table.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    top_results = [results_df[results_df['increment'] == inc].nlargest(3, 'val_accuracy') for inc in increments]
    top_df = pd.concat(top_results)
    plt.figure(figsize=(14, 10))
    markers = ['o', 's', 'D', '^', 'v', '<', '>', 'p', '*', 'h', '+', 'x']
    for i, increment in enumerate(increments):
        subset = top_df[top_df['increment'] == increment]
        marker = markers[i % len(markers)]
        plt.scatter(subset['learning_rate'], subset['weight_decay'], s=subset['val_accuracy']*1000, label=increment, marker=marker, alpha=0.7)
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Learning Rate (log scale)', fontsize=14)
    plt.ylabel('Weight Decay (log scale)', fontsize=14)
    plt.title('Best Hyperparameter Combinations by Increment', fontsize=16)
    plt.legend(title='Increment')
    plt.grid(True, which='both', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'best_hyperparameter_combinations.png'), dpi=300)
    plt.close()
    
    return best_params

def get_gpu_info():
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        gpu_names = [torch.cuda.get_device_name(i) for i in range(num_gpus)]
        return {"available": True, "count": num_gpus, "devices": gpu_names}
    return {"available": False, "count": 0, "devices": []}

if __name__ == "__main__":
    gpu_info = get_gpu_info()
    log(f"Starting parallel hyperparameter search experiment at {datetime.now()}", log_queue)
    if gpu_info["available"]:
        for i, name in enumerate(gpu_info["devices"]):
            log(f"  - GPU {i}: {name}", log_queue)
    else:
        log("  - No GPUs available, will use CPU", log_queue)
    
    log(f"Testing {len(LEARNING_RATES)} learning rates: {LEARNING_RATES}", log_queue)
    log(f"Testing {len(WEIGHT_DECAYS)} weight decays: {WEIGHT_DECAYS}", log_queue)
    log(f"Total {len(HYPERPARAMETER_COMBINATIONS)} combinations to test", log_queue)
    log(f"Results will be saved to {RESULTS_DIR}", log_queue)
    
    results = run_incremental_experiment_parallel()
    best_params = plot_hyperparameter_comparison(results)
    
    log(f"\nExperiment completed at {datetime.now()}", log_queue)
    log(f"Best hyperparameters for each increment:", log_queue)
    log(best_params[['increment', 'learning_rate', 'weight_decay', 'val_accuracy']].to_string(index=False), log_queue)
    log(f"\nAll results saved to {RESULTS_DIR}", log_queue)
    
    if gpu_info["available"]:
        estimated_sequential_time = results['training_time'].sum()
        actual_parallel_time = results.groupby('increment')['training_time'].sum().max()
        speedup = estimated_sequential_time / actual_parallel_time
        log(f"\nParallel training speedup: {speedup:.2f}x", log_queue)
        log(f"Estimated sequential training time: {estimated_sequential_time/3600:.2f} hours", log_queue)
        log(f"Actual parallel training time: {actual_parallel_time/3600:.2f} hours", log_queue)
    
    log_queue.put(None)
    log_thread.join()