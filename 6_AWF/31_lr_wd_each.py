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

# Configuration
BASE_DATA_DIR = "/scratch/bowenxi/dit/neural_tangent_kernel/feature_swin_b/auto_ML_CA_1/auto_ML_CA"
RESULTS_DIR = "incremental_results_2_layer_0507_1"
NUM_CLASSES = 1000
BATCH_SIZE = 2048
EPOCHS = 100
LEARNING_RATES_T = [1e-4, 1e-3, 1e-2]
LEARNING_RATES_S = [1e-5, 1e-4, 1e-3]
WEIGHT_DECAY = 1e-3
# MAX_WORKERS = len(LEARNING_RATES_T)
MAX_WORKERS = 9

os.makedirs(RESULTS_DIR, exist_ok=True)

class FeatureTransformer(nn.Module):
    def __init__(self, A):
        super().__init__()
        self.register_buffer('A', torch.tensor(A, dtype=torch.float32))
        self.w0 = nn.Parameter(torch.tensor(0.1))
        self.w1 = nn.Parameter(torch.tensor(0.01))
        self.w2 = nn.Parameter(torch.tensor(0.001))

    def forward(self, F):
        A = self.A
        A2 = A @ A
        A3 = A2 @ A
        I = torch.eye(A.size(0), device=A.device)
        transform = I + self.w0 * A + self.w1 * A2 + self.w2 * A3
        return F @ transform

class AdaptiveMLP(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, num_classes)
        )
    def forward(self, x):
        return self.layers(x)

def load_data_increment(increment_name):
    train_path = os.path.join(BASE_DATA_DIR, f"{increment_name}_train_tangent.npz")
    val_path = os.path.join(BASE_DATA_DIR, f"{increment_name}_val_tangent.npz")

    with np.load(train_path) as data:
        X_train = data['features']
        y_train = data['labels']

    with np.load(val_path) as data:
        X_val = data['features']
        y_val = data['labels']

    return X_train, y_train, X_val, y_val

def standardize_features(X_train, X_val):
    mean = np.mean(X_train, axis=0)
    std = np.std(X_train, axis=0) + 1e-8
    return (X_train - mean) / std, (X_val - mean) / std

def train_model_worker(params):
    X_train = params['X_train']
    y_train = params['y_train']
    X_val = params['X_val']
    y_val = params['y_val']
    increment_name = params['increment_name']
    transformer_lr = params['transformer_lr']
    model_lr = params['model_lr']
    A = params['A']
    gpu_id = params.get('gpu_id', 0)

    device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")

    train_dataset = TensorDataset(torch.FloatTensor(X_train), torch.LongTensor(y_train))
    val_dataset = TensorDataset(torch.FloatTensor(X_val), torch.LongTensor(y_val))

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

    transformer = FeatureTransformer(A).to(device)
    model = AdaptiveMLP(X_train.shape[1], NUM_CLASSES).to(device)

    optimizer = torch.optim.Adam([
        {'params': transformer.parameters(), 'lr': transformer_lr, 'weight_decay': WEIGHT_DECAY},
        {'params': model.parameters(), 'lr': model_lr, 'weight_decay': WEIGHT_DECAY}
    ])

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    criterion = nn.CrossEntropyLoss()

    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    best_acc = 0
    best_epoch = 0
    start_time = time.time()

    for epoch in range(EPOCHS):
        model.train()
        transformer.train()
        train_loss, correct, total = 0, 0, 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            transformed = transformer(inputs)
            outputs = model(transformed)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        train_acc = correct / total
        history['train_loss'].append(train_loss / len(train_loader))
        history['train_acc'].append(train_acc)

        model.eval()
        transformer.eval()
        val_loss, correct, total = 0, 0, 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                transformed = transformer(inputs)
                outputs = model(transformed)
                loss = criterion(outputs, labels)

                val_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

        val_acc = correct / total
        history['val_loss'].append(val_loss / len(val_loader))
        history['val_acc'].append(val_acc)

        scheduler.step()

        if val_acc > best_acc:
            best_acc = val_acc
            best_epoch = epoch
            model_path = os.path.join(RESULTS_DIR, f"best_model_{increment_name}_t_lr_{transformer_lr:.1e}_m_lr_{model_lr:.1e}.pt")
            torch.save({
                'transformer': transformer.state_dict(),
                'model': model.state_dict()
            }, model_path)

    training_time = time.time() - start_time
    return {
        'increment': increment_name,
        'transformer_lr': transformer_lr,
        'model_lr': model_lr,
        'history': history,
        'best_acc': best_acc,
        'best_epoch': best_epoch,
        'training_time': training_time
    }

def run_incremental_experiment_parallel():
    train_files = sorted(glob.glob(os.path.join(BASE_DATA_DIR, "*_train_tangent.npz")))
    increment_names = [os.path.basename(f).replace("_train_tangent.npz", "") for f in train_files]

    mp.set_start_method('spawn', force=True)
    all_results = []

    for increment_name in increment_names:
        X_train, y_train, X_val, y_val = load_data_increment(increment_name)
        X_train_std, X_val_std = standardize_features(X_train, X_val)

        A_path = os.path.join(BASE_DATA_DIR, f"{increment_name}_A_matrix.npy")
        A = np.load(A_path)  # A should be shape [1024, 1024]

        training_params = []
        i = 0
        for t_lr in LEARNING_RATES_T:
            for s_lr in LEARNING_RATES_S:
                gpu_id = i % torch.cuda.device_count() if torch.cuda.is_available() else None
                training_params.append({
                    'X_train': X_train_std,
                    'y_train': y_train,
                    'X_val': X_val_std,
                    'y_val': y_val,
                    'increment_name': increment_name,
                    'transformer_lr': t_lr,
                    'model_lr': s_lr,
                    'gpu_id': gpu_id,
                    'A': A
                })
                i += 1

        results = []
        with concurrent.futures.ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
            future_to_params = {executor.submit(train_model_worker, params): params for params in training_params}
            for future in concurrent.futures.as_completed(future_to_params):
                params = future_to_params[future]
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    print(f"Training failed for {params['increment_name']} with transformer_lr={params['transformer_lr']:.1e} and model_lr={params['model_lr']:.1e}: {e}")

        for result in results:
            all_results.append({
                'increment': result['increment'],
                'transformer_lr': result['transformer_lr'],
                'model_lr': result['model_lr'],
                'val_accuracy': result['best_acc'],
                'best_epoch': result['best_epoch'],
                'training_time': result['training_time']
            })

    results_df = pd.DataFrame(all_results)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_df.to_csv(os.path.join(RESULTS_DIR, f'incremental_results_{timestamp}.csv'), index=False)
    return results_df


if __name__ == "__main__":
    print(f"Starting training at {datetime.now()}")
    results = run_incremental_experiment_parallel()
    print("Training completed.")
