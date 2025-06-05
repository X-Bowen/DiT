import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# ---- Load your data ----
# Assume you already have these loaded as NumPy arrays:
#   F: (N x D) feature matrix
#   labels: (N,) label vector
#   A: (D x D) = F^T F matrix

# Replace with actual data loading
F_np = /scratch/bowenxi/dit/neural_tangent_kernel/feature_swin_b/incremental_synthetic/  # e.g., np.load("image_features_w_label_train.npz")["features"]
labels_np = ...  # e.g., np.load("image_features_w_label_train.npz")["labels"]
A_np = ...  # e.g., np.load("original_A_matrix.npy")

# ---- Convert to PyTorch tensors ----
device = "cuda" if torch.cuda.is_available() else "cpu"
F_tensor = torch.from_numpy(F_np).float().to(device)         # (N x D)
labels_tensor = torch.from_numpy(labels_np).long().to(device)  # (N,)
A_tensor = torch.from_numpy(A_np).float().to(device)         # (D x D)
N, D = F_tensor.shape
num_classes = 1000  # for ImageNet

# ---- Dataset & Dataloader ----
batch_size = 8192
dataset = TensorDataset(F_tensor, labels_tensor)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)

class PolyTransform(nn.Module):
    def __init__(self, A):
        super().__init__()
        self.A = A.detach()  
        self.I = torch.eye(A.size(0), device=A.device)
        self.w0 = nn.Parameter(torch.tensor(0.01))
        self.w1 = nn.Parameter(torch.tensor(0.001))
        self.w2 = nn.Parameter(torch.tensor(0.0001))

    def forward(self, F):
        A = self.A
        A2 = A @ A
        A3 = A2 @ A
        poly_mat = self.I + self.w0 * A + self.w1 * A2 + self.w2 * A3  # (D x D)
        return F @ poly_mat  


class NTKClassifier(nn.Module):
    def __init__(self, A, num_classes):
        super().__init__()
        self.poly = PolyTransform(A)
        self.classifier = nn.Linear(A.size(0), num_classes)

    def forward(self, F):
        transformed = self.poly(F)         # (N x D)
        logits = self.classifier(transformed)  # (N x num_classes)
        return logits

# ---- Initialize model ----
model = NTKClassifier(A_tensor, num_classes).to(device)

# ---- Loss & Optimizer ----
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# ---- Training Loop ----
num_epochs = 5
for epoch in range(num_epochs):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (F_batch, y_batch) in enumerate(dataloader):
        optimizer.zero_grad()
        outputs = model(F_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += y_batch.size(0)
        correct += predicted.eq(y_batch).sum().item()

        if batch_idx % 10 == 0:
            print(f"Epoch {epoch+1}/{num_epochs} | Batch {batch_idx} | "
                  f"Loss: {loss.item():.4f} | Acc: {100.*correct/total:.2f}%")

    epoch_loss = total_loss / len(dataloader)
    epoch_acc = 100. * correct / total
    print(f"Epoch {epoch+1} completed. Avg Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.2f}%")

# ---- Save learned weights if needed ----
torch.save(model.state_dict(), "ntk_poly_model.pth")
print(f"Learned weights: w0={model.poly.w0.item():.6f}, w1={model.poly.w1.item():.6f}, w2={model.poly.w2.item():.6f}")
