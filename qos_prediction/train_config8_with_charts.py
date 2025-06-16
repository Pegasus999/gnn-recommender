import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from torch_geometric.data import Data
from torch_geometric.nn import GraphConv
import matplotlib.pyplot as plt
import os

# 1. Load & normalize RT
rt = np.loadtxt("rtMatrix.txt")                     # [num_users, num_services]
num_users, num_services = rt.shape
observed_mask = (rt > 0)

max_rt = float(rt[observed_mask].max())
rt_norm = np.zeros_like(rt, dtype=np.float32)
rt_norm[observed_mask] = rt[observed_mask] / max_rt

# 2. Build homogeneous graph: users 0..U-1, services U..U+S-1
num_nodes = num_users + num_services

"""
Enhanced Node Features (11-dimensional):
0. is_user: Binary indicator for user nodes
1. is_service: Binary indicator for service nodes
2. degrees_norm: Normalized degree (connectivity)
3. avg_rt: Average response time for the node
4. std_rt: Standard deviation of response times
5. cv_rt: Coefficient of variation (std/mean) - relative variability
6. min_rt: Minimum response time observed
7. max_rt_node: Maximum response time observed  
8. range_rt: Response time range (max - min)
9. connectivity_ratio: Ratio of connections to total possible connections
10. activity_level: Normalized activity level
"""

# Compute enhanced node features
degrees = np.zeros(num_nodes, dtype=np.float32)
sum_rt = np.zeros(num_nodes, dtype=np.float32)
sum_rt_squared = np.zeros(num_nodes, dtype=np.float32)
min_rt = np.full(num_nodes, np.inf, dtype=np.float32)
max_rt_node = np.zeros(num_nodes, dtype=np.float32)

# Node type indicators (binary features)
is_user = np.zeros(num_nodes, dtype=np.float32)
is_service = np.zeros(num_nodes, dtype=np.float32)

# User features
for u in range(num_users):
    is_user[u] = 1.0
    sel = np.where(observed_mask[u])[0]
    degrees[u] = len(sel)
    if len(sel) > 0:
        rt_values = rt_norm[u, sel]
        sum_rt[u] = rt_values.sum()
        sum_rt_squared[u] = (rt_values ** 2).sum()
        min_rt[u] = rt_values.min()
        max_rt_node[u] = rt_values.max()
    else:
        min_rt[u] = 0.0

# Service features
for s in range(num_services):
    idx = num_users + s
    is_service[idx] = 1.0
    sel = np.where(observed_mask[:, s])[0]
    degrees[idx] = len(sel)
    if len(sel) > 0:
        rt_values = rt_norm[sel, s]
        sum_rt[idx] = rt_values.sum()
        sum_rt_squared[idx] = (rt_values ** 2).sum()
        min_rt[idx] = rt_values.min()
        max_rt_node[idx] = rt_values.max()
    else:
        min_rt[idx] = 0.0

# Normalize degree into [0,1]
max_deg = float(degrees.max())
if max_deg == 0:
    max_deg = 1.0
degrees_norm = degrees / max_deg

# Compute statistical features
avg_rt = np.zeros_like(sum_rt)
variance_rt = np.zeros_like(sum_rt)
nonzero = degrees > 0

avg_rt[nonzero] = sum_rt[nonzero] / degrees[nonzero]
# Variance = E[X²] - (E[X])²
variance_rt[nonzero] = (sum_rt_squared[nonzero] / degrees[nonzero]) - (avg_rt[nonzero] ** 2)
variance_rt = np.maximum(variance_rt, 0)  # Ensure non-negative due to numerical precision

# Standard deviation
std_rt = np.sqrt(variance_rt)

# Coefficient of variation (std/mean) - measure of relative variability
cv_rt = np.zeros_like(std_rt)
nonzero_avg = (avg_rt > 1e-8)
cv_rt[nonzero_avg] = std_rt[nonzero_avg] / avg_rt[nonzero_avg]

# Range (max - min)
range_rt = max_rt_node - min_rt
range_rt[~nonzero] = 0.0

# Connectivity ratio (for users: services connected / total services, for services: users connected / total users)
connectivity_ratio = np.zeros(num_nodes, dtype=np.float32)
connectivity_ratio[:num_users] = degrees[:num_users] / num_services  # Users
connectivity_ratio[num_users:] = degrees[num_users:] / num_users     # Services

# Activity level (normalized by maximum possible connections)
activity_level = degrees / np.maximum(num_services, num_users)

# Enhanced node features: [node_type, degree_norm, avg_rt, std_rt, cv_rt, min_rt, max_rt, range_rt, connectivity_ratio, activity_level]
x_np = np.stack([
    is_user,           # 0: user indicator
    is_service,        # 1: service indicator  
    degrees_norm,      # 2: normalized degree
    avg_rt,           # 3: average response time
    std_rt,           # 4: standard deviation of response time
    cv_rt,            # 5: coefficient of variation
    min_rt,           # 6: minimum response time
    max_rt_node,      # 7: maximum response time
    range_rt,         # 8: response time range
    connectivity_ratio, # 9: connectivity ratio
    activity_level     # 10: activity level
], axis=1)

x = torch.from_numpy(x_np).to(torch.float)

# 3. Build edge_index and edge_weight for observed edges
edge_src, edge_dst, edge_weight = [], [], []
for u in range(num_users):
    for s in np.where(observed_mask[u])[0]:
        edge_src.append(u)
        edge_dst.append(num_users + s)
        edge_weight.append(rt_norm[u, s])

# Make bidirectional
edge_index = torch.tensor([
    edge_src + edge_dst,
    edge_dst + edge_src
], dtype=torch.long)
w = torch.tensor(edge_weight + edge_weight, dtype=torch.float)

data = Data(x=x, edge_index=edge_index, edge_weight=w)

# 4. Split observed edges (only first half for train/val/test)
num_obs = len(edge_weight)
all_idx = np.arange(num_obs)
train_idx, temp_idx = train_test_split(all_idx, test_size=0.20, random_state=42)
val_idx, test_idx  = train_test_split(temp_idx, test_size=0.50, random_state=42)

train_mask = torch.zeros(2 * num_obs, dtype=torch.bool)
val_mask   = torch.zeros(2 * num_obs, dtype=torch.bool)
test_mask  = torch.zeros(2 * num_obs, dtype=torch.bool)

train_mask[torch.tensor(train_idx)] = True
val_mask[torch.tensor(val_idx)]     = True
test_mask[torch.tensor(test_idx)]   = True

# Mirror masks for service->user edges
train_mask[num_obs: 2*num_obs][train_idx] = True
val_mask[num_obs: 2*num_obs][val_idx]     = True
test_mask[num_obs: 2*num_obs][test_idx]   = True

data.train_mask = train_mask
data.val_mask   = val_mask
data.test_mask  = test_mask

# Target vector (normalized RT) for each directed edge
rt_all = torch.cat([torch.tensor(edge_weight), torch.tensor(edge_weight)])  # [2*num_obs]

# 5. Compute symmetric normalized edge weights
deg = torch.zeros(num_nodes, dtype=torch.float)
deg.index_add_(0, edge_index[0], w)
deg.index_add_(0, edge_index[1], w)
deg_inv_sqrt = deg.pow(-0.5)
deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
edge_weight_norm = deg_inv_sqrt[edge_index[0]] * w * deg_inv_sqrt[edge_index[1]]

# 6. Define GAE with three GraphConv layers, dropout, larger dims
class QoSGAE(nn.Module):
    def __init__(self, in_channels, hidden_channels, latent_channels, dropout=0.5):
        super().__init__()
        self.conv1 = GraphConv((in_channels, in_channels), hidden_channels)
        self.conv2 = GraphConv((hidden_channels, hidden_channels), latent_channels)
        self.conv3 = GraphConv((latent_channels, latent_channels), latent_channels)
        self.dropout = dropout
        self.decoder = nn.Sequential(
            nn.Linear(latent_channels * 2, hidden_channels),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(hidden_channels, 1)
        )

    def encode(self, x, edge_index, edge_weight_norm):
        h = self.conv1(x, edge_index, edge_weight=edge_weight_norm)
        h = F.relu(h)
        h = F.dropout(h, p=self.dropout, training=self.training)
        h = self.conv2(h, edge_index, edge_weight=edge_weight_norm)
        h = F.relu(h)
        h = F.dropout(h, p=self.dropout, training=self.training)
        h = self.conv3(h, edge_index, edge_weight=edge_weight_norm)
        return h  # [num_nodes, latent_channels]

    def decode(self, z, edge_index):
        src = edge_index[0, :num_obs]
        dst = edge_index[1, :num_obs]
        h_src = z[src]
        h_dst = z[dst]
        h_cat = torch.cat([h_src, h_dst], dim=1)
        return self.decoder(h_cat).squeeze()

    def forward(self, x, edge_index, edge_weight_norm):
        z = self.encode(x, edge_index, edge_weight_norm)
        pred_full = torch.zeros(edge_index.size(1), device=z.device)
        pred = self.decode(z, edge_index)
        pred_full[:num_obs] = pred
        pred_full[num_obs:2*num_obs] = pred
        return pred_full, z

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Move data to device
data = data.to(device)
edge_index = data.edge_index.to(device)
edge_weight_norm = edge_weight_norm.to(device)
rt_all = rt_all.to(device)

train_mask = data.train_mask.to(device)
val_mask   = data.val_mask.to(device)
test_mask  = data.test_mask.to(device)

rt_train = rt_all[train_mask]
rt_val   = rt_all[val_mask]
rt_test  = rt_all[test_mask]

# Config 8: Aggressive learning (from the original tuner.py)
config = {"hidden_channels": 192, "latent_channels": 96, "dropout": 0.3, "lr": 8e-4, "weight_decay": 1e-5, "patience": 8}

print(f"Training Configuration 8 - Aggressive Learning:")
print(f"Config: {config}")
print(f"{'='*60}")

# Initialize model with config 8
model = QoSGAE(
    in_channels=11, 
    hidden_channels=config["hidden_channels"], 
    latent_channels=config["latent_channels"], 
    dropout=config["dropout"]
).to(device)

optimizer = torch.optim.Adam(
    model.parameters(), 
    lr=config["lr"], 
    weight_decay=config["weight_decay"]
)
criterion = nn.MSELoss()

# Training tracking variables
train_losses = []
train_rmse_list = []
train_mae_list = []
val_rmse_list = []
val_mae_list = []
epochs_list = []

best_val_rmse = float("inf")
patience = config["patience"]
patience_counter = 0
best_epoch = 0

def train_epoch(model, optimizer, criterion):
    model.train()
    optimizer.zero_grad()
    pred_all, _ = model(data.x, edge_index, edge_weight_norm)
    loss = criterion(pred_all[train_mask], rt_train)
    loss.backward()
    optimizer.step()
    return loss.item()

@torch.no_grad()
def evaluate(model, mask, rt_target):
    model.eval()
    pred_all, _ = model(data.x, edge_index, edge_weight_norm)
    pred = pred_all[mask]
    mse = F.mse_loss(pred, rt_target).item()
    return float(np.sqrt(mse)), float(F.l1_loss(pred, rt_target).item())

# Training loop with detailed tracking
print("Starting training...")
for epoch in range(1, 201):
    loss = train_epoch(model, optimizer, criterion)
    train_rmse, train_mae = evaluate(model, train_mask, rt_train)
    val_rmse, val_mae = evaluate(model, val_mask, rt_val)
    
    # Store metrics for plotting
    train_losses.append(loss)
    train_rmse_list.append(train_rmse)
    train_mae_list.append(train_mae)
    val_rmse_list.append(val_rmse)
    val_mae_list.append(val_mae)
    epochs_list.append(epoch)
    
    if val_rmse < best_val_rmse:
        best_val_rmse = val_rmse
        torch.save(model.state_dict(), "best_gae_config8.pt")
        patience_counter = 0
        best_epoch = epoch
    else:
        patience_counter += 1

    if patience_counter >= patience:
        print(f"Early stopping at epoch {epoch} (no val_rmse improvement for {patience} epochs)")
        break

    if epoch % 10 == 0 or epoch <= 10:
        print(f"Epoch {epoch:03d} | Loss: {loss:.4f} | Train RMSE: {train_rmse:.4f} | Val RMSE: {val_rmse:.4f} | Train MAE: {train_mae:.4f} | Val MAE: {val_mae:.4f}")

# Load best model and evaluate on test set
model.load_state_dict(torch.load("best_gae_config8.pt", weights_only=False, map_location=device))
test_rmse, test_mae = evaluate(model, test_mask, rt_test)

print(f"\nConfig 8 Final Results:")
print(f"  Best Epoch: {best_epoch}")
print(f"  Best Val RMSE: {best_val_rmse:.4f}")
print(f"  Test RMSE: {test_rmse:.4f} | Test MAE: {test_mae:.4f}")

# Create charts directory if it doesn't exist
os.makedirs("charts", exist_ok=True)

# Plot 1: Training Loss
plt.figure(figsize=(12, 8))
plt.subplot(2, 2, 1)
plt.plot(epochs_list, train_losses, 'b-', linewidth=2, label='Training Loss')
plt.xlabel('Epoch')
plt.ylabel('MSE Loss')
plt.title('Training Loss Over Time (Config 8)')
plt.grid(True, alpha=0.3)
plt.legend()

# Plot 2: RMSE Comparison
plt.subplot(2, 2, 2)
plt.plot(epochs_list, train_rmse_list, 'b-', linewidth=2, label='Train RMSE')
plt.plot(epochs_list, val_rmse_list, 'r-', linewidth=2, label='Validation RMSE')
plt.axvline(x=best_epoch, color='g', linestyle='--', alpha=0.7, label=f'Best Epoch ({best_epoch})')
plt.xlabel('Epoch')
plt.ylabel('RMSE')
plt.title('RMSE Comparison (Config 8)')
plt.grid(True, alpha=0.3)
plt.legend()

# Plot 3: MAE Comparison
plt.subplot(2, 2, 3)
plt.plot(epochs_list, train_mae_list, 'b-', linewidth=2, label='Train MAE')
plt.plot(epochs_list, val_mae_list, 'r-', linewidth=2, label='Validation MAE')
plt.axvline(x=best_epoch, color='g', linestyle='--', alpha=0.7, label=f'Best Epoch ({best_epoch})')
plt.xlabel('Epoch')
plt.ylabel('MAE')
plt.title('MAE Comparison (Config 8)')
plt.grid(True, alpha=0.3)
plt.legend()

# Plot 4: Combined metrics
plt.subplot(2, 2, 4)
plt.plot(epochs_list, train_losses, 'b-', linewidth=1, alpha=0.7, label='Training Loss (scaled)')
# Scale loss to be comparable to RMSE for visualization
loss_scaled = np.array(train_losses) * (max(val_rmse_list) / max(train_losses))
plt.plot(epochs_list, loss_scaled, 'c--', linewidth=1, alpha=0.7, label='Training Loss (scaled)')
plt.plot(epochs_list, val_rmse_list, 'r-', linewidth=2, label='Validation RMSE')
plt.axvline(x=best_epoch, color='g', linestyle='--', alpha=0.7, label=f'Best Epoch ({best_epoch})')
plt.xlabel('Epoch')
plt.ylabel('Metric Value')
plt.title('Combined Training Metrics (Config 8)')
plt.grid(True, alpha=0.3)
plt.legend()

plt.tight_layout()
plt.savefig('charts/config8_training_curves.png', dpi=300, bbox_inches='tight')
plt.show()

# Create a detailed loss curve plot
plt.figure(figsize=(10, 6))
plt.plot(epochs_list, train_losses, 'b-', linewidth=2, label='Training Loss')
plt.axvline(x=best_epoch, color='g', linestyle='--', alpha=0.7, label=f'Best Epoch ({best_epoch})')
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('MSE Loss', fontsize=12)
plt.title('Training Loss Curve - Config 8 (Aggressive Learning)', fontsize=14)
plt.grid(True, alpha=0.3)
plt.legend(fontsize=11)
plt.savefig('charts/config8_loss_curve.png', dpi=300, bbox_inches='tight')
plt.show()

# Create a validation performance plot
plt.figure(figsize=(10, 6))
plt.plot(epochs_list, val_rmse_list, 'r-', linewidth=2, label='Validation RMSE')
plt.plot(epochs_list, val_mae_list, 'orange', linewidth=2, label='Validation MAE')
plt.axvline(x=best_epoch, color='g', linestyle='--', alpha=0.7, label=f'Best Epoch ({best_epoch})')
plt.axhline(y=best_val_rmse, color='r', linestyle=':', alpha=0.7, label=f'Best Val RMSE ({best_val_rmse:.4f})')
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Error Metric', fontsize=12)
plt.title('Validation Performance - Config 8 (Aggressive Learning)', fontsize=14)
plt.grid(True, alpha=0.3)
plt.legend(fontsize=11)
plt.savefig('charts/config8_validation_performance.png', dpi=300, bbox_inches='tight')
plt.show()

# Final evaluation and reconstruction
with torch.no_grad():
    pred_all, z = model(data.x, edge_index, edge_weight_norm)
    preds_norm = pred_all[:num_obs].cpu().numpy()

pred_matrix = np.zeros_like(rt_norm, dtype=np.float32)
idx = 0
for u in range(num_users):
    for s in np.where(observed_mask[u])[0]:
        pred_matrix[u, s] = preds_norm[idx]
        idx += 1

pred_matrix_denorm = pred_matrix * max_rt
reconstructed_rt = pred_matrix_denorm.copy()
reconstructed_rt[observed_mask] = rt[observed_mask]

actuals   = rt[observed_mask]
preds_obs = pred_matrix_denorm[observed_mask]
rmse_obs  = float(np.sqrt(np.mean((preds_obs - actuals)**2)))
mae_obs   = float(np.mean(np.abs(preds_obs - actuals)))

# Save results
np.savetxt("predicted_matrix_config8.txt", pred_matrix_denorm, fmt="%.4f")

print(f"\nFinal Observed-edge Reconstruction Errors (Config 8):")
print(f"  RMSE = {rmse_obs:.4f} | MAE = {mae_obs:.4f}")
print(f"\nSaved predicted matrix to: predicted_matrix_config8.txt")
print(f"Charts saved to charts/ directory:")
print(f"  - config8_training_curves.png (comprehensive overview)")
print(f"  - config8_loss_curve.png (training loss)")
print(f"  - config8_validation_performance.png (validation metrics)")

# Save training history to file
training_history = {
    'epoch': epochs_list,
    'train_loss': train_losses,
    'train_rmse': train_rmse_list,
    'train_mae': train_mae_list,
    'val_rmse': val_rmse_list,
    'val_mae': val_mae_list,
    'best_epoch': best_epoch,
    'best_val_rmse': best_val_rmse,
    'test_rmse': test_rmse,
    'test_mae': test_mae,
    'config': config
}

import pickle
with open('config8_training_history.pkl', 'wb') as f:
    pickle.dump(training_history, f)

print(f"Training history saved to: config8_training_history.pkl")

# Create summary text file
with open("config8_results_summary.txt", "w") as f:
    f.write("Config 8 Training Results - Aggressive Learning\n")
    f.write("=" * 50 + "\n\n")
    f.write(f"Configuration: {config}\n\n")
    f.write(f"Training Summary:\n")
    f.write(f"  Total Epochs: {len(epochs_list)}\n")
    f.write(f"  Best Epoch: {best_epoch}\n")
    f.write(f"  Best Validation RMSE: {best_val_rmse:.4f}\n")
    f.write(f"  Final Test RMSE: {test_rmse:.4f}\n")
    f.write(f"  Final Test MAE: {test_mae:.4f}\n\n")
    f.write(f"Observed-edge Reconstruction:\n")
    f.write(f"  RMSE: {rmse_obs:.4f}\n")
    f.write(f"  MAE: {mae_obs:.4f}\n\n")
    f.write(f"Files Generated:\n")
    f.write(f"  - best_gae_config8.pt (best model weights)\n")
    f.write(f"  - predicted_matrix_config8.txt (predictions)\n")
    f.write(f"  - config8_training_history.pkl (complete history)\n")
    f.write(f"  - charts/config8_*.png (training curves)\n")

print("Summary saved to: config8_results_summary.txt")
print("\nTraining complete for Config 8!")
