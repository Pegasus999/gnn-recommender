import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from torch_geometric.data import Data
from torch_geometric.nn import GraphConv

# 1. Load & normalize RT
rt = np.loadtxt("rtMatrix.txt")                     # [num_users, num_services]
num_users, num_services = rt.shape
observed_mask = (rt > 0)

max_rt = float(rt[observed_mask].max())
rt_norm = np.zeros_like(rt, dtype=np.float32)
rt_norm[observed_mask] = rt[observed_mask] / max_rt

# 2. Build homogeneous graph: users 0..U-1, services U..U+S-1
num_nodes = num_users + num_services

# Compute raw degree & sum(rt_norm) per node
degrees = np.zeros(num_nodes, dtype=np.float32)
sum_rt   = np.zeros(num_nodes, dtype=np.float32)

for u in range(num_users):
    sel = np.where(observed_mask[u])[0]
    degrees[u] = len(sel)
    sum_rt[u]  = rt_norm[u, sel].sum() if len(sel) else 0.0

for s in range(num_services):
    sel = np.where(observed_mask[:, s])[0]
    idx = num_users + s
    degrees[idx] = len(sel)
    sum_rt[idx]  = rt_norm[sel, s].sum() if len(sel) else 0.0

# Normalize degree into [0,1]
max_deg = float(degrees.max())
if max_deg == 0:
    max_deg = 1.0
degrees_norm = degrees / max_deg

avg_rt = np.zeros_like(sum_rt)
nonzero = degrees > 0
avg_rt[nonzero] = sum_rt[nonzero] / degrees[nonzero]

# Node features: [degree_norm, avg_rt]
x_np = np.stack([degrees_norm, avg_rt], axis=1)
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
model = QoSGAE(in_channels=2, hidden_channels=128, latent_channels=64, dropout=0.5).to(device)

data = data.to(device)
edge_index = data.edge_index.to(device)
edge_weight_norm = edge_weight_norm.to(device)
rt_all = rt_all.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=5e-4, weight_decay=1e-4)
criterion = nn.MSELoss()

train_mask = data.train_mask.to(device)
val_mask   = data.val_mask.to(device)
test_mask  = data.test_mask.to(device)

rt_train = rt_all[train_mask]
rt_val   = rt_all[val_mask]
rt_test  = rt_all[test_mask]

# 7. Training loop with early stopping
best_val_rmse = float("inf")
patience = 10
patience_counter = 0

def train_epoch():
    model.train()
    optimizer.zero_grad()
    pred_all, _ = model(data.x, edge_index, edge_weight_norm)
    loss = criterion(pred_all[train_mask], rt_train)
    loss.backward()
    optimizer.step()
    return loss.item()

@torch.no_grad()
def evaluate(mask, rt_target):
    model.eval()
    pred_all, _ = model(data.x, edge_index, edge_weight_norm)
    pred = pred_all[mask]
    mse = F.mse_loss(pred, rt_target).item()
    return float(np.sqrt(mse)), float(F.l1_loss(pred, rt_target).item())

for epoch in range(1, 201):
    loss = train_epoch()
    val_rmse, val_mae = evaluate(val_mask, rt_val)
    if val_rmse < best_val_rmse:
        best_val_rmse = val_rmse
        torch.save(model.state_dict(), "best_gae.pt")
        patience_counter = 0
    else:
        patience_counter += 1

    if patience_counter >= patience:
        print(f"Early stopping at epoch {epoch} (no val_rmse improvement for {patience} epochs)")
        break

    if epoch % 10 == 0:
        train_rmse, train_mae = evaluate(train_mask, rt_train)
        print(f"Epoch {epoch:03d} | Loss: {loss:.4f} | Train RMSE: {train_rmse:.4f} | Val RMSE: {val_rmse:.4f}")

# 8. Load best model & test on held-out edges
model.load_state_dict(torch.load("best_gae.pt",weights_only=False, map_location=device))
test_rmse, test_mae = evaluate(test_mask, rt_test)
print(f"Final Test | RMSE: {test_rmse:.4f} | MAE: {test_mae:.4f}")

# 9. Reconstruct full RT matrix & compute observed-edge error
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
np.savetxt("predicted_full_matrix.txt", pred_matrix_denorm, fmt="%.4f")
print(f"Observed-edge Reconstruction Errors:\n  RMSE = {rmse_obs:.4f} | MAE = {mae_obs:.4f}")
