import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GraphConv

# Ensure this script runs in the same folder as:
# - rtMatrix.txt
# - best_gae_config8.pt

# ─── Step 1: Load & normalize the original rtMatrix ───
rt = np.loadtxt("rtMatrix.txt")              # shape = [num_users, num_services]
num_users, num_services = rt.shape
observed_mask = (rt > 0)

# Find the maximum observed RT to normalize
max_rt = float(rt[observed_mask].max())

# Build a normalized rt matrix (zeros where missing)
rt_norm = np.zeros_like(rt, dtype=np.float32)
rt_norm[observed_mask] = rt[observed_mask] / max_rt

# ─── Step 2: Build enhanced node features exactly as in training ───
num_nodes = num_users + num_services

# Preallocate arrays for node‐level statistics
degrees       = np.zeros(num_nodes, dtype=np.float32)
sum_rt        = np.zeros(num_nodes, dtype=np.float32)
sum_rt_squared= np.zeros(num_nodes, dtype=np.float32)
min_rt        = np.full(num_nodes, np.inf, dtype=np.float32)
max_rt_node   = np.zeros(num_nodes, dtype=np.float32)

is_user    = np.zeros(num_nodes, dtype=np.float32)
is_service = np.zeros(num_nodes, dtype=np.float32)

# Compute user‐node features
for u in range(num_users):
    is_user[u] = 1.0
    connected_services = np.where(observed_mask[u])[0]
    deg_u = len(connected_services)
    degrees[u] = deg_u
    if deg_u > 0:
        vals = rt_norm[u, connected_services]
        sum_rt[u] = vals.sum()
        sum_rt_squared[u] = (vals ** 2).sum()
        min_rt[u] = vals.min()
        max_rt_node[u] = vals.max()
    else:
        min_rt[u] = 0.0

# Compute service‐node features (shift index by num_users)
for s in range(num_services):
    idx = num_users + s
    is_service[idx] = 1.0
    connected_users = np.where(observed_mask[:, s])[0]
    deg_s = len(connected_users)
    degrees[idx] = deg_s
    if deg_s > 0:
        vals = rt_norm[connected_users, s]
        sum_rt[idx] = vals.sum()
        sum_rt_squared[idx] = (vals ** 2).sum()
        min_rt[idx] = vals.min()
        max_rt_node[idx] = vals.max()
    else:
        min_rt[idx] = 0.0

# Normalize degrees to [0,1]
max_deg = float(degrees.max())
if max_deg == 0:
    max_deg = 1.0
degrees_norm = degrees / max_deg

# Compute avg, variance, std, cv for each node
avg_rt = np.zeros_like(sum_rt)
variance_rt = np.zeros_like(sum_rt)
nonzero = (degrees > 0)
avg_rt[nonzero] = sum_rt[nonzero] / degrees[nonzero]
variance_rt[nonzero] = (sum_rt_squared[nonzero] / degrees[nonzero]) - (avg_rt[nonzero] ** 2)
variance_rt = np.maximum(variance_rt, 0.0)
std_rt = np.sqrt(variance_rt)

cv_rt = np.zeros_like(std_rt)
nonzero_avg = (avg_rt > 1e-8)
cv_rt[nonzero_avg] = std_rt[nonzero_avg] / avg_rt[nonzero_avg]

# Range feature (max – min)
range_rt = max_rt_node - min_rt
range_rt[~nonzero] = 0.0

# Connectivity ratio: for users = deg/num_services, for services = deg/num_users
connectivity_ratio = np.zeros(num_nodes, dtype=np.float32)
connectivity_ratio[:num_users] = degrees[:num_users] / num_services
connectivity_ratio[num_users:] = degrees[num_users:] / num_users

# Activity level = degree / max possible connections
activity_level = degrees / np.maximum(num_services, num_users)

# Stack features into an (num_nodes × 11) array:
# [is_user, is_service, degrees_norm, avg_rt, std_rt, cv_rt, min_rt, max_rt_node, range_rt, connectivity_ratio, activity_level]
x_np = np.stack([
    is_user,
    is_service,
    degrees_norm,
    avg_rt,
    std_rt,
    cv_rt,
    min_rt,
    max_rt_node,
    range_rt,
    connectivity_ratio,
    activity_level
], axis=1)

# Convert to a PyTorch tensor
x = torch.from_numpy(x_np).to(torch.float)

# ─── Step 3: Build edge_index & normalized edge weights for observed edges ───
edge_src   = []
edge_dst   = []
edge_weight = []

# For each observed (user, service) pair, add one directed edge
for u in range(num_users):
    for s in np.where(observed_mask[u])[0]:
        edge_src.append(u)
        edge_dst.append(num_users + s)
        edge_weight.append(rt_norm[u, s])

# Make edges bidirectional by duplicating
edge_index = torch.tensor([
    edge_src + edge_dst,
    edge_dst + edge_src
], dtype=torch.long)

# Duplicate weights for reverse edges
w = torch.tensor(edge_weight + edge_weight, dtype=torch.float)

# Compute symmetric normalization: Ā = D⁻½ A D⁻½
deg = torch.zeros(num_nodes, dtype=torch.float)
deg.index_add_(0, edge_index[0], w)
deg.index_add_(0, edge_index[1], w)
deg_inv_sqrt = deg.pow(-0.5)
deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
edge_weight_norm = deg_inv_sqrt[edge_index[0]] * w * deg_inv_sqrt[edge_index[1]]

# Wrap into a Data object if needed (not strictly necessary for inference here)
data = Data(x=x, edge_index=edge_index, edge_weight=edge_weight_norm)

# ─── Step 4: Define the QoSGAE model class (identical to training) ───
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
        return h  # returns [num_nodes, latent_channels]

    def decode_edge_list(self, z, edge_list):
        """
        Given z: [num_nodes, latent_channels]
        and edge_list: tensor of shape [2, N_pairs], returns decoded preds of shape [N_pairs].
        """
        src = edge_list[0]  # shape = [N_pairs]
        dst = edge_list[1]  # shape = [N_pairs]
        h_src = z[src]
        h_dst = z[dst]
        h_cat = torch.cat([h_src, h_dst], dim=1)  # [N_pairs, latent_channels*2]
        out = self.decoder(h_cat).squeeze()       # [N_pairs]
        return out

# ─── Step 5: Instantiate model, load weights, switch to eval ───
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

model = QoSGAE(
    in_channels   = 11,   # matches feature dimension
    hidden_channels = 192,
    latent_channels = 96,
    dropout       = 0.3
).to(device)

# Load the trained weights; adjust filename if yours differs
checkpoint_path = "best_gae_config8.pt"
model.load_state_dict(torch.load(checkpoint_path, map_location=device))
model.eval()

# Move node features & edge data to device
x = x.to(device)
edge_index = edge_index.to(device)
edge_weight_norm = edge_weight_norm.to(device)

# ─── Step 6: Run a forward pass to get latent embeddings ───
with torch.no_grad():
    z = model.encode(x, edge_index, edge_weight_norm)  # [num_nodes, latent_channels]

# ─── Step 7: Decode every (user, service) pair to reconstruct full matrix ───
# Build all U×S user–service pairs
user_ids    = torch.arange(num_users, device=device).unsqueeze(1).repeat(1, num_services).view(-1)
service_ids = (torch.arange(num_services, device=device) + num_users).unsqueeze(0).repeat(num_users, 1).view(-1)

# Concatenate embeddings for each pair and run through decoder
h_u_all = z[user_ids]         # [num_users*num_services, latent_channels]
h_s_all = z[service_ids]      # [num_users*num_services, latent_channels]
h_cat_all = torch.cat([h_u_all, h_s_all], dim=1)  # [num_users*num_services, latent_channels*2]

with torch.no_grad():
    preds_flat = model.decoder(h_cat_all).squeeze()  # [num_users*num_services]

# Reshape and de-normalize
preds_all_norm = preds_flat.view(num_users, num_services).cpu().numpy()
full_pred_matrix = preds_all_norm * max_rt

# ─── Step 8: (Optional) Overwrite observed slots with ground truth ───
full_pred_matrix[observed_mask] = rt[observed_mask]

# ─── Step 9: Save the reconstructed matrix ───
output_filename = "reconstructed_full_matrix_config8.txt"
np.savetxt(output_filename, full_pred_matrix, fmt="%.6f")
print(f"Reconstructed matrix saved to '{output_filename}'")
