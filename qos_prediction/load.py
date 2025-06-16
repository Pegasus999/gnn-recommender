import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GATConv
from qos import QoSGAE


# 2. Load original RT matrix and rebuild the PyG Data object
rt = np.loadtxt("rtMatrix.txt")               # shape: [num_users, num_services]
num_users, num_services = rt.shape
observed_mask = (rt > 0)

# Build edge_index and edge_attr from observed entries
edge_src = []
edge_dst = []
edge_attr = []
for u in range(num_users):
    for s in np.where(observed_mask[u])[0]:
        edge_src.append(u)
        edge_dst.append(num_users + s)
        edge_attr.append([rt[u, s]])

edge_index = torch.tensor([edge_src, edge_dst], dtype=torch.long)
edge_attr  = torch.tensor(edge_attr, dtype=torch.float)

# Create identity features for all nodes
num_nodes = num_users + num_services
x = torch.eye(num_nodes, dtype=torch.float)

# You can reuse the same train/val/test splits, 
# but for final reconstruction+error we only need observed edges.
data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
data.num_users    = num_users
data.num_services = num_services

# 3. Load saved model weights
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = QoSGAE(
    num_nodes=num_nodes,
    in_channels=num_nodes,
    hidden_channels=64,      # must match training
    latent_channels=32       # must match training
).to(device)

model.load_state_dict(torch.load("best_gae.pt", map_location=device))
model.eval()

# 4. Compute embeddings and predicted RT for all observed edges
data = data.to(device)
edge_idx_all = data.edge_index
rt_all_tensor = data.edge_attr.squeeze().to(device)

with torch.no_grad():
    _, z = model(data.x.to(device), edge_idx_all)
    pred_all = model.decode(z, edge_idx_all)  # [num_edges]

# 5. Reconstruct full RT matrix (user × service)
# Build all possible user→service edges
users = torch.arange(num_users, device=device)
services = torch.arange(num_users, num_users + num_services, device=device)
uu, ss = torch.meshgrid(users, services, indexing="ij")
full_edges = torch.stack([uu.flatten(), ss.flatten()], dim=0)  # [2, num_users*num_services]

with torch.no_grad():
    preds_full = model.decode(z, full_edges).cpu().numpy()  # [num_users*num_services]

pred_matrix = preds_full.reshape(num_users, num_services)

# 6. For observed entries, compute error metrics
#    Note: we compare pred_matrix[u,s] vs rt[u,s] only where rt>0
mask = observed_mask
actuals   = rt[mask]
preds_obs = pred_matrix[mask]

mse  = np.mean((preds_obs - actuals) ** 2)
rmse = np.sqrt(mse)
mae  = np.mean(np.abs(preds_obs - actuals))

print(f"Observed-edge Reconstruction Errors:")
print(f"  RMSE = {rmse:.4f}")
print(f"  MAE  = {mae:.4f}")

# 7. Save the reconstructed matrix (optional)
reconstructed_rt = pred_matrix.copy()
reconstructed_rt[mask] = rt[mask]  # keep original for observed
np.savetxt("reconstructed_rtMatrix.txt", reconstructed_rt, fmt="%.4f")

# 8. If you want to inspect a specific user's first 10 services:
user_id = 5  # replace with desired user index
print(f"\nUser {user_id} | service_id | Actual | Predicted")
for service_id in range(10):
    actual    = rt[user_id, service_id]
    predicted = pred_matrix[user_id, service_id]
    print(f"  {service_id:02d}       | {actual:>6.4f} | {predicted:>6.4f}")

