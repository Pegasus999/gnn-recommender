import numpy as np

# 1. Load the original RT matrix and the reconstructed matrix
rt          = np.loadtxt("rtMatrix.txt")             # shape [num_users, num_services]
recon_rt    = np.loadtxt("predicted_full_matrix.txt")# same shape

# 2. Get all observed (non‐zero) RTs
observed_mask = (rt > 0)
observed_rt   = rt[observed_mask]    # 1D array of all real RT values

# 3. Compute baseline: global mean & std
mu  = observed_rt.mean()
std = observed_rt.std()
print(f"Global mean RT μ = {mu:.4f} seconds")
print(f"Global std  RT σ = {std:.4f} seconds\n")

# 4. Baseline RMSE (predict μ everywhere, compare only on observed entries)
pred_baseline = np.full_like(observed_rt, mu)
rmse_baseline = np.sqrt(np.mean((pred_baseline - observed_rt)**2))
mae_baseline  = np.mean(np.abs(pred_baseline - observed_rt))
print(f"Baseline (predict μ) → RMSE = {rmse_baseline:.4f}, MAE = {mae_baseline:.4f}\n")

# 5. Your model’s error on observed entries:
preds_obs = recon_rt[observed_mask]
rmse_model = np.sqrt(np.mean((preds_obs - observed_rt)**2))
mae_model  = np.mean(np.abs(preds_obs - observed_rt))
print(f"Model        → RMSE = {rmse_model:.4f}, MAE = {mae_model:.4f}\n")

# 6. Relative error metrics (normalized by μ and σ)
nrmse_mu  = rmse_model / mu  if mu!=0 else float('inf')
nrmse_std = rmse_model / std if std!=0 else float('inf')
print(f"Normalized RMSE by μ:  {nrmse_mu:.4f}")
print(f"Normalized RMSE by σ:  {nrmse_std:.4f}")

