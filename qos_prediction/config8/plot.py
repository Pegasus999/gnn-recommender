import numpy as np
import matplotlib.pyplot as plt

# ─── Replace these with your actual file paths ─────────────────────────────────
# Original QoS matrix (e.g., rtMatrix.txt)
original_matrix = np.loadtxt('rtMatrix.txt')

predicted_matrix = np.loadtxt('predicted_matrix.txt')

# ─── Specify the user ID (0-based index) ────────────────────────────────────────
user_id = 25  # ← change to whichever user you want to inspect

# ─── Extract the first 10 services for that user ─────────────────────────────────
actual   = original_matrix[user_id, :10]
predicted = predicted_matrix[user_id, :10]

# ─── Plotting ───────────────────────────────────────────────────────────────────
services = np.arange(1, 11)            # Services 1 through 10
x = np.arange(len(services))
width = 0.35

fig, ax = plt.subplots()
ax.bar(x - width/2, actual,   width, label='Actual')
ax.bar(x + width/2, predicted, width, label='Predicted')

ax.set_xlabel('Service Index (first 10)')
ax.set_ylabel('QoS Value')
ax.set_title(f'User {user_id} QoS: Actual vs. Predicted\n(first 10 services)')
ax.set_xticks(x)
ax.set_xticklabels(services)
ax.legend()

plt.tight_layout()
plt.show()

