import numpy as np
import pandas as pd

# === Load data ===
try:
    pred_matrix = np.loadtxt("predicted.txt")  # Your model's prediction
    rt_matrix = np.loadtxt("rtMatrix.txt")                      # Original matrix with known values
    
    # Validate matrix dimensions
    if pred_matrix.shape != rt_matrix.shape:
        raise ValueError("Prediction and original matrices must have the same dimensions")
        
except FileNotFoundError as e:
    print(f"Error: Could not find required data files. {e}")
    exit(1)
except Exception as e:
    print(f"Error loading data: {e}")
    exit(1)

def recommend_services(user_id: int, top_n: int = 5):
    """
    Recommend top-N services with lowest predicted response time
    for a given user (excluding already known interactions).
    """
    # Input validation
    if user_id < 0 or user_id >= pred_matrix.shape[0]:
        raise ValueError(f"User ID {user_id} is out of range. Valid range: 0-{pred_matrix.shape[0]-1}")
    
    if top_n <= 0:
        raise ValueError("top_n must be a positive integer")
    
    user_pred = pred_matrix[user_id]
    user_actual = rt_matrix[user_id]

    # Debug information
    print(f"Debug: User {user_id} original values range: {user_actual.min():.3f} to {user_actual.max():.3f}")
    print(f"Debug: User {user_id} predicted values range: {user_pred.min():.3f} to {user_pred.max():.3f}")
    print(f"Debug: Services marked as -1 (unused): {np.sum(user_actual == -1)}")
    print(f"Debug: Services with predictions > 0: {np.sum(user_pred > 0)}")

    # Mask services the user hasn't used (marked as -1 in original)
    mask = user_actual == -1
    filtered_preds = np.where(mask, user_pred, np.inf)
    
    # Remove services with 0 or negative predictions
    valid_pred_mask = filtered_preds > 0
    filtered_preds = np.where(valid_pred_mask, filtered_preds, np.inf)
    
    # Check if there are any services to recommend
    available_services = np.sum(filtered_preds != np.inf)
    if available_services == 0:
        print("Debug: No valid services found (all predictions are 0 or negative)")
        return pd.DataFrame(columns=["Rank", "Service ID", "Predicted Response Time (s)"])
    
    # Adjust top_n if there aren't enough services available
    actual_top_n = min(top_n, available_services)

    # Get top-N service indices with lowest predicted RT
    top_indices = np.argsort(filtered_preds)[:actual_top_n]
    top_scores = filtered_preds[top_indices]
    
    # Filter out infinite values
    valid_mask = top_scores != np.inf
    top_indices = top_indices[valid_mask]
    top_scores = top_scores[valid_mask]

    # Return results as a table
    return pd.DataFrame({
        "Rank": np.arange(1, len(top_indices) + 1),
        "Service ID": top_indices,
        "Predicted Response Time (s)": top_scores
    })

# === Example usage ===
if __name__ == "__main__":
    user_id = 35  
    top_n = 5

    try:
        recommendations = recommend_services(user_id, top_n)
        if recommendations.empty:
            print(f"\nNo services available to recommend for User {user_id} (no unused services found).")
        else:
            print(f"\nTop {len(recommendations)} Recommended Services for User {user_id}:\n")
            print(recommendations)
    except Exception as e:
        print(f"Error generating recommendations: {e}")
