#!/usr/bin/env python3
"""
Utilities for API Recommendation System

This module contains shared utility functions and classes used across the project.
"""

import torch
import numpy as np
import random
import matplotlib.pyplot as plt
import seaborn as sns
import os
from collections import defaultdict, Counter
from sklearn.metrics import (
    roc_auc_score, average_precision_score, 
    accuracy_score, precision_score, recall_score
)
from tqdm import tqdm
import torch.nn.functional as F

# Set random seeds for reproducibility
def set_seed(seed=42):
    """Set random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class DataBalancer:
    """Handles data balancing for API edge distribution"""
    
    def __init__(self, data_path: str = "dataset.pt"):
        self.data = torch.load(data_path, weights_only=False)
        self.analyze_current_balance()
    
    def analyze_current_balance(self):
        """Analyzes current edge distribution balance"""
        print("\nAnalyzing current data balance...")
        
        edge_index = self.data['mashup', 'uses', 'api'].edge_index
        mashup_ids = edge_index[0].cpu().numpy()
        api_ids = edge_index[1].cpu().numpy()
        
        # Count edges per API
        api_edge_counts = Counter(api_ids)
        
        print("Edge distribution statistics:")
        counts_array = np.array(list(api_edge_counts.values()))
        print(f"   Mean edges per API: {counts_array.mean():.2f}")
        print(f"   Median edges per API: {np.median(counts_array):.2f}")
        print(f"   Std dev: {counts_array.std():.2f}")
        print(f"   Min edges: {counts_array.min()}")
        print(f"   Max edges: {counts_array.max()}")
        print(f"   APIs with >40 edges: {np.sum(counts_array > 40)}")
        print(f"   APIs with >20 edges: {np.sum(counts_array > 20)}")
        
        # Count edges per mashup
        mashup_edge_counts = Counter(mashup_ids)
        mashup_counts_array = np.array(list(mashup_edge_counts.values()))
        
        print("\nMashup distribution statistics:")
        print(f"   Mean edges per mashup: {mashup_counts_array.mean():.2f}")
        print(f"   Median edges per mashup: {np.median(mashup_counts_array):.2f}")
        print(f"   Max edges per mashup: {mashup_counts_array.max()}")
        
        self.edge_stats = {
            'api_counts': api_edge_counts,
            'mashup_counts': mashup_edge_counts,
            'api_stats': {
                'mean': counts_array.mean(),
                'median': np.median(counts_array),
                'std': counts_array.std(),
                'max': counts_array.max()
            }
        }
    
    def create_balanced_splits(self, val_ratio: float = 0.1, test_ratio: float = 0.1, 
                              max_edges_per_api: int = 20, min_edges_per_api: int = 2):
        """Creates balanced train/validation/test splits with controlled API edge distribution"""
        print(f"\nCreating balanced splits (max {max_edges_per_api} edges per API)...")
        
        edge_index = self.data['mashup', 'uses', 'api'].edge_index
        edges = edge_index.t().cpu().numpy()  # Convert to (num_edges, 2)
        
        # Group edges by API
        api_edges = defaultdict(list)
        for i, (mashup_id, api_id) in enumerate(edges):
            api_edges[api_id].append(i)
        
        # Apply balanced sampling per API
        balanced_edge_indices = []
        
        for api_id, edge_indices in api_edges.items():
            if len(edge_indices) >= min_edges_per_api:
                if len(edge_indices) > max_edges_per_api:
                    # Randomly sample edges for this API
                    sampled = random.sample(edge_indices, max_edges_per_api)
                    balanced_edge_indices.extend(sampled)
                else:
                    balanced_edge_indices.extend(edge_indices)
        
        print(f"Balanced edges: {len(edges)} → {len(balanced_edge_indices)} (reduction: {(1-len(balanced_edge_indices)/len(edges))*100:.1f}%)")
        
        # Shuffle balanced edges
        random.shuffle(balanced_edge_indices)
        balanced_edges = edges[balanced_edge_indices]
        
        # Create splits
        num_edges = len(balanced_edges)
        num_val = int(val_ratio * num_edges)
        num_test = int(test_ratio * num_edges)
        num_train = num_edges - num_val - num_test
        
        train_edges = torch.tensor(balanced_edges[:num_train].T, dtype=torch.long)
        val_edges = torch.tensor(balanced_edges[num_train:num_train + num_val].T, dtype=torch.long)
        test_edges = torch.tensor(balanced_edges[num_train + num_val:].T, dtype=torch.long)
        
        print(f"Split sizes - Train: {train_edges.size(1)}, Val: {val_edges.size(1)}, Test: {test_edges.size(1)}")
        
        return train_edges, val_edges, test_edges


def create_negative_edges_balanced(data, pos_edges: torch.Tensor, num_neg: int = None, device: str = 'cpu'):
    """Create balanced negative edges"""
    if num_neg is None:
        num_neg = pos_edges.size(1)
    
    num_mashups = data['mashup'].x.size(0)
    num_apis = data['api'].x.size(0)
    
    # Create set of positive edges for fast lookup
    pos_edge_set = set(map(tuple, pos_edges.t().cpu().numpy()))
    
    # Sample negative edges with API balancing
    api_neg_count = defaultdict(int)
    max_neg_per_api = max(5, num_neg // num_apis * 2)  # Limit negatives per API
    
    neg_edges = []
    attempts = 0
    max_attempts = num_neg * 10
    
    while len(neg_edges) < num_neg and attempts < max_attempts:
        mashup_id = random.randint(0, num_mashups - 1)
        api_id = random.randint(0, num_apis - 1)
        
        edge = (mashup_id, api_id)
        if edge not in pos_edge_set and api_neg_count[api_id] < max_neg_per_api:
            neg_edges.append(edge)
            api_neg_count[api_id] += 1
        
        attempts += 1
    
    if len(neg_edges) < num_neg:
        print(f"⚠️ Could only generate {len(neg_edges)} negative edges (requested {num_neg})")
    
    return torch.tensor(neg_edges, dtype=torch.long, device=device).t()


def prepare_data_splits_balanced(data, device='cpu', max_edges_per_api=20):
    """
    Split edges using the balanced approach
    """
    # Create data balancer
    balancer = DataBalancer()
    balancer.data = data
    
    # Create balanced splits
    train_edges, val_edges, test_edges = balancer.create_balanced_splits(
        max_edges_per_api=max_edges_per_api
    )
    
    # Move to device
    train_edges = train_edges.to(device)
    val_edges = val_edges.to(device)
    test_edges = test_edges.to(device)
    
    # Create data objects for training
    train_data = data.clone()
    val_data = data.clone()
    test_data = data.clone()
    
    # Set training edges
    train_data['mashup', 'uses', 'api'].edge_index = train_edges
    train_data['api', 'rev_uses', 'mashup'].edge_index = train_edges[[1, 0]]
    
    # For validation and test, use training edges for message passing
    val_data['mashup', 'uses', 'api'].edge_index = train_edges
    val_data['api', 'rev_uses', 'mashup'].edge_index = train_edges[[1, 0]]
    
    test_data['mashup', 'uses', 'api'].edge_index = train_edges
    test_data['api', 'rev_uses', 'mashup'].edge_index = train_edges[[1, 0]]
    
    print(f"Data splits created:")
    print(f"  Train edges: {train_edges.size(1)}")
    print(f"  Val edges: {val_edges.size(1)}")
    print(f"  Test edges: {test_edges.size(1)}")
    
    return train_data, val_data, test_data, train_edges, val_edges, test_edges


def train_epoch(model, data, train_edges, optimizer, device):
    """Performs a single training epoch including forward pass, loss computation and optimization"""
    model.train()
    optimizer.zero_grad()
    
    # Create training graph
    train_data = data.clone()
    train_data['mashup', 'uses', 'api'].edge_index = train_edges
    train_data['api', 'rev_uses', 'mashup'].edge_index = train_edges[[1, 0]]
    
    # Forward pass
    x_dict = {node_type: x.to(device) for node_type, x in train_data.x_dict.items()}
    edge_index_dict = {edge_type: edge_index.to(device) 
                       for edge_type, edge_index in train_data.edge_index_dict.items()}
    
    # Get embeddings
    z_dict = model(x_dict, edge_index_dict)
    
    # Create balanced positive and negative samples
    num_edges = train_edges.size(1)
    neg_edges = create_negative_edges_balanced(data, train_edges, num_edges, device)
    
    # Compute scores using the decode method from HeteroGNN
    pos_scores = model.decode(z_dict['mashup'], z_dict['api'], train_edges)
    neg_scores = model.decode(z_dict['mashup'], z_dict['api'], neg_edges)
    
    # Binary classification loss
    scores = torch.cat([pos_scores, neg_scores])
    labels = torch.cat([torch.ones_like(pos_scores), torch.zeros_like(neg_scores)])
    
    loss = F.binary_cross_entropy_with_logits(scores, labels)
    
    # Add L2 regularization
    l2_reg = 0
    for param in model.parameters():
        l2_reg += torch.norm(param)
    loss += 1e-5 * l2_reg
    
    # Backward pass
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
    
    # Compute prediction statistics
    with torch.no_grad():
        probs = torch.sigmoid(scores)
        pred_mean = probs.mean().item()
        pred_std = probs.std().item()
        pred_min = probs.min().item()
        pred_max = probs.max().item()
        pos_pred_mean = probs[:len(pos_scores)].mean().item() if len(pos_scores) > 0 else 0
        neg_pred_mean = probs[len(pos_scores):].mean().item() if len(neg_scores) > 0 else 0
    
    return loss.item(), {
        'pred_mean': pred_mean,
        'pred_std': pred_std,
        'pred_min': pred_min,
        'pred_max': pred_max,
        'pos_pred_mean': pos_pred_mean,
        'neg_pred_mean': neg_pred_mean
    }


def evaluate(model, data, train_edges, eval_edges, device):
    """Evaluation with metrics"""
    model.eval()
    
    with torch.no_grad():
        # Create evaluation graph (without evaluation edges)
        eval_data = data.clone()
        eval_data['mashup', 'uses', 'api'].edge_index = train_edges
        eval_data['api', 'rev_uses', 'mashup'].edge_index = train_edges[[1, 0]]
        
        # Get embeddings
        x_dict = {node_type: x.to(device) for node_type, x in eval_data.x_dict.items()}
        edge_index_dict = {edge_type: edge_index.to(device) 
                           for edge_type, edge_index in eval_data.edge_index_dict.items()}
        
        z_dict = model(x_dict, edge_index_dict)
        
        # Generate negative edges for evaluation
        neg_edges = create_negative_edges_balanced(data, eval_edges, eval_edges.size(1), device)
        
        # Compute scores
        pos_scores = model.decode(z_dict['mashup'], z_dict['api'], eval_edges)
        neg_scores = model.decode(z_dict['mashup'], z_dict['api'], neg_edges)
        
        # Combine scores and labels
        scores = torch.cat([pos_scores, neg_scores])
        labels = torch.cat([
            torch.ones(pos_scores.size(0), device=device),
            torch.zeros(neg_scores.size(0), device=device)
        ])
        
        # Convert to probabilities
        probs = torch.sigmoid(scores).cpu().numpy()
        labels_np = labels.cpu().numpy()
        
        # Compute metrics
        auc = roc_auc_score(labels_np, probs)
        ap = average_precision_score(labels_np, probs)
        
        # Additional metrics
        pred_labels = (probs > 0.5).astype(int)
        accuracy = accuracy_score(labels_np, pred_labels)
        precision = precision_score(labels_np, pred_labels, zero_division=0)
        recall = recall_score(labels_np, pred_labels, zero_division=0)
        
        return {
            'auc': auc,
            'ap': ap,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall
        }


def plot_training_curves(train_losses, val_aucs, val_aps, pred_stats=None, config=None, save_path_prefix="training_curves"):
    """Plot training curves"""
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        if pred_stats and any(len(v) > 0 for k, v in pred_stats.items()):
            # Plot with three subplots when we have prediction statistics
            fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 6))
        else:
            # Original two subplot layout
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Loss curve
        ax1.plot(train_losses, label='Training Loss', color='blue', linewidth=2)
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training Loss')
        ax1.legend()
        ax1.grid(True, linestyle='--', alpha=0.7)
        
        # Add smoothed loss if enough data points
        if len(train_losses) > 10:
            alpha = 0.3
            smoothed_losses = []
            smoothed = train_losses[0]
            for loss in train_losses:
                smoothed = alpha * loss + (1 - alpha) * smoothed
                smoothed_losses.append(smoothed)
            ax1.plot(smoothed_losses, label='Smoothed Loss', color='red', linestyle='--', linewidth=2)
            ax1.legend()
        
        # Metrics curve
        epochs = range(len(val_aucs))
            
        ax2.plot(epochs, val_aucs, label='Validation AUC', marker='.', linewidth=2, color='green')
        ax2.plot(epochs, val_aps, label='Validation AP', marker='.', linewidth=2, color='purple')
        ax2.set_xlabel('Epoch') 
        ax2.set_ylabel('Score')
        ax2.set_title('Validation Metrics')
        ax2.legend()
        ax2.grid(True, linestyle='--', alpha=0.7)
        
        # Plot prediction statistics if available
        if pred_stats and any(len(v) > 0 for k, v in pred_stats.items()):
            epochs = range(len(pred_stats['pred_mean']))
            
            # Plot prediction means for positive and negative samples
            ax3.plot(epochs, pred_stats['pos_pred_mean'], label='Pos Pred Mean', 
                    color='green', linewidth=2)
            ax3.plot(epochs, pred_stats['neg_pred_mean'], label='Neg Pred Mean', 
                    color='red', linewidth=2)
            ax3.plot(epochs, pred_stats['pred_mean'], label='Overall Mean', 
                    color='blue', linewidth=1, linestyle='--')
                    
            # Add ideal line for reference (perfect separation)
            ax3.axhline(y=0.5, color='gray', linestyle=':', alpha=0.7)
            
            ax3.set_ylim(-0.1, 1.1)  # Ensure we see the full 0-1 range
            ax3.set_xlabel('Epoch')
            ax3.set_ylabel('Prediction Value')
            ax3.set_title('Prediction Statistics')
            ax3.legend()
            ax3.grid(True, linestyle='--', alpha=0.7)
        
        # Add config info if provided
        if config:
            config_text = f"Config: lr={config['lr']}, dim={config['hidden_dim']}, layers={config['num_layers']}"
            if 'dropout' in config:
                config_text += f", dropout={config['dropout']}"
            if 'use_batch_norm' in config:
                config_text += f", BN={config['use_batch_norm']}"
            fig.suptitle(config_text, fontsize=12)
        
        plt.tight_layout()
        plot_filename = f"{save_path_prefix}.png"
        plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
        print(f"Training curves saved to {plot_filename}")
        plt.close(fig)
        
        # Plot additional prediction stats in a separate figure
        if pred_stats and any(len(v) > 0 for k, v in pred_stats.items()):
            fig, ax = plt.subplots(figsize=(10, 6))
            epochs = range(len(pred_stats['pred_std']))
            
            ax.plot(epochs, pred_stats['pred_std'], label='Pred Std Dev', 
                   color='orange', linewidth=2)
            ax.plot(epochs, pred_stats['pred_max'], label='Pred Max', 
                   color='green', linewidth=1, alpha=0.7)
            ax.plot(epochs, pred_stats['pred_min'], label='Pred Min', 
                   color='red', linewidth=1, alpha=0.7)
            
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Value')
            ax.set_title('Prediction Distribution Statistics')
            ax.legend()
            ax.grid(True, linestyle='--', alpha=0.7)
            
            plt.tight_layout()
            stats_filename = f"{save_path_prefix}_pred_stats.png"
            plt.savefig(stats_filename, dpi=300, bbox_inches='tight')
            print(f"Prediction statistics plot saved to {stats_filename}")
            plt.close(fig)
        
    except Exception as e:
        print(f"Error plotting training curves: {e}")
        plt.close('all')


def save_embeddings(model, data, device, save_path='node_embeddings.pt'):
    """Saves learned node embeddings to a file"""
    try:
        model.eval()
        with torch.no_grad():
            x_dict = {node_type: x.to(device) for node_type, x in data.x_dict.items()}
            edge_index_dict = {edge_type: edge_index.to(device) 
                               for edge_type, edge_index in data.edge_index_dict.items()}
            
            embeddings = model(x_dict, edge_index_dict)
            
            # Convert to CPU and save
            embeddings_cpu = {node_type: emb.cpu() for node_type, emb in embeddings.items()}
            torch.save(embeddings_cpu, save_path)
            print(f"Node embeddings saved to {save_path}")
            
    except Exception as e:
        print(f"Error saving embeddings: {e}")

def plot_hyperparameter_analysis(results_df, save_dir='hetero_gnn_results/plots'):
    """Create plots analyzing hyperparameter performance"""
    try:
        os.makedirs(save_dir, exist_ok=True)
        
        # Set plotting style with fallback
        try:
            plt.style.use('seaborn-v0_8')
        except:
            try:
                plt.style.use('seaborn')
            except:
                pass  # Use default style
        
        hyperparams = ['lr', 'hidden_dim', 'num_layers', 'dropout', 'weight_decay', 'heads', 'max_edges_per_api', 'use_lr_scheduler']
        metrics = ['best_val_auc', 'test_auc', 'test_ap']
        
        for param in hyperparams:
            if param not in results_df.columns or results_df[param].isna().all():
                continue
                
            try:
                plt.figure(figsize=(10, 6))
                
                if param in ['use_lr_scheduler']:
                    # Categorical parameters
                    df_grouped = results_df.groupby(param)[metrics].mean().reset_index()
                    
                    if len(df_grouped) > 0:
                        x = np.arange(len(df_grouped[param]))
                        width = 0.25
                        
                        fig, ax = plt.subplots(figsize=(12, 7))
                        ax.bar(x - width, df_grouped['best_val_auc'], width, label='Val AUC', color='skyblue')
                        ax.bar(x, df_grouped['test_auc'], width, label='Test AUC', color='lightgreen')
                        ax.bar(x + width, df_grouped['test_ap'], width, label='Test AP', color='salmon')
                        
                        ax.set_xlabel(f'{param.replace("_", " ").title()}')
                        ax.set_ylabel('Score')
                        ax.set_title(f'Performance by {param.replace("_", " ").title()}')
                        ax.set_xticks(x)
                        ax.set_xticklabels(df_grouped[param])
                        ax.legend()
                        ax.grid(True, linestyle='--', alpha=0.3)
                        
                else:
                    # Numerical parameters
                    if results_df[param].nunique() <= 1:
                        continue
                        
                    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
                    
                    for i, metric in enumerate(metrics):
                        try:
                            sns.boxplot(x=param, y=metric, data=results_df, ax=axes[i])
                            axes[i].set_title(f'{metric.replace("_", " ").title()} vs {param.replace("_", " ").title()}')
                            axes[i].set_xlabel(param.replace("_", " ").title())
                            axes[i].set_ylabel(metric.replace("_", " ").title())
                            axes[i].grid(True, linestyle='--', alpha=0.3)
                        except Exception as e:
                            print(f"Error creating boxplot for {param}-{metric}: {e}")
                            continue
                
                plt.tight_layout()
                plt.savefig(f"{save_dir}/hyperparam_{param}_analysis.png", dpi=300, bbox_inches='tight')
                plt.close()
                
            except Exception as e:
                print(f"Error creating plot for parameter {param}: {e}")
                plt.close('all')
                continue
        
        # Create correlation heatmap
        try:
            plt.figure(figsize=(12, 10))
            
            numerical_cols = results_df.select_dtypes(include=[np.number]).columns
            valid_cols = [col for col in numerical_cols 
                         if results_df[col].nunique() > 1 and col not in ['config_id', 'id']]
            
            if len(valid_cols) > 1:
                corr_matrix = results_df[valid_cols].corr()
                sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, fmt='.2f', 
                           linewidths=0.5, cbar_kws={"shrink": 0.8})
                plt.title('Hyperparameter Correlation Matrix', fontsize=16)
                plt.tight_layout()
                plt.savefig(f"{save_dir}/hyperparameter_correlation_matrix.png", dpi=300, bbox_inches='tight')
            
            plt.close()
            
        except Exception as e:
            print(f"Error creating correlation matrix: {e}")
            plt.close('all')
        
        # Create top configs plot
        try:
            top_n = min(10, len(results_df))
            if top_n > 0:
                top_configs = results_df.sort_values('test_auc', ascending=False).head(top_n).reset_index(drop=True)
                
                plt.figure(figsize=(15, 8))
                x = range(len(top_configs))
                width = 0.35
                
                plt.bar([i - width/2 for i in x], top_configs['test_auc'], width, label='Test AUC', color='lightgreen')
                plt.bar([i + width/2 for i in x], top_configs['test_ap'], width, label='Test AP', color='salmon')
                
                plt.xlabel('Configuration Rank')
                plt.ylabel('Score')
                plt.title('Top Performing Configurations')
                plt.xticks(x, [f"{i+1}" for i in x])
                plt.legend()
                plt.grid(True, linestyle='--', alpha=0.3)
                
                plt.tight_layout()
                plt.savefig(f"{save_dir}/top_configurations.png", dpi=300, bbox_inches='tight')
                plt.close()
                
        except Exception as e:
            print(f"Error creating top configurations plot: {e}")
            plt.close('all')
        
        print(f"Hyperparameter analysis plots saved to {save_dir}")
        
    except Exception as e:
        print(f"Error in hyperparameter analysis: {e}")
        plt.close('all')

def convert_to_serializable(obj):
    """Convert numpy types to Python native types for serialization"""
    if isinstance(obj, (np.integer, np.int64)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, (np.ndarray,)):
        return obj.tolist()
    elif isinstance(obj, (np.bool_)):
        return bool(obj)
    return obj

def convert_dataframe_values(df):
    """Convert all numpy types in a DataFrame to Python native types"""
    return df.applymap(lambda x: float(x) if isinstance(x, (np.floating, np.float32, np.float64)) 
                      else int(x) if isinstance(x, (np.integer, np.int64))
                      else bool(x) if isinstance(x, np.bool_)
                      else x)
