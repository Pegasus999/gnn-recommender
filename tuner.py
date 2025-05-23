import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import json
import pandas as pd
from tqdm import tqdm
from model import HeteroGNN
from utils import (
    set_seed, DataBalancer, create_negative_edges_balanced,
    prepare_data_splits_balanced, train_epoch, evaluate, 
    plot_training_curves, save_embeddings, plot_hyperparameter_analysis,
    convert_to_serializable, convert_dataframe_values
)

set_seed()

def run_experiment(config, data, device):
    """Runs a single experiment with the specified configuration"""
    try:
        print(f"\n--- Running Experiment with Config ID: {config['id']} ---")
        set_seed(42)

        # Use balanced data splits
        train_data, val_data, test_data, train_edges, val_edges, test_edges = prepare_data_splits_balanced(
            data, device, max_edges_per_api=config.get('max_edges_per_api', 20)
        )
        
        # Get input feature dimensions
        in_channels = data['mashup'].x.size(1)
        
        # Create HeteroGNN model
        model = HeteroGNN(
            in_channels=in_channels,
            hidden_channels=config['hidden_dim'],
            out_channels=config.get('out_channels', 128),
            num_layers=config['num_layers'],
            dropout=config.get('dropout', 0.2),
            num_heads=config.get('heads', 4)
        ).to(device)

        # Optimizer setup
        optimizer = torch.optim.AdamW(
            model.parameters(), 
            lr=config['lr'], 
            weight_decay=config.get('weight_decay', 1e-5),
            betas=(0.9, 0.999)
        )
        
        # Learning rate scheduler
        scheduler = None
        if config.get('use_lr_scheduler', False):
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='max', factor=0.7, patience=10, verbose=True
            )

        num_epochs = config.get('num_epochs', 100)
        patience = config.get('patience', 25)
        best_val_auc_experiment = 0
        patience_counter = 0
        best_epoch = 0

        train_losses_exp = []
        val_aucs_exp = []
        val_aps_exp = []
        
        # For tracking prediction statistics
        pred_stats_history = {
            'pred_mean': [],
            'pred_std': [],
            'pos_pred_mean': [],
            'neg_pred_mean': [],
            'pred_min': [],
            'pred_max': []
        }
        
        # Create checkpoint directory
        os.makedirs(f"hetero_gnn_results/models/checkpoints", exist_ok=True)
        model_checkpoint_path = f"hetero_gnn_results/models/checkpoints/model_{config['id']}.pt"

        for epoch in tqdm(range(num_epochs), desc=f"Config {config['id']}"):
            try:
                # Use training function
                train_loss, pred_stats = train_epoch(
                    model, data, train_edges, optimizer, device
                )
                train_losses_exp.append(train_loss)
                
                # Store prediction statistics
                for key, value in pred_stats.items():
                    pred_stats_history[key].append(value)
                
                # Log prediction statistics every 10 epochs
                if epoch % 10 == 0:
                    stats_msg = f"Epoch {epoch} pred stats: "
                    stats_msg += f"mean={pred_stats['pred_mean']:.3f}, "
                    stats_msg += f"std={pred_stats['pred_std']:.3f}, "
                    stats_msg += f"pos={pred_stats['pos_pred_mean']:.3f}, "
                    stats_msg += f"neg={pred_stats['neg_pred_mean']:.3f}"
                    print(stats_msg)

                # Evaluate every 5 epochs
                if epoch % 5 == 0 or epoch == num_epochs - 1:
                    # Use evaluation function
                    val_metrics = evaluate(model, data, train_edges, val_edges, device)
                    val_auc = val_metrics['auc']
                    val_ap = val_metrics['ap']
                    
                    val_aucs_exp.append(val_auc)
                    val_aps_exp.append(val_ap)
                    
                    print(f"Epoch {epoch}: Val AUC={val_auc:.4f}, Val AP={val_ap:.4f}")
                    
                    if scheduler:
                        scheduler.step(val_auc)

                    if val_auc > best_val_auc_experiment:
                        best_val_auc_experiment = val_auc
                        best_epoch = epoch
                        patience_counter = 0
                        # Save the best model
                        torch.save({
                            'model_state_dict': model.state_dict(),
                            'config': config,
                            'epoch': epoch,
                            'val_auc': val_auc
                        }, model_checkpoint_path)
                    else:
                        patience_counter += 1
                    
                    if patience_counter >= patience:
                        print(f"Early stopping at epoch {epoch} for config {config['id']}")
                        break
                else:
                    # For non-evaluation epochs, still append to maintain consistency
                    val_aucs_exp.append(val_aucs_exp[-1] if val_aucs_exp else 0.5)
                    val_aps_exp.append(val_aps_exp[-1] if val_aps_exp else 0.5)
                        
            except Exception as e:
                print(f"Error in epoch {epoch} for config {config['id']}: {e}")
                break
        
        # Load best model and evaluate on test set
        try:
            if os.path.exists(model_checkpoint_path):
                checkpoint = torch.load(model_checkpoint_path, weights_only=False, map_location=device)
                model.load_state_dict(checkpoint['model_state_dict'])
            test_metrics = evaluate(model, data, train_edges, test_edges, device)
            test_auc = test_metrics['auc']
            test_ap = test_metrics['ap']
        except Exception as e:
            print(f"Error loading/evaluating best model for config {config['id']}: {e}")
            test_auc, test_ap = 0.0, 0.0

        # Plot training curves
        try:
            plot_save_prefix = f"hetero_gnn_results/plots/training_curves_config_{config['id']}"
            os.makedirs(os.path.dirname(plot_save_prefix), exist_ok=True)
            plot_training_curves(train_losses_exp, val_aucs_exp, val_aps_exp, 
                               pred_stats=pred_stats_history,
                               config=config, save_path_prefix=plot_save_prefix)
        except Exception as e:
            print(f"Error plotting training curves for config {config['id']}: {e}")

        # Save embeddings
        try:
            embeddings_path = f"hetero_gnn_results/models/embeddings_config_{config['id']}.pt"
            save_embeddings(model, train_data, device, save_path=embeddings_path)
        except Exception as e:
            print(f"Error saving embeddings for config {config['id']}: {e}")
            embeddings_path = ""

        return {
            'config_id': int(config['id']),
            'lr': float(config['lr']),
            'hidden_dim': int(config['hidden_dim']),
            'num_layers': int(config['num_layers']),
            'dropout': float(config.get('dropout', 0.2)),
            'weight_decay': float(config.get('weight_decay', 1e-5)),
            'use_lr_scheduler': bool(config.get('use_lr_scheduler', False)),
            'heads': int(config.get('heads', 4)),
            'max_edges_per_api': int(config.get('max_edges_per_api', 20)),
            'best_val_auc': float(best_val_auc_experiment),
            'test_auc': float(test_auc),
            'test_ap': float(test_ap),
            'best_epoch': int(best_epoch),
            'early_stop': bool(epoch < num_epochs - 1 if 'epoch' in locals() else False),
            'model_path': model_checkpoint_path if os.path.exists(model_checkpoint_path) else "",
            'embeddings_path': embeddings_path
        }
        
    except Exception as e:
        print(f"Critical error in experiment for config {config['id']}: {e}")
        return {
            'config_id': int(config['id']),
            'lr': float(config['lr']),
            'hidden_dim': int(config['hidden_dim']),
            'num_layers': int(config['num_layers']),
            'dropout': float(config.get('dropout', 0.2)),
            'weight_decay': float(config.get('weight_decay', 1e-5)),
            'use_lr_scheduler': bool(config.get('use_lr_scheduler', False)),
            'heads': int(config.get('heads', 4)),
            'max_edges_per_api': int(config.get('max_edges_per_api', 20)),
            'best_val_auc': 0.0,
            'test_auc': 0.0,
            'test_ap': 0.0,
            'best_epoch': 0,
            'early_stop': True,
            'model_path': "",
            'embeddings_path': ""
        }

def main():
    try:
        # Set seed for reproducibility
        set_seed(42)
        
        # Setup device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {device}")
        
        # Load data
        data_path = 'dataset.pt'
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Data file not found: {data_path}")
        
        data = torch.load(data_path, weights_only=False)
        print(f"Loaded heterogeneous graph with node types: {data.node_types}")
        print(f"Edge types: {data.edge_types}")
        
        # Setup experiment directory
        results_dir = "hetero_gnn_results"
        os.makedirs(f"{results_dir}/logs", exist_ok=True)
        os.makedirs(f"{results_dir}/models/checkpoints", exist_ok=True)
        os.makedirs(f"{results_dir}/plots", exist_ok=True)
        
        # Define hyperparameter configurations to test
        configs = []
        
        # Learning rates
        learning_rates = [0.001, 0.005, 0.01]
        
        # Hidden dimensions
        hidden_dims = [128, 256, 512]
        
        # Number of layers
        num_layers_options = [2, 3, 4]
        
        # Dropout rates
        dropout_rates = [0.2, 0.3, 0.4]
        
        # Weight decay options
        weight_decay_options = [1e-5, 1e-4, 1e-3]
        
        # Number of attention heads
        heads_options = [4, 8]
        
        # Max edges per API for balanced sampling
        max_edges_per_api_options = [15, 20, 25]
        
        # Generate configurations
        config_id = 0
        
        # Base configurations - test core hyperparameters
        for lr in learning_rates:
            for hidden_dim in hidden_dims:
                for num_layers in num_layers_options:
                    config = {
                        'id': config_id,
                        'lr': lr,
                        'hidden_dim': hidden_dim,
                        'num_layers': num_layers,
                        'dropout': 0.3,  # Default dropout
                        'weight_decay': 1e-4,  # Default weight decay
                        'heads': 4,  # Default heads
                        'max_edges_per_api': 20,  # Default max edges
                        'num_epochs': 150,
                        'patience': 25
                    }
                    configs.append(config)
                    config_id += 1
        
        # Standard configurations with best base hyperparameters
        best_base = {
            'lr': 0.001, 
            'hidden_dim': 256, 
            'num_layers': 3,
            'num_epochs': 150,
            'patience': 25
        }
        
        # Test different dropout rates
        for dropout in dropout_rates:
            config = best_base.copy()
            config.update({
                'id': config_id,
                'dropout': dropout,
                'weight_decay': 1e-4,
                'heads': 4,
                'max_edges_per_api': 20
            })
            configs.append(config)
            config_id += 1
        
        # Test different weight decay values
        for wd in weight_decay_options:
            config = best_base.copy()
            config.update({
                'id': config_id,
                'dropout': 0.3,
                'weight_decay': wd,
                'heads': 4,
                'max_edges_per_api': 20
            })
            configs.append(config)
            config_id += 1
        
        # Test different attention heads
        for heads in heads_options:
            config = best_base.copy()
            config.update({
                'id': config_id,
                'dropout': 0.3,
                'weight_decay': 1e-4,
                'heads': heads,
                'max_edges_per_api': 20
            })
            configs.append(config)
            config_id += 1
        
        # Test different max edges per API
        for max_edges in max_edges_per_api_options:
            config = best_base.copy()
            config.update({
                'id': config_id,
                'dropout': 0.3,
                'weight_decay': 1e-4,
                'heads': 4,
                'max_edges_per_api': max_edges
            })
            configs.append(config)
            config_id += 1
        
        # Add learning rate scheduler tests
        for use_scheduler in [True, False]:
            config = best_base.copy()
            config.update({
                'id': config_id,
                'dropout': 0.3,
                'weight_decay': 1e-4,
                'heads': 4,
                'max_edges_per_api': 20,
                'use_lr_scheduler': use_scheduler
            })
            configs.append(config)
            config_id += 1
        
        print(f"Created {len(configs)} configurations for hyperparameter tuning")
        
        # Run experiments
        results = []
        for config in configs:
            try:
                result = run_experiment(config, data, device)
                results.append(result)
                
                # Save results after each experiment to avoid data loss
                results_df = pd.DataFrame(results)
                # Convert any NumPy types to Python native types before saving
                results_df = convert_dataframe_values(results_df)
                results_df.to_csv(f"{results_dir}/logs/experiment_results.csv", index=False)
                
                print(f"Config {config['id']} completed. Test AUC: {result['test_auc']:.4f}, Test AP: {result['test_ap']:.4f}")
                
            except Exception as e:
                print(f"Error in experiment with config {config['id']}: {e}")
                continue
        
        # Analyze results
        results_df = pd.DataFrame(results)
        
        # Convert any NumPy types to Python native types before saving
        results_df = convert_dataframe_values(results_df)
        
        # Save final results
        results_df.to_csv(f"{results_dir}/logs/final_results.csv", index=False)
        
        # Find best configuration
        best_idx = results_df['test_auc'].idxmax()
        best_config = results_df.iloc[best_idx]
        
        print("\n--- Best Configuration ---")
        print(f"Config ID: {best_config['config_id']}")
        print(f"Learning Rate: {best_config['lr']}")
        print(f"Hidden Dimension: {best_config['hidden_dim']}")
        print(f"Number of Layers: {best_config['num_layers']}")
        print(f"Dropout: {best_config['dropout']}")
        print(f"Weight Decay: {best_config['weight_decay']}")
        print(f"Attention Heads: {best_config['heads']}")
        print(f"Max Edges per API: {best_config['max_edges_per_api']}")
        print(f"Use LR Scheduler: {best_config.get('use_lr_scheduler', False)}")
        print(f"Test AUC: {best_config['test_auc']:.4f}")
        print(f"Test AP: {best_config['test_ap']:.4f}")
        
        # Create hyperparameter analysis plots
        plot_hyperparameter_analysis(results_df, save_dir=f"{results_dir}/plots")
        
        # Save best model path and config for future use
        
        # Create a serializable dictionary with converted values
        best_model_info = {}
        
        # Convert config values
        config_dict = {}
        for k, v in best_config.items():
            if k not in ['config_id', 'best_epoch', 'early_stop', 'model_path', 'embeddings_path']:
                config_dict[k] = convert_to_serializable(v)
        
        best_model_info['config'] = config_dict
        best_model_info['model_path'] = best_config['model_path']
        best_model_info['embeddings_path'] = best_config['embeddings_path']
        best_model_info['test_auc'] = float(best_config['test_auc'])
        best_model_info['test_ap'] = float(best_config['test_ap'])
        
        with open(f"{results_dir}/best_model_info.json", 'w') as f:
            json.dump(best_model_info, f, indent=4)
            
        print(f"\nExperiments completed. Results saved to {results_dir}")
        
    except Exception as e:
        print(f"Error in main function: {e}")

if __name__ == '__main__':
    main()