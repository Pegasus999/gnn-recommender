# Training module for GNN model
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pickle
import random
import matplotlib.pyplot as plt
from collections import defaultdict, Counter
from typing import Dict, List, Tuple
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score, precision_score, recall_score
from model import HeteroGNN
from utils import (
    set_seed, DataBalancer, convert_to_serializable, convert_dataframe_values,
    create_negative_edges_balanced
)



class MashupAPITrainer:
    """Training pipeline for Mashup-API recommendation model"""
    
    def __init__(self, data_path: str = "dataset.pt", device: str = None):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Load  data
        self.data = torch.load(data_path, weights_only=False, map_location=self.device)
        self.data = self.data.to(self.device)
        
        # Initialize data balancer
        self.balancer = DataBalancer(data_path)
        
        # Load vectorizer for inference setup
        self.load_vectorizer()
        
        # Model parameters
        self.in_channels = self.data['mashup'].x.size(1)
        self.hidden_channels = 256
        self.out_channels = 128
        
        print(f"Graph Statistics:")
        print(f"- Mashups: {self.data['mashup'].x.size(0)}")
        print(f"- APIs: {self.data['api'].x.size(0)}")
        print(f"- Original edges: {self.data['mashup', 'uses', 'api'].edge_index.size(1)}")
        print(f"- Feature dim: {self.in_channels}")
        
        if hasattr(self.data, 'metadata_dict'):
            meta = self.data.metadata_dict
            print(f"   - TF-IDF features: {meta['num_tfidf_features']}")
            print(f"   - Tag features: {meta['num_tag_features']}")
    
    def load_vectorizer(self):
        """Load vectorizer for inference"""
        try:
            with open("tfidf_vectorizer.pkl", 'rb') as f:
                self.vectorizer = pickle.load(f)
            print("Loaded vectorizer")
        except FileNotFoundError:
            try:
                with open("tfidf_vectorizer.pkl", 'rb') as f:
                    self.vectorizer = pickle.load(f)
                print("Loaded original vectorizer")
            except FileNotFoundError:
                print("No vectorizer found")
                self.vectorizer = None
    
    def create_negative_edges_balanced(self, pos_edges: torch.Tensor, num_neg: int = None):
        """Use the utility function to create balanced negative edges"""
        return create_negative_edges_balanced(self.data, pos_edges, num_neg, self.device)
    
    def train_epoch_(self, model: nn.Module, optimizer: torch.optim.Optimizer, 
                           train_edges: torch.Tensor, criterion: nn.Module) -> float:
        """ training epoch with negative sampling"""
        model.train()
        
        # Create training graph
        train_data = self.data.clone()
        train_data['mashup', 'uses', 'api'].edge_index = train_edges
        train_data['api', 'rev_uses', 'mashup'].edge_index = train_edges[[1, 0]]
        
        # Forward pass
        z_dict = model(train_data.x_dict, train_data.edge_index_dict)
        
        # Create balanced positive and negative samples
        num_edges = train_edges.size(1)
        neg_edges = self.create_negative_edges_balanced(train_edges, num_edges)
        
        # Compute scores
        pos_scores = model.decode(z_dict['mashup'], z_dict['api'], train_edges)
        neg_scores = model.decode(z_dict['mashup'], z_dict['api'], neg_edges)
        
        # Binary classification loss
        scores = torch.cat([pos_scores, neg_scores])
        labels = torch.cat([torch.ones_like(pos_scores), torch.zeros_like(neg_scores)])
        
        loss = criterion(scores, labels)
        
        # Add L2 regularization
        l2_reg = 0
        for param in model.parameters():
            l2_reg += torch.norm(param)
        loss += 1e-5 * l2_reg
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        return loss.item()
    
    def evaluate_(self, model: nn.Module, train_edges: torch.Tensor, 
                        val_edges: torch.Tensor) -> Dict[str, float]:
        """ evaluation with more metrics"""
        model.eval()
        
        with torch.no_grad():
            # Create evaluation graph (without validation edges)
            eval_data = self.data.clone()
            eval_data['mashup', 'uses', 'api'].edge_index = train_edges
            eval_data['api', 'rev_uses', 'mashup'].edge_index = train_edges[[1, 0]]
            
            # Get embeddings
            z_dict = model(eval_data.x_dict, eval_data.edge_index_dict)
            
            # Positive scores (validation edges)
            pos_scores = model.decode(z_dict['mashup'], z_dict['api'], val_edges)
            
            # Negative scores (same number as positive)
            neg_edges = self.create_negative_edges_balanced(val_edges, val_edges.size(1))
            neg_scores = model.decode(z_dict['mashup'], z_dict['api'], neg_edges)
            
            # Combine scores and labels
            scores = torch.cat([pos_scores, neg_scores]).cpu().numpy()
            labels = torch.cat([
                torch.ones(pos_scores.size(0)), 
                torch.zeros(neg_scores.size(0))
            ]).cpu().numpy()
            
            # Calculate metrics
            auc = roc_auc_score(labels, scores)
            ap = average_precision_score(labels, scores)
            
            # Additional metrics
            predictions = (scores > 0.5).astype(int)
            accuracy = accuracy_score(labels, predictions)
            precision = precision_score(labels, predictions)
            recall = recall_score(labels, predictions)
            
            return {
                'auc': auc,
                'ap': ap,
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'pos_score_mean': pos_scores.mean().item(),
                'neg_score_mean': neg_scores.mean().item()
            }
    
    def train_(self, epochs: int = 200, lr: float = 0.001, weight_decay: float = 1e-5,
                     patience: int = 25, save_path: str = "model.pt",
                     max_edges_per_api: int = 20):
        """ training pipeline"""
        
        print(f"\nStarting  training...")
        
        # Create balanced data splits
        train_edges, val_edges, test_edges = self.balancer.create_balanced_splits(
            max_edges_per_api=max_edges_per_api
        )
        
        train_edges = train_edges.to(self.device)
        val_edges = val_edges.to(self.device)
        test_edges = test_edges.to(self.device)
        
        # Initialize  model
        model = HeteroGNN(
            in_channels=self.in_channels,
            hidden_channels=self.hidden_channels,
            out_channels=self.out_channels,
            num_layers=3,
            dropout=0.3,  # Increased dropout
            num_heads=4
        ).to(self.device)
        
        #  optimizer setup
        optimizer = torch.optim.AdamW(
            model.parameters(), 
            lr=lr, 
            weight_decay=weight_decay,
            betas=(0.9, 0.999)
        )
        
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=0.7, patience=10
        )
        
        criterion = nn.BCEWithLogitsLoss()
        
        # Training tracking
        best_val_auc = 0
        patience_counter = 0
        train_losses = []
        val_aucs = []
        val_aps = []
        
        print(f"Training on {train_edges.size(1)} edges, validating on {val_edges.size(1)} edges")
        
        for epoch in tqdm(range(epochs), desc="Training"):
            # Training
            train_loss = self.train_epoch_(model, optimizer, train_edges, criterion)
            train_losses.append(train_loss)
            
            # Validation every 5 epochs
            if epoch % 5 == 0:
                val_metrics = self.evaluate_(model, train_edges, val_edges)
                val_auc = val_metrics['auc']
                val_ap = val_metrics['ap']
                
                val_aucs.append(val_auc)
                val_aps.append(val_ap)
                
                # Learning rate scheduling
                scheduler.step(val_auc)
                
                # Early stopping and model saving
                if val_auc > best_val_auc:
                    best_val_auc = val_auc
                    patience_counter = 0
                    
                    # Save best model
                    model_state = {
                        'model_state_dict': model.state_dict(),
                        'model_config': {
                            'in_channels': self.in_channels,
                            'hidden_channels': self.hidden_channels,
                            'out_channels': self.out_channels,
                            'num_layers': 3,
                            'dropout': 0.3,
                            'num_heads': 4
                        },
                        'epoch': epoch,
                        'best_val_auc': best_val_auc,
                        'val_metrics': val_metrics,
                        'vectorizer_path': 'tfidf_vectorizer.pkl'
                    }
                    torch.save(model_state, save_path)
                else:
                    patience_counter += 1
                
                # Print progress
                if epoch % 10 == 0:
                    print(f"\nEpoch {epoch}:")
                    print(f"  Train Loss: {train_loss:.4f}")
                    print(f"  Val AUC: {val_auc:.4f} (Best: {best_val_auc:.4f})")
                    print(f"  Val AP: {val_ap:.4f}")
                    print(f"  Val Accuracy: {val_metrics['accuracy']:.4f}")
                    print(f"  Pos/Neg Score: {val_metrics['pos_score_mean']:.3f}/{val_metrics['neg_score_mean']:.3f}")
                
                # Early stopping
                if patience_counter >= patience:
                    print(f"\nEarly stopping at epoch {epoch} (patience {patience})")
                    break
        
        # Final test evaluation
        print(f"\nFinal test evaluation...")
        model.load_state_dict(torch.load(save_path, weights_only=False)['model_state_dict'])
        test_metrics = self.evaluate_(model, train_edges, test_edges)
        
        print(f"\nFinal Results:")
        print(f"Test AUC: {test_metrics['auc']:.4f}")
        print(f"Test AP: {test_metrics['ap']:.4f}")
        print(f"Test Accuracy: {test_metrics['accuracy']:.4f}")
        print(f"Test Precision: {test_metrics['precision']:.4f}")
        print(f"Test Recall: {test_metrics['recall']:.4f}")
        
        # Plot training curves
        self.plot__training_curves(train_losses, val_aucs, val_aps)
        
        return model, test_metrics
    
    def plot__training_curves(self, train_losses: List[float], 
                                    val_aucs: List[float], val_aps: List[float]):
        """Plot  training curves"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # Training loss
        ax1.plot(train_losses)
        ax1.set_title('Training Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.grid(True)
        
        # Validation AUC
        epochs_val = range(0, len(train_losses), 5)[:len(val_aucs)]
        ax2.plot(epochs_val, val_aucs, 'g-')
        ax2.set_title('Validation AUC')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('AUC')
        ax2.grid(True)
        
        # Validation AP
        ax3.plot(epochs_val, val_aps, 'r-')
        ax3.set_title('Validation Average Precision')
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('AP')
        ax3.grid(True)
        
        # Loss histogram
        ax4.hist(train_losses[-50:], bins=20, alpha=0.7)
        ax4.set_title('Recent Training Loss Distribution')
        ax4.set_xlabel('Loss')
        ax4.set_ylabel('Frequency')
        ax4.grid(True)
        
        plt.tight_layout()
        plt.savefig('_training_curves.png', dpi=300, bbox_inches='tight')
        print("Saved training curves to _training_curves.png")
        plt.show()


if __name__ == "__main__":
    # Run  training
    print("Starting Training Pipeline babyyyyy")
    
    # Train  model
    trainer = MashupAPITrainer("dataset.pt")
    
    model, test_metrics = trainer.train_(
        epochs=150,
        lr=0.001,
        weight_decay=1e-4,
        patience=25,
        max_edges_per_api=20  # Stricter edge limit
    )
    
    print("Training completed!")
    print(f"Final test metrics: {test_metrics}")
