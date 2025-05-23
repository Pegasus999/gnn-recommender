import torch
from torch_geometric.nn import HeteroConv, GATConv
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple


class HeteroGNN(nn.Module):
    """Heterogeneous Graph Neural Network for Mashup-API Recommendation"""
    
    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int, 
                 num_layers: int = 3, dropout: float = 0.2, num_heads: int = 4):
        super().__init__()
        
        self.num_layers = num_layers
        self.dropout = dropout
        
        # Multi-layer GNN with proper residual connections
        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        self.residual_projections = nn.ModuleList()
        
        for i in range(num_layers):
            # Input/output dimensions for each layer
            input_dim = in_channels if i == 0 else hidden_channels
            output_dim = hidden_channels
            
            conv = HeteroConv({
                ('mashup', 'uses', 'api'): GATConv(
                    input_dim, output_dim // num_heads,
                    heads=num_heads, dropout=dropout, concat=True,
                    add_self_loops=False 
                ),
                ('api', 'rev_uses', 'mashup'): GATConv(
                    input_dim, output_dim // num_heads,
                    heads=num_heads, dropout=dropout, concat=True,
                    add_self_loops=False  
                )
            }, aggr='mean') 
            
            self.convs.append(conv)
            
            # Batch normalization for each node type
            bn_dict = nn.ModuleDict({
                'mashup': nn.BatchNorm1d(hidden_channels),
                'api': nn.BatchNorm1d(hidden_channels)
            })
            self.batch_norms.append(bn_dict)
            
            # Residual projection layers (only needed for first layer)
            if i == 0 and input_dim != output_dim:
                proj_dict = nn.ModuleDict({
                    'mashup': nn.Linear(input_dim, output_dim),
                    'api': nn.Linear(input_dim, output_dim)
                })
                self.residual_projections.append(proj_dict)
            else:
                self.residual_projections.append(None)
        
        # Final projection layers with batch norm
        self.final_bn_mashup = nn.BatchNorm1d(hidden_channels)
        self.final_bn_api = nn.BatchNorm1d(hidden_channels)
        
        self.lin_mashup = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels // 2, out_channels)
        )
        
        self.lin_api = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels // 2, out_channels)
        )
        
        # Initialize weights
        self.reset_parameters()
    
    def reset_parameters(self):
        """Initialize model parameters"""
        for conv in self.convs:
            conv.reset_parameters()
        for module in [self.lin_mashup, self.lin_api]:
            if hasattr(module, 'reset_parameters'):
                module.reset_parameters()
    
    def forward(self, x_dict: Dict[str, torch.Tensor], 
                edge_index_dict: Dict[Tuple[str, str, str], torch.Tensor]) -> Dict[str, torch.Tensor]:
        
        for i, (conv, bn_dict, proj_dict) in enumerate(zip(self.convs, self.batch_norms, self.residual_projections)):
            # Store input for residual connection
            residual_dict = x_dict.copy()
            
            # Apply convolution
            x_dict_new = conv(x_dict, edge_index_dict)
            
            # Apply batch normalization and activation
            for node_type in x_dict_new.keys():
                x_dict_new[node_type] = bn_dict[node_type](x_dict_new[node_type])
                x_dict_new[node_type] = F.relu(x_dict_new[node_type])
                x_dict_new[node_type] = F.dropout(x_dict_new[node_type], 
                                                 p=self.dropout, training=self.training)
            
            # Apply residual connection
            for node_type in x_dict_new.keys():
                if node_type in residual_dict:
                    if proj_dict is not None:
                        # Project residual to match dimensions (first layer)
                        residual = proj_dict[node_type](residual_dict[node_type])
                    else:
                        # Direct residual connection (subsequent layers)
                        residual = residual_dict[node_type]
                    
                    x_dict_new[node_type] = x_dict_new[node_type] + residual
            
            x_dict = x_dict_new
        
        # Final batch normalization
        x_dict['mashup'] = self.final_bn_mashup(x_dict['mashup'])
        x_dict['api'] = self.final_bn_api(x_dict['api'])
        
        # Final projections
        return {
            'mashup': self.lin_mashup(x_dict['mashup']),
            'api': self.lin_api(x_dict['api']),
        }
    
    def decode(self, mashup_z: torch.Tensor, api_z: torch.Tensor, 
               edge_index: torch.Tensor) -> torch.Tensor:
        """Decode edge scores for given edges"""
        src, dst = edge_index
        # Cosine similarity + dot product for scoring
        mashup_emb = F.normalize(mashup_z[src], p=2, dim=1)
        api_emb = F.normalize(api_z[dst], p=2, dim=1)
        return (mashup_emb * api_emb).sum(dim=1)
    
    def decode_all(self, mashup_z: torch.Tensor, api_z: torch.Tensor) -> torch.Tensor:
        """Decode all possible mashup-API pairs"""
        mashup_emb = F.normalize(mashup_z, p=2, dim=1)
        api_emb = F.normalize(api_z, p=2, dim=1)
        return mashup_emb @ api_emb.t()
