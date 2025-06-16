# Visualization utilities for the API recommendation system
# This module contains visualization and clustering analysis functions

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from typing import Dict, List, Any
import pandas as pd
import torch


class VisualizationUtils:
    """Visualization utilities for embedding analysis and clustering"""
    
    def __init__(self, api_embeddings, apis_df, data, model, tag_analysis):
        self.api_embeddings = api_embeddings
        self.apis_df = apis_df
        self.data = data
        self.model = model
        self.tag_analysis = tag_analysis
    
    def visualize_embeddings(self, input_features, input_tags: str, input_description: str = "", 
                           method: str = 'tsne', save_path: str = None, get_recommendations_func=None):
        """Visualize embeddings using PCA or t-SNE"""
        
        print(f"Creating {method.upper()} visualization...")
        
        # Get input embedding
        input_tensor = torch.tensor(input_features, dtype=torch.float).unsqueeze(0)
        
        with torch.no_grad():
            temp_data = self.data.clone()
            temp_mashup_x = torch.cat([temp_data['mashup'].x, input_tensor], dim=0)
            temp_data['mashup'].x = temp_mashup_x
            z_dict = self.model(temp_data.x_dict, temp_data.edge_index_dict)
            input_embedding = z_dict['mashup'][-1].cpu().numpy()
        
        # Sample embeddings for visualization (too many points make it cluttered)
        sample_size = min(500, len(self.api_embeddings))
        sample_indices = np.random.choice(len(self.api_embeddings), sample_size, replace=False)
        sample_embeddings = self.api_embeddings[sample_indices]
        
        # Combine with input embedding
        all_embeddings = np.vstack([sample_embeddings, input_embedding.reshape(1, -1)])
        
        # Apply dimensionality reduction
        if method.lower() == 'pca':
            reducer = PCA(n_components=2, random_state=42)
        else:  # t-SNE
            reducer = TSNE(n_components=2, random_state=42, perplexity=min(30, sample_size-1))
        
        reduced_embeddings = reducer.fit_transform(all_embeddings)
        
        # Create visualization
        plt.figure(figsize=(12, 8))
        
        # Plot API embeddings
        plt.scatter(reduced_embeddings[:-1, 0], reduced_embeddings[:-1, 1], 
                   c='lightblue', alpha=0.6, s=20, label='APIs')
        
        # Plot input embedding
        plt.scatter(reduced_embeddings[-1, 0], reduced_embeddings[-1, 1], 
                   c='red', s=100, marker='*', label=f'Input: {input_tags}', 
                   edgecolors='black', linewidth=1)
        
        # Get recommendations and highlight them if function provided
        if get_recommendations_func:
            recommendations = get_recommendations_func(input_tags, input_description, 
                                                     top_k=10, debug=False)
            
            # Find top recommendations in sample
            top_api_indices = [rec['api_idx'] for rec in recommendations[:5]]
            sample_mask = np.isin(sample_indices, top_api_indices)
            
            if np.any(sample_mask):
                matched_indices = np.where(sample_mask)[0]
                plt.scatter(reduced_embeddings[matched_indices, 0], reduced_embeddings[matched_indices, 1],
                           c='green', s=60, marker='o', label='Top Recommendations',
                           edgecolors='black', linewidth=1)
        
        plt.title(f'{method.upper()} Visualization of API Embeddings\nInput: {input_tags}')
        plt.xlabel(f'{method.upper()} Component 1')
        plt.ylabel(f'{method.upper()} Component 2')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved visualization to {save_path}")
        
        plt.show()
        
        # Print explained variance for PCA
        if method.lower() == 'pca':
            explained_var = reducer.explained_variance_ratio_
            print(f"PCA Explained Variance: {explained_var[0]:.3f}, {explained_var[1]:.3f} (Total: {sum(explained_var):.3f})")
    
    def analyze_tag_based_clusters(self, save_path: str = None):
        """Analyzes embedding clustering patterns based on API tags"""
        
        print("Analyzing tag-based clustering...")
        
        if not self.tag_analysis:
            print("Tag analysis not available")
            return
        
        # Get top common tags
        common_tags = list(self.tag_analysis['common_tags'])[:10]
        
        if len(common_tags) < 3:
            print("Not enough common tags for analysis")
            return
        
        # Create tag-based groups
        tag_groups = {tag: [] for tag in common_tags}
        
        for api_idx in range(len(self.apis_df)):
            api_id = self.data['api'].node_id[api_idx].item()
            api_row = self.apis_df[self.apis_df['api_id'] == api_id].iloc[0]
            
            api_tags_raw = api_row.get('tags', '')
            if pd.notna(api_tags_raw):
                api_tags = set(tag.strip().lower() for tag in str(api_tags_raw).split(",") if tag.strip())
                
                for tag in common_tags:
                    if tag in api_tags:
                        tag_groups[tag].append(api_idx)
        
        # Filter groups with enough members
        valid_groups = {tag: indices for tag, indices in tag_groups.items() if len(indices) >= 5}
        
        if len(valid_groups) < 2:
            print("ERROR: Not enough APIs per tag category for clustering analysis")
            return
        
        print(f"Analyzing {len(valid_groups)} tag groups:")
        for tag, indices in valid_groups.items():
            print(f"   {tag}: {len(indices)} APIs")
        
        # Calculate intra-group vs inter-group similarities
        results = {}
        
        for tag, indices in valid_groups.items():
            group_embeddings = self.api_embeddings[indices]
            
            # Intra-group similarity
            if len(group_embeddings) > 1:
                intra_sim = cosine_similarity(group_embeddings)
                # Remove diagonal (self-similarity)
                mask = ~np.eye(intra_sim.shape[0], dtype=bool)
                intra_similarities = intra_sim[mask]
                
                results[tag] = {
                    'intra_mean': np.mean(intra_similarities),
                    'intra_std': np.std(intra_similarities),
                    'count': len(indices)
                }
        
        # Inter-group similarities
        inter_similarities = []
        group_names = list(valid_groups.keys())
        
        for i, tag1 in enumerate(group_names):
            for j, tag2 in enumerate(group_names):
                if i < j:  # Avoid duplicates
                    emb1 = self.api_embeddings[valid_groups[tag1]]
                    emb2 = self.api_embeddings[valid_groups[tag2]]
                    
                    inter_sim = cosine_similarity(emb1, emb2)
                    inter_similarities.extend(inter_sim.flatten())
        
        inter_mean = np.mean(inter_similarities)
        inter_std = np.std(inter_similarities)
        
        # Print results
        print(f"\nClustering Analysis Results:")
        print("-" * 50)
        
        print("Intra-group similarities (same tag):")
        for tag, stats in results.items():
            print(f"  {tag}: {stats['intra_mean']:.3f} ± {stats['intra_std']:.3f} (n={stats['count']})")
        
        print(f"\nInter-group similarity (different tags): {inter_mean:.3f} ± {inter_std:.3f}")
        
        # Calculate clustering quality metric
        avg_intra = np.mean([stats['intra_mean'] for stats in results.values()])
        clustering_score = (avg_intra - inter_mean) / (avg_intra + inter_mean)
        
        print(f"\nClustering Quality Score: {clustering_score:.3f}")
        print("   (Higher values indicate intra-group > inter-group similarity)")
        
        # Visualization
        if save_path or True:  # Always show for now
            self._plot_tag_clustering(valid_groups, results, inter_mean, save_path)
    
    def _plot_tag_clustering(self, valid_groups: Dict, results: Dict, 
                           inter_mean: float, save_path: str = None):
        """Plot tag clustering analysis"""
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Plot 1: Intra vs Inter similarities
        tags = list(results.keys())
        intra_means = [results[tag]['intra_mean'] for tag in tags]
        intra_stds = [results[tag]['intra_std'] for tag in tags]
        
        x_pos = np.arange(len(tags))
        ax1.bar(x_pos, intra_means, yerr=intra_stds, alpha=0.7, 
                label='Intra-group', capsize=5)
        ax1.axhline(y=inter_mean, color='red', linestyle='--', label=f'Inter-group mean ({inter_mean:.3f})')
        
        ax1.set_xlabel('Tag Categories')
        ax1.set_ylabel('Cosine Similarity')
        ax1.set_title('Intra-group vs Inter-group Similarities')
        ax1.set_xticks(x_pos)
        ax1.set_xticklabels(tags, rotation=45)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: t-SNE of selected tag groups
        selected_tags = list(valid_groups.keys())[:5]  # Top 5 for clarity
        colors = plt.cm.Set3(np.linspace(0, 1, len(selected_tags)))
        
        # Sample embeddings for t-SNE
        sample_embeddings = []
        sample_labels = []
        sample_colors = []
        
        for i, tag in enumerate(selected_tags):
            indices = valid_groups[tag][:20]  # Max 20 per group
            embeddings = self.api_embeddings[indices]
            sample_embeddings.append(embeddings)
            sample_labels.extend([tag] * len(embeddings))
            sample_colors.extend([colors[i]] * len(embeddings))
        
        all_embeddings = np.vstack(sample_embeddings)
        
        # Apply t-SNE
        tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(all_embeddings)-1))
        tsne_embeddings = tsne.fit_transform(all_embeddings)
        
        # Plot t-SNE
        for i, tag in enumerate(selected_tags):
            mask = np.array(sample_labels) == tag
            ax2.scatter(tsne_embeddings[mask, 0], tsne_embeddings[mask, 1],
                       c=[colors[i]], label=tag, alpha=0.7, s=30)
        
        ax2.set_xlabel('t-SNE Component 1')
        ax2.set_ylabel('t-SNE Component 2')
        ax2.set_title('t-SNE Visualization by Tag Categories')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved clustering analysis to {save_path}")
        
        plt.show()
