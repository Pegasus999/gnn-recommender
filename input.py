# input.py with tag importance and debugging capabilities
import torch
import pandas as pd
import numpy as np
import pickle
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from typing import List, Dict, Any
from model import HeteroGNN
import warnings
warnings.filterwarnings('ignore')

class RecommendationSystem:
    """ recommendation system with tag importance and debugging"""
    
    def __init__(self, model_path: str = "model.pt", 
                 data_path: str = "dataset.pt",
                 vectorizer_path: str = "tfidf_vectorizer.pkl"):
        
        print("🚀 Loading  Recommendation System...")
        
        # Load model
        checkpoint = torch.load(model_path, weights_only=False, map_location='cpu')
        config = checkpoint['config']  # Using 'config' instead of 'model_config'
        
        # Map config parameters to model parameters
        # Based on the model checkpoint's keys and the model's expected parameters
        model_config = {
            'in_channels': 2458,  # Based on the checkpoint's model size
            'hidden_channels': config.get('hidden_dim', 256),  # Map hidden_dim to hidden_channels
            'out_channels': 128,  # Based on the checkpoint's model size
            'num_layers': config.get('num_layers', 2),
            'dropout': config.get('dropout', 0.2),
            'num_heads': config.get('heads', 4)  # Map heads to num_heads
        }
        
        # Import model here to avoid circular imports
        self.model = HeteroGNN(**model_config)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        # Load data
        self.data = torch.load(data_path, weights_only=False, map_location='cpu')
        
        # Load  vectorizer
        with open(vectorizer_path, 'rb') as f:
            self.vectorizer = pickle.load(f)
            
        # Load tag analysis for debugging
        try:
            with open("tag_analysis.pkl", 'rb') as f:
                self.tag_analysis = pickle.load(f)
        except FileNotFoundError:
            self.tag_analysis = None
            print("⚠️ Tag analysis not found")
        
        # Load CSVs for raw data access
        self.mashups_df = pd.read_csv("./csv/mashup_nodes.csv")
        self.apis_df = pd.read_csv("./csv/api_nodes.csv")
        
        print("✅  recommendation system loaded")
        print(f"📊 Model info: {checkpoint.get('val_auc', 'N/A')} best AUC")
        
        # Cache embeddings for faster inference
        self._cache_embeddings()
        
        # Build dataset tag vocabulary for explainability
        self._build_tag_vocabulary()
    
    def _cache_embeddings(self):
        """Pre-compute embeddings for all nodes"""
        print("🧠 Caching embeddings...")
        with torch.no_grad():
            z_dict = self.model(self.data.x_dict, self.data.edge_index_dict)
            self.mashup_embeddings = z_dict['mashup'].cpu().numpy()
            self.api_embeddings = z_dict['api'].cpu().numpy()
        print("✅ Embeddings cached")
    
    def _build_tag_vocabulary(self):
        """Build API tag vocabulary for input validation (users describe needed capabilities using API-style tags)"""
        print("🏷️ Building API capability vocabulary...")
        
        mashup_tags = set()
        api_tags = set()
        self.mashup_tag_frequency = {}
        self.api_tag_frequency = {}
        
        # Collect tags from mashups (currently mostly empty)
        for _, mashup_row in self.mashups_df.iterrows():
            mashup_tags_raw = mashup_row.get('tags', '')
            if pd.notna(mashup_tags_raw) and str(mashup_tags_raw).strip() != '[]':
                tags = [tag.strip().lower() for tag in str(mashup_tags_raw).split(",") if tag.strip()]
                for tag in tags:
                    mashup_tags.add(tag)
                    self.mashup_tag_frequency[tag] = self.mashup_tag_frequency.get(tag, 0) + 1
        
        # Collect tags from APIs (these describe API capabilities - main vocabulary)
        for _, api_row in self.apis_df.iterrows():
            api_tags_raw = api_row.get('tags', '')
            if pd.notna(api_tags_raw) and str(api_tags_raw).strip() != '[]':
                tags = [tag.strip().lower() for tag in str(api_tags_raw).split(",") if tag.strip()]
                for tag in tags:
                    api_tags.add(tag)
                    self.api_tag_frequency[tag] = self.api_tag_frequency.get(tag, 0) + 1
        
        # Input validation should be based on API capability tags (users describe what they need)
        self.dataset_tags = api_tags  # For input validation
        self.api_tags = api_tags  # For API capabilities
        self.common_tags = sorted(self.api_tag_frequency.keys(), key=lambda x: self.api_tag_frequency[x], reverse=True)
        
        print(f"✅ API capability vocabulary built: {len(self.dataset_tags)} unique capability tags")
        if mashup_tags:
            print(f"📊 Found {len(mashup_tags)} mashup tags (supplementary)")
        print(f"📈 Top 10 capability tags: {self.common_tags[:10]}")
    
    def validate_input_tags(self, input_tags: str, show_suggestions: bool = True) -> dict:
        """Validate input tags against dataset vocabulary and provide suggestions"""
        # Parse input tags
        if isinstance(input_tags, str):
            tag_list = [tag.strip().lower() for tag in input_tags.split(",") if tag.strip()]
        else:
            tag_list = input_tags
        
        input_tag_set = set(tag_list)
        known_tags = input_tag_set & self.dataset_tags
        unknown_tags = input_tag_set - self.dataset_tags
        
        validation_result = {
            'valid': len(known_tags) > 0,  # At least one known tag
            'total_tags': len(tag_list),
            'known_tags': len(known_tags),
            'unknown_tags': len(unknown_tags),
            'coverage': len(known_tags) / len(tag_list) if tag_list else 0,
            'known_tag_list': list(known_tags),
            'unknown_tag_list': list(unknown_tags),
            'suggestions': [],
            'warnings': []
        }
        
        # Add warnings
        if not validation_result['valid']:
            validation_result['warnings'].append("No known tags found in input")
        
        if validation_result['coverage'] < 0.5:
            validation_result['warnings'].append(f"Low tag coverage ({validation_result['coverage']:.1%})")
        
        # Find suggestions for unknown tags
        if unknown_tags and show_suggestions:
            all_suggestions = []
            for unknown_tag in unknown_tags:
                similar_tags = self._find_similar_tags(unknown_tag, limit=2)
                all_suggestions.extend(similar_tags)
            
            # Get most common suggestions
            validation_result['suggestions'] = list(set(all_suggestions))[:5]
        
        return validation_result
    
    def _find_similar_tags(self, target_tag: str, limit: int = 3) -> list:
        """Find similar API capability tags using string similarity"""
        from difflib import SequenceMatcher
        
        similarities = []
        for dataset_tag in self.dataset_tags:
            if dataset_tag == '[]':  # Skip empty tag placeholder
                continue
            similarity = SequenceMatcher(None, target_tag.lower(), dataset_tag.lower()).ratio()
            similarities.append((dataset_tag, similarity))
        
        # Sort by similarity and API tag frequency
        similarities.sort(key=lambda x: (x[1], self.api_tag_frequency.get(x[0], 0)), reverse=True)
        return [tag for tag, sim in similarities[:limit] if sim > 0.3]
    
    def generate_explanation(self, recommendation: dict, input_tag_set: set, rank: int) -> str:
        """Generate human-readable explanation for why this API helps build the mashup"""
        api_tags = recommendation['api_tag_set']
        tag_overlap = len(input_tag_set & api_tags)
        total_input_tags = len(input_tag_set)
        embedding_score = recommendation['embedding_score']
        
        # Tag overlap explanation (mashup needs → API capabilities)
        if tag_overlap > 0:
            tag_explanation = f"Provides {tag_overlap}/{total_input_tags} needed capabilities"
        else:
            tag_explanation = "No direct capability overlap"
        
        # Semantic similarity explanation
        if embedding_score >= 0.8:
            similarity_explanation = f"High relevance ({embedding_score:.2f})"
            quality = "Excellent"
        elif embedding_score >= 0.7:
            similarity_explanation = f"Good relevance ({embedding_score:.2f})"
            quality = "Good"
        elif embedding_score >= 0.6:
            similarity_explanation = f"Moderate relevance ({embedding_score:.2f})"
            quality = "Fair"
        else:
            similarity_explanation = f"Low relevance ({embedding_score:.2f})"
            quality = "Weak"
        
        # Coverage explanation (how much of mashup needs are covered)
        coverage = recommendation['tag_coverage']
        if coverage >= 0.8:
            coverage_explanation = f"Covers most needs ({coverage:.0%})"
        elif coverage >= 0.5:
            coverage_explanation = f"Covers key needs ({coverage:.0%})"
        else:
            coverage_explanation = f"Partial coverage ({coverage:.0%})"
        
        # Combine explanations
        explanation = f"{tag_explanation} • {similarity_explanation} • {coverage_explanation} • {quality} fit for your mashup"
        
        return explanation
    
    def process_input_(self, tags: str, description: str = "", 
                             include_descriptions: bool = True) -> np.ndarray:
        """ input processing matching training procedure"""
        
        # Clean and process tags
        if tags:
            tag_list = [tag.strip().lower() for tag in tags.split(",") if tag.strip()]
            tag_text = " ".join(tag_list)
        else:
            tag_text = ""
        
        # Process description
        if description and include_descriptions:
            desc_text = " ".join(description.lower().split())
        else:
            desc_text = ""
        
        # Combine with tag emphasis (3x weight like in training)
        combined_text = f"{tag_text} {tag_text} {tag_text} {desc_text}".strip()
        
        # Transform using  vectorizer
        tfidf_features = self.vectorizer.transform([combined_text]).toarray()[0]
        
        # For tag-only features, we need the tag vectorizer from training
        # For now, we'll pad with zeros (this should be addressed with proper tag vectorizer)
        if hasattr(self.data, 'metadata_dict'):
            num_tag_features = self.data.metadata_dict['num_tag_features']
            tag_features = np.zeros(num_tag_features)  # Placeholder
        else:
            tag_features = np.zeros(500)  # Default size
        
        # Concatenate features like in training
        features = np.concatenate([tfidf_features, tag_features])
        
        return features
    
    def get__recommendations(self, input_tags: str, input_description: str = "",
                                   top_k: int = 10, tag_boost_factor: float = 75.0,
                                   debug: bool = True, explainability: bool = True) -> List[Dict[str, Any]]:
        """ recommendations with explainability and input validation"""
        
        if debug:
            print(f"\n🔍  API Recommendation Debug for requirements: '{input_tags}'")
            print(f"📝 Project description: '{input_description[:100]}...' " if len(input_description) > 100 else f"📝 Project description: '{input_description}'")
            print(f"🏷️ Tag boost factor: {tag_boost_factor}")
        
        # Validate input tags (should describe needed capabilities)
        validation_result = self.validate_input_tags(input_tags, show_suggestions=debug)
        
        if not validation_result['valid']:
            print(f"\n❌ Input Validation Failed!")
            for warning in validation_result['warnings']:
                print(f"   {warning}")
            if validation_result['suggestions']:
                print(f"   💡 Try these API capability tags: {', '.join(validation_result['suggestions'][:5])}")
            return []
        
        if debug and validation_result['coverage'] < 1.0:
            print(f"\n⚠️ Input Tag Coverage Warning:")
            print(f"   Known capability tags: {validation_result['known_tags']}")
            print(f"   Unknown tags: {validation_result['unknown_tags']}")
            if validation_result['suggestions']:
                print(f"   Suggestions: {', '.join(validation_result['suggestions'][:5])}")
        
        # Process input
        input_features = self.process_input_(input_tags, input_description)
        input_tensor = torch.tensor(input_features, dtype=torch.float).unsqueeze(0)
        
        # Get input embedding
        with torch.no_grad():
            # Create temporary data with input
            temp_data = self.data.clone()
            temp_mashup_x = torch.cat([temp_data['mashup'].x, input_tensor], dim=0)
            temp_data['mashup'].x = temp_mashup_x
            
            # Get embeddings
            z_dict = self.model(temp_data.x_dict, temp_data.edge_index_dict)
            input_embedding = z_dict['mashup'][-1].cpu().numpy()
        
        # Calculate embedding similarities
        embedding_similarities = cosine_similarity([input_embedding], self.api_embeddings)[0]
        
        #  tag-based scoring
        api_scores = []
        input_tag_set = set(tag.strip().lower() for tag in input_tags.split(",") if tag.strip())
        
        if debug:
            print(f"🏷️ Required capabilities: {input_tag_set}")
        
        for api_idx in range(len(self.apis_df)):
            api_id = self.data['api'].node_id[api_idx].item()
            api_row = self.apis_df[self.apis_df['api_id'] == api_id].iloc[0]
            
            # Get API tags
            api_tags_raw = api_row.get('tags', '')
            if pd.notna(api_tags_raw):
                api_tag_set = set(tag.strip().lower() for tag in str(api_tags_raw).split(",") if tag.strip())
            else:
                api_tag_set = set()
            
            #  tag overlap calculation
            tag_overlap = len(input_tag_set & api_tag_set)
            tag_coverage = tag_overlap / len(input_tag_set) if input_tag_set else 0
            api_tag_coverage = tag_overlap / len(api_tag_set) if api_tag_set else 0
            
            # Multi-factor tag bonus
            tag_bonus = 0
            if tag_overlap > 0:
                # Base overlap bonus
                tag_bonus += tag_overlap * tag_boost_factor
                
                # Coverage bonuses
                tag_bonus += tag_coverage * 20  # Input coverage bonus
                tag_bonus += api_tag_coverage * 15  # API coverage bonus
                
                # Exact match bonus
                if input_tag_set == api_tag_set:
                    tag_bonus += 30
                
                # Substantial overlap bonus
                if tag_coverage > 0.7:
                    tag_bonus += 25
            
            # Get base embedding score
            embedding_score = embedding_similarities[api_idx]
            
            # Combined score with strong tag emphasis
            final_score = embedding_score + (tag_bonus / 100.0)  # Normalize tag bonus
            
            api_scores.append({
                'api_id': api_id,
                'api_idx': api_idx,
                'name': api_row.get('title', f'API_{api_id}'),
                'tags': api_tags_raw,
                'api_tag_set': api_tag_set,
                'embedding_score': embedding_score,
                'tag_overlap': tag_overlap,
                'tag_coverage': tag_coverage,
                'tag_bonus': tag_bonus,
                'final_score': final_score,
                'description': api_row.get('description', '')
            })
        
        # Sort by final score
        api_scores.sort(key=lambda x: x['final_score'], reverse=True)
        top_recommendations = api_scores[:top_k]
        
        # Add explainability to recommendations
        if explainability:
            for i, rec in enumerate(top_recommendations):
                rec['explanation'] = self.generate_explanation(rec, input_tag_set, i + 1)
                rec['validation_info'] = validation_result
        
        if debug:
            self.debug_recommendations(input_tag_set, top_recommendations, input_embedding)
        
        # Display explainability if requested
        if explainability and not debug:  # Don't duplicate if debug mode already shows details
            self.display_explainability(top_recommendations[:5], input_tag_set)
        
        return top_recommendations
    
    def display_explainability(self, recommendations: List[Dict], input_tag_set: set):
        """Display user-friendly explanations for API recommendations"""
        print(f"\n💡 Why These APIs Are Recommended for Your Mashup:")
        print("-" * 60)
        
        for i, rec in enumerate(recommendations, 1):
            print(f"\n{i}. {rec['name']}")
            print(f"   {rec['explanation']}")
            
            # Additional context
            if rec['api_tag_set']:
                all_api_tags = sorted(rec['api_tag_set'])
                matching_capabilities = sorted(input_tag_set & rec['api_tag_set'])
                additional_capabilities = sorted(rec['api_tag_set'] - input_tag_set)
                
                if matching_capabilities:
                    print(f"   🎯 Provides needed: {', '.join(matching_capabilities)}")
                if additional_capabilities:
                    print(f"   ➕ Bonus capabilities: {', '.join(additional_capabilities[:5])}")
            
            # Score breakdown
            print(f"   📊 Score breakdown: Relevance({rec['embedding_score']:.2f}) + Capability bonus({rec['tag_bonus']:.0f}) = {rec['final_score']:.2f}")
    
    def debug_recommendations(self, input_tag_set: set, recommendations: List[Dict], 
                            input_embedding: np.ndarray):
        """Debug recommendation quality with  explainability"""
        print(f"\n🔍 Top {len(recommendations)} API Recommendations for Mashup Debug:")
        print("-" * 80)
        
        for i, rec in enumerate(recommendations[:5]):  # Show top 5
            print(f"\n{i+1}. {rec['name']} (ID: {rec['api_id']})")
            print(f"   📊 Final Score: {rec['final_score']:.3f}")
            print(f"   🧠 Relevance Score: {rec['embedding_score']:.3f}")
            print(f"   🏷️ Capability Bonus: {rec['tag_bonus']:.1f}")
            print(f"   🔗 Capabilities Provided: {rec['tag_overlap']}/{len(input_tag_set)}")
            print(f"   📈 Need Coverage: {rec['tag_coverage']:.2f}")
            print(f"   🏷️ API Capabilities: {rec['api_tag_set']}")
            print(f"   ✅ Addresses Needs: {input_tag_set & rec['api_tag_set']}")
            
            # Add explanation
            if 'explanation' in rec:
                print(f"   💡 Explanation: {rec['explanation']}")
            
            if rec['description']:
                print(f"   📝 Description: {rec['description'][:100]}...")
        
        # Embedding similarity analysis
        self.debug_embeddings_similarity(input_embedding, recommendations)
    
    def debug_embeddings_similarity(self, input_embedding: np.ndarray, 
                                  recommendations: List[Dict]):
        """Debug embedding similarity patterns"""
        print(f"\n🧠 Embedding Similarity Analysis:")
        print("-" * 50)
        
        # Get embeddings for top recommendations
        top_embeddings = [self.api_embeddings[rec['api_idx']] for rec in recommendations[:5]]
        
        # Calculate pairwise similarities
        similarities = cosine_similarity([input_embedding] + top_embeddings)
        
        print("📊 Cosine Similarities Matrix:")
        print("     Input   API1   API2   API3   API4   API5")
        for i, row in enumerate(similarities):
            if i == 0:
                print(f"Input  {' '.join(f'{sim:.3f}' for sim in row)}")
            else:
                print(f"API{i}   {' '.join(f'{sim:.3f}' for sim in row)}")
        
        # Check for embedding clustering patterns
        print(f"\n🎯 Embedding Quality Check:")
        input_to_top5_sims = [similarities[0][i+1] for i in range(min(5, len(recommendations)))]
        print(f"   Mean similarity to top 5: {np.mean(input_to_top5_sims):.3f}")
        print(f"   Std similarity to top 5: {np.std(input_to_top5_sims):.3f}")
        
        # Tag-embedding correlation check
        tag_overlaps = [rec['tag_overlap'] for rec in recommendations[:5]]
        if len(tag_overlaps) > 1:
            correlation = np.corrcoef(input_to_top5_sims, tag_overlaps)[0, 1]
            print(f"   Tag-Embedding correlation: {correlation:.3f}")
    
    def visualize_embeddings(self, input_tags: str, input_description: str = "", 
                           method: str = 'tsne', save_path: str = None):
        """Visualize embeddings using PCA or t-SNE"""
        
        print(f"🎨 Creating {method.upper()} visualization...")
        
        # Get input embedding
        input_features = self.process_input_(input_tags, input_description)
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
        
        # Get recommendations and highlight them
        recommendations = self.get__recommendations(input_tags, input_description, 
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
            print(f"💾 Saved visualization to {save_path}")
        
        plt.show()
        
        # Print explained variance for PCA
        if method.lower() == 'pca':
            explained_var = reducer.explained_variance_ratio_
            print(f"📊 PCA Explained Variance: {explained_var[0]:.3f}, {explained_var[1]:.3f} (Total: {sum(explained_var):.3f})")
    
    def analyze_tag_based_clusters(self, save_path: str = None):
        """Analyze if embeddings cluster by tag categories"""
        
        print("🔍 Analyzing tag-based clustering...")
        
        if not self.tag_analysis:
            print("❌ Tag analysis not available")
            return
        
        # Get top common tags
        common_tags = list(self.tag_analysis['common_tags'])[:10]
        
        if len(common_tags) < 3:
            print("❌ Not enough common tags for analysis")
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
            print("❌ Not enough APIs per tag category for clustering analysis")
            return
        
        print(f"📊 Analyzing {len(valid_groups)} tag groups:")
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
        print(f"\n📊 Clustering Analysis Results:")
        print("-" * 50)
        
        print("Intra-group similarities (same tag):")
        for tag, stats in results.items():
            print(f"  {tag}: {stats['intra_mean']:.3f} ± {stats['intra_std']:.3f} (n={stats['count']})")
        
        print(f"\nInter-group similarity (different tags): {inter_mean:.3f} ± {inter_std:.3f}")
        
        # Calculate clustering quality metric
        avg_intra = np.mean([stats['intra_mean'] for stats in results.values()])
        clustering_score = (avg_intra - inter_mean) / (avg_intra + inter_mean)
        
        print(f"\n🎯 Clustering Quality Score: {clustering_score:.3f}")
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
        ax1.axhline(y=inter_mean, color='red', linestyle='--', 
                    label=f'Inter-group mean ({inter_mean:.3f})')
        
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
            print(f"💾 Saved clustering analysis to {save_path}")
        
        plt.show()
    
    def get_recommendations_with_explanations(self, input_tags: str, input_description: str = "",
                                            top_k: int = 5, explainability_mode: bool = True) -> List[Dict[str, Any]]:
        """Get API recommendations based on your requirements with built-in explainability"""
        
        print(f"🔍 Finding APIs for your needs: '{input_tags}'")
        if input_description:
            print(f"📝 Project description: '{input_description[:50]}...' " if len(input_description) > 50 else f"📝 Project description: '{input_description}'")
        
        # Get recommendations with explainability
        recommendations = self.get__recommendations(
            input_tags=input_tags,
            input_description=input_description,
            top_k=top_k,
            debug=False,  # Disable debug for cleaner output
            explainability=explainability_mode
        )
        
        if not recommendations:
            print("❌ No API recommendations could be generated. Please check your requirement tags.")
            return []
        
        return recommendations
    
    def interactive_recommendation_session(self):
        """Interactive session for getting API recommendations based on your requirements"""
        
        print("🎯 Interactive API Recommendation Session")
        print("=" * 50)
        print("Describe what you need and get API recommendations with explanations!")
        print("Type 'quit' to exit.\n")
        
        while True:
            try:
                # Get input from user
                input_tags = input("🏷️ Enter capability tags (what you need, comma-separated): ").strip()
                
                if input_tags.lower() == 'quit':
                    print("👋 Thanks for using the API recommendation system!")
                    break
                
                if not input_tags:
                    print("⚠️ Please enter at least one capability tag.\n")
                    continue
                
                input_description = input("📝 Enter project description (optional): ").strip()
                
                try:
                    top_k = int(input("🔢 Number of API recommendations (1-10, default 5): ") or "5")
                    top_k = max(1, min(10, top_k))  # Clamp between 1-10
                except ValueError:
                    top_k = 5
                
                print(f"\n{'='*60}")
                
                # Get recommendations
                recommendations = self.get_recommendations_with_explanations(
                    input_tags=input_tags,
                    input_description=input_description,
                    top_k=top_k,
                    explainability_mode=True
                )
                
                if recommendations:
                    print(f"\n✨ Found {len(recommendations)} API recommendations!")
                
                print(f"\n{'='*60}\n")
                
            except KeyboardInterrupt:
                print("\n👋 Session interrupted. Goodbye!")
                break
            except Exception as e:
                print(f"❌ Error: {e}")
                print("Please try again.\n")


def demo__recommendations():
    """Demo the  recommendation system with explainability"""
    
    print("🎬  Recommendation System Demo with Explainability")
    print("=" * 60)
    
    try:
        # Initialize system
        rec_sys = RecommendationSystem()
        
        # Test cases with different tag patterns
        test_cases = [
            {
                "tags": "maps, geolocation, navigation",
                "description": "Need mapping and location services for mobile app",
                "name": "Mapping Service"
            },
            {
                "tags": "social, media, sharing",
                "description": "Social media integration for content sharing",
                "name": "Social Media"
            },
            {
                "tags": "payment, finance, transaction",
                "description": "Payment processing for e-commerce platform",
                "name": "Payment Service"
            },
            {
                "tags": "weather, forecast, climate",
                "description": "Weather data for agricultural application",
                "name": "Weather Service"
            },
            {
                "tags": "invalid, unknown, nonexistent",
                "description": "Test case with unknown tags",
                "name": "Unknown Tags Test"
            }
        ]
        
        # Demo explainability mode
        print(f"\n{'='*60}")
        print("🎯 EXPLAINABILITY MODE DEMO")
        print(f"{'='*60}")
        
        for i, test_case in enumerate(test_cases[:3], 1):  # Test first 3 cases
            print(f"\n🔍 Test Case {i}: {test_case['name']}")
            print("-" * 40)
            
            # Get recommendations with explainability
            recommendations = rec_sys.get_recommendations_with_explanations(
                input_tags=test_case['tags'],
                input_description=test_case['description'],
                top_k=3,
                explainability_mode=True
            )
        
        # Demo input validation
        print(f"\n{'='*60}")
        print("⚠️ INPUT VALIDATION DEMO")
        print(f"{'='*60}")
        
        # Test with unknown tags
        print(f"\n🧪 Testing with unknown tags...")
        recommendations = rec_sys.get_recommendations_with_explanations(
            input_tags="invalidtag, nonexistent, fakecategory",
            input_description="This should trigger validation warnings",
            top_k=3,
            explainability_mode=True
        )
        
        # Test with mixed known/unknown tags
        print(f"\n🧪 Testing with mixed known/unknown tags...")
        recommendations = rec_sys.get_recommendations_with_explanations(
            input_tags="web, invalidtag, api, nonexistent",
            input_description="Mix of valid and invalid tags",
            top_k=3,
            explainability_mode=True
        )
        
        # Demo debug mode vs explainability mode
        print(f"\n{'='*60}")
        print("🔬 DEBUG MODE vs EXPLAINABILITY MODE")
        print(f"{'='*60}")
        
        test_case = test_cases[0]  # Use first test case
        
        print(f"\n🔍 DEBUG MODE (Technical Details):")
        print("-" * 40)
        debug_recommendations = rec_sys.get__recommendations(
            input_tags=test_case['tags'],
            input_description=test_case['description'],
            top_k=3,
            debug=True,
            explainability=False
        )
        
        print(f"\n💡 EXPLAINABILITY MODE (User-Friendly):")
        print("-" * 40)
        explainable_recommendations = rec_sys.get__recommendations(
            input_tags=test_case['tags'],
            input_description=test_case['description'],
            top_k=3,
            debug=False,
            explainability=True
        )
        
        # Show interactive session option
        print(f"\n{'='*60}")
        print("🎮 INTERACTIVE SESSION AVAILABLE")
        print(f"{'='*60}")
        print("To start an interactive recommendation session, uncomment the line below:")
        print("# rec_sys.interactive_recommendation_session()")
        
        print(f"\n✅ Demo completed successfully!")
        print("🔍 Key features demonstrated:")
        print("   • Input tag validation with suggestions")
        print("   • Human-readable explanations")
        print("   • Tag overlap and similarity scoring")
        print("   • Quality indicators (Excellent/Good/Fair/Weak)")
        print("   • Coverage and match analysis")
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()


def quick_demo():
    """Quick demo showing just the explainability features"""
    
    print("⚡ Quick Explainability Demo")
    print("=" * 30)
    
    try:
        rec_sys = RecommendationSystem()
        
        # Single test case
        recommendations = rec_sys.get_recommendations_with_explanations(
            input_tags="maps, location, gps",
            input_description="Mobile app needs location services",
            top_k=3
        )
        
        print("\n✅ Quick demo completed!")
        
    except Exception as e:
        print(f"❌ Quick demo failed: {e}")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "quick":
        quick_demo()
    else:
        demo__recommendations()
