# Debug utilities for the API recommendation system
# This module contains debugging functions separated from production code

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from typing import Dict, List, Any
import pandas as pd


class DebugUtils:
    """Debugging utilities for recommendation system analysis"""
    
    def __init__(self, api_embeddings, apis_df, data, tag_analysis):
        self.api_embeddings = api_embeddings
        self.apis_df = apis_df
        self.data = data
        self.tag_analysis = tag_analysis
    
    def debug_recommendations(self, input_tag_set: set, recommendations: List[Dict], 
                            input_embedding: np.ndarray):
        """Displays detailed information about recommendations for debugging"""
        print(f"\nTop {len(recommendations)} API Recommendations for Mashup Debug:")
        print("-" * 80)
        
        for i, rec in enumerate(recommendations[:5]):  # Show top 5
            print(f"\n{i+1}. {rec['name']} (ID: {rec['api_id']})")
            print(f"Final Score: {rec['final_score']:.3f}")
            print(f"Relevance Score: {rec['embedding_score']:.3f}")
            print(f"Capability Bonus: {rec['tag_bonus']:.1f}")
            print(f"Capabilities Provided: {rec['tag_overlap']}/{len(input_tag_set)}")
            print(f"Need Coverage: {rec['tag_coverage']:.2f}")
            print(f"API Capabilities: {rec['api_tag_set']}")
            print(f"Addresses Needs: {input_tag_set & rec['api_tag_set']}")
            
            # Add explanation
            if 'explanation' in rec:
                print(f"   Explanation: {rec['explanation']}")
            
            if rec['description']:
                print(f"   Description: {rec['description'][:100]}...")
        
        # Embedding similarity analysis
        self.debug_embeddings_similarity(input_embedding, recommendations)
    
    def debug_embeddings_similarity(self, input_embedding: np.ndarray, 
                                  recommendations: List[Dict]):
        """Analyzes embedding similarity patterns between input and recommendations"""
        print(f"\nEmbedding Similarity Analysis:")
        print("-" * 50)
        
        # Get embeddings for top recommendations
        top_embeddings = [self.api_embeddings[rec['api_idx']] for rec in recommendations[:5]]
        
        # Calculate pairwise similarities
        similarities = cosine_similarity([input_embedding] + top_embeddings)
        
        print("Cosine Similarities Matrix:")
        print("     Input   API1   API2   API3   API4   API5")
        for i, row in enumerate(similarities):
            if i == 0:
                print(f"Input  {' '.join(f'{sim:.3f}' for sim in row)}")
            else:
                print(f"API{i}   {' '.join(f'{sim:.3f}' for sim in row)}")
        
        # Check for embedding clustering patterns
        print(f"\nEmbedding Quality Check:")
        input_to_top5_sims = [similarities[0][i+1] for i in range(min(5, len(recommendations)))]
        print(f"   Mean similarity to top 5: {np.mean(input_to_top5_sims):.3f}")
        print(f"   Std similarity to top 5: {np.std(input_to_top5_sims):.3f}")
        
        # Tag-embedding correlation check
        tag_overlaps = [rec['tag_overlap'] for rec in recommendations[:5]]
        if len(tag_overlaps) > 1:
            correlation = np.corrcoef(input_to_top5_sims, tag_overlaps)[0, 1]
            print(f"   Tag-Embedding correlation: {correlation:.3f}")
    
    def debug_mashup_style_recommendations(self, input_tag_set: set, recommendations: List[Dict], 
                                         input_embedding: np.ndarray):
        """Debug information for mashup-style recommendations"""
        print(f"\nTop {len(recommendations)} API Recommendations for Mashup Style Debug:")
        print("-" * 80)
        
        for i, rec in enumerate(recommendations[:5]):
            print(f"\n{i+1}. {rec['name']} (ID: {rec['api_id']})")
            print(f"Final Score: {rec['final_score']:.3f}")
            print(f"Semantic Score: {rec['embedding_score']:.3f}")
            print(f"Style Bonus: {rec['style_bonus']:.1f}")
            print(f"Style Overlap: {rec['style_overlap']}/{len(input_tag_set)}")
            print(f"Style Compatibility: {rec['style_compatibility']:.2f}")
            print(f"API Style Tags: {rec['api_tag_set']}")
            print(f"Shared Style Elements: {input_tag_set & rec['api_tag_set']}")
            
            if 'explanation' in rec:
                print(f"Explanation: {rec['explanation']}")
            
            if rec['description']:
                print(f"Description: {rec['description'][:100]}...")
