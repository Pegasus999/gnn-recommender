# Input module for the recommendation system
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
from debug_utils import DebugUtils
from demo_functions import display_mashup_style_explainability
import warnings
warnings.filterwarnings('ignore')

class RecommendationSystem:
    """Recommendation system that provides API recommendations based on user input tags"""
    
    def __init__(self, model_path: str = "model.pt", 
                 data_path: str = "dataset.pt",
                 vectorizer_path: str = "tfidf_vectorizer.pkl"):
        
        print("Loading  Recommendation System...")
        
        # Load model
        checkpoint = torch.load(model_path, weights_only=False, map_location='cpu')
        config = checkpoint['config']  # Using 'config' instead of 'model_config'
        
        # Map config parameters to model parameters
        model_config = {
            'in_channels': 2458,  
            'hidden_channels': config.get('hidden_dim', 256),  # Map hidden_dim to hidden_channels
            'out_channels': 128,  
            'num_layers': config.get('num_layers', 2),
            'dropout': config.get('dropout', 0.2),
            'num_heads': config.get('heads', 4)  # Map heads to num_heads
        }
        
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
            print("Warning: Tag analysis not found")
        
        # Load CSVs for raw data access
        self.mashups_df = pd.read_csv("./csv/mashup_nodes.csv")
        self.apis_df = pd.read_csv("./csv/api_nodes.csv")
        
        print("Recommendation system loaded")
        print(f"Model info: {checkpoint.get('val_auc', 'N/A')} best AUC")
        
        # Cache embeddings for faster inference
        self._cache_embeddings()
        
        # Build dataset tag vocabulary for explainability
        self._build_tag_vocabulary()
        
        # Create tag vectorizer for proper feature processing
        if not self.load_tag_vectorizer():
            self._create_tag_vectorizer()
            # Save the newly created vectorizer
            self.save_tag_vectorizer()
        
        # Get expected feature dimensions for validation
        self.feature_dims = self._get_expected_feature_dimensions()
        print(f"Expected feature dimensions: TF-IDF={self.feature_dims['tfidf']}, Tags={self.feature_dims['tag']}, Total={self.feature_dims['total']}")
        
        # Initialize debug utilities
        self.debug_utils = DebugUtils(self.api_embeddings, self.apis_df, self.data, self.tag_analysis)
    def _cache_embeddings(self):
        """Pre-compute embeddings for all nodes"""
        print("Caching embeddings...")
        with torch.no_grad():
            z_dict = self.model(self.data.x_dict, self.data.edge_index_dict)
            self.mashup_embeddings = z_dict['mashup'].cpu().numpy()
            self.api_embeddings = z_dict['api'].cpu().numpy()
        print("Embeddings cached")
    
    def _build_tag_vocabulary(self):
        """Build API tag vocabulary for input validation (users describe needed capabilities using API-style tags)"""
        print("Building API capability vocabulary...")
        
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
        
        print(f"API capability vocabulary built: {len(self.dataset_tags)} unique capability tags")
        if mashup_tags:
            print(f"Found {len(mashup_tags)} mashup tags (supplementary)")
        print(f"Top 10 capability tags: {self.common_tags[:10]}")
    
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
            if dataset_tag == '[]':  # Skip empty tag placeholder , PS: TOOK ME AGES TO FIGURE OUT WHY THIS WASN'T WORKING LOL
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
                             include_descriptions: bool = True, debug: bool = False) -> np.ndarray:
        """Processes input tags and descriptions into feature vectors with proper tag vectorization"""
        
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
        
        # Transform using TF-IDF vectorizer
        tfidf_features = self.vectorizer.transform([combined_text]).toarray()[0]
        
        # Create proper tag features using tag vectorizer
        if self.tag_vectorizer is not None and tag_text:
            tag_features = self.tag_vectorizer.transform([tag_text]).toarray()[0]
        else:
            # Fallback to zeros if no tag vectorizer or no tags
            tag_features = np.zeros(self.feature_dims['tag'])
        
        # Handle dimension mismatches
        if len(tfidf_features) != self.feature_dims['tfidf']:
            if debug:
                print(f"Warning: TF-IDF dimension mismatch. Expected {self.feature_dims['tfidf']}, got {len(tfidf_features)}")
            # Pad or truncate to expected size
            if len(tfidf_features) < self.feature_dims['tfidf']:
                tfidf_features = np.pad(tfidf_features, (0, self.feature_dims['tfidf'] - len(tfidf_features)))
            else:
                tfidf_features = tfidf_features[:self.feature_dims['tfidf']]
        
        if len(tag_features) != self.feature_dims['tag']:
            if debug:
                print(f"Warning: Tag dimension mismatch. Expected {self.feature_dims['tag']}, got {len(tag_features)}")
            # Pad or truncate to expected size
            if len(tag_features) < self.feature_dims['tag']:
                tag_features = np.pad(tag_features, (0, self.feature_dims['tag'] - len(tag_features)))
            else:
                tag_features = tag_features[:self.feature_dims['tag']]
        
        # Concatenate features like in training
        features = np.concatenate([tfidf_features, tag_features])
        
        if debug:
            print(f"Feature processing debug:")
            print(f"  Input tags: {tag_list if tags else 'None'}")
            print(f"  TF-IDF features: {len(tfidf_features)} (non-zero: {np.count_nonzero(tfidf_features)})")
            print(f"  Tag features: {len(tag_features)} (non-zero: {np.count_nonzero(tag_features)})")
            print(f"  Total features: {len(features)} (expected: {self.feature_dims['total']})")
        
        return features
    
    def get__recommendations(self, input_tags: str, input_description: str = "", top_k: int = 10, tag_boost_factor: float = 75.0, debug: bool = True, explainability: bool = True) -> List[Dict[str, Any]]:
        """Generates API recommendations based on input tags and description"""
        if debug:
            print(f"\nAPI Recommendation Debug for requirements: '{input_tags}'")
        print(f"Project description: '{input_description[:100]}...' " if len(input_description) > 100 else f"Project description: '{input_description}'")
        print(f"Tag boost factor: {tag_boost_factor}")

        # Validate input tags (should describe needed capabilities)
        validation_result = self.validate_input_tags(input_tags, show_suggestions=debug)

        if not validation_result['valid']:
            print(f"\nInput Validation Failed!")
            for warning in validation_result['warnings']:
                print(f"   {warning}")
            if validation_result['suggestions']:
                print(f"   Try these API capability tags: {', '.join(validation_result['suggestions'][:5])}")
            return []

        if debug and validation_result['coverage'] < 1.0:
            print(f"\nInput Tag Coverage Warning:")
            print(f"   Known capability tags: {validation_result['known_tags']}")
            print(f"   Unknown tags: {validation_result['unknown_tags']}")
            if validation_result['suggestions']:
                print(f"   Suggestions: {', '.join(validation_result['suggestions'][:5])}")

        # Process input
        input_features = self.process_input_(input_tags, input_description)
        input_tensor = torch.tensor(input_features, dtype=torch.float).unsqueeze(0)

        # Get input embedding
        with torch.no_grad():
            temp_data = self.data.clone()
            temp_mashup_x = torch.cat([temp_data['mashup'].x, input_tensor], dim=0)
            temp_data['mashup'].x = temp_mashup_x
            z_dict = self.model(temp_data.x_dict, temp_data.edge_index_dict)
            input_embedding = z_dict['mashup'][-1].cpu().numpy()

        # Calculate embedding similarities
        embedding_similarities = cosine_similarity([input_embedding], self.api_embeddings)[0]

        # Tag-based scoring
        api_scores = []
        input_tag_set = set(tag.strip().lower() for tag in input_tags.split(",") if tag.strip())

        if debug:
            print(f"Required capabilities: {input_tag_set}")
        
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
            self.debug_utils.debug_recommendations(input_tag_set, top_recommendations, input_embedding)
        
        # Display explainability if requested
        if explainability and not debug:  # Don't duplicate if debug mode already shows details
            self.display_explainability(top_recommendations[:5], input_tag_set)
        
        return top_recommendations
    
    def display_explainability(self, recommendations: List[Dict], input_tag_set: set):
        """Display user-friendly explanations for API recommendations"""
        print(f"\nWhy These APIs Are Recommended for Your Mashup:")
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
                    print(f"   Provides needed: {', '.join(matching_capabilities)}")
                if additional_capabilities:
                    print(f"    Bonus capabilities: {', '.join(additional_capabilities[:5])}")
            
            # Score breakdown
            print(f"   Score breakdown: Relevance({rec['embedding_score']:.2f}) + Capability bonus({rec['tag_bonus']:.0f}) = {rec['final_score']:.2f}")
    
    def get_recommendations_with_explanations(self, input_tags: str, input_description: str = "",
                                            top_k: int = 5, explainability_mode: bool = True) -> List[Dict[str, Any]]:
        """Get API recommendations based on your requirements with built-in explainability"""
        
        print(f"Finding APIs for you: '{input_tags}'")
        if input_description:
            print(f"Project description: '{input_description[:50]}...' " if len(input_description) > 50 else f"Project description: '{input_description}'")
        
        # Get recommendations with explainability
        recommendations = self.get__recommendations(
            input_tags=input_tags,
            input_description=input_description,
            top_k=top_k,
            debug=False,  
            explainability=explainability_mode
        )
        
        if not recommendations:
            print("No API recommendations could be generated. Please check your requirement tags.")
            return []
        
        return recommendations
    
    def interactive_recommendation_session(self):
        """Interactive session for getting API recommendations based on your requirements"""
        
        print("Interactive API Recommendation Session")
        print("=" * 50)
        print("Get API recommendations with explanations!")
        print("Type 'quit' to exit.\n")
        
        while True:
            try:
                # Get recommendation approach from user
                print("Choose recommendation approach:")
                print("  1. API capability-based (what you need/require)")
                print("  2. Mashup style-based (what you're building)")
                approach_choice = input("Enter choice (1-2, default 1): ").strip() or "1"
                
                if approach_choice.lower() == 'quit':
                    print("Bye lol!")
                    break
                
                # Validate approach choice
                if approach_choice not in ['1', '2']:
                    print("Invalid choice. Please enter 1 or 2.\n")
                    continue
                
                approach_is_api = approach_choice == '1'
                
                # Get input from user with context-specific prompts
                if approach_is_api:
                    input_tags = input("Enter capability tags (what you need, comma-separated): ").strip()
                    approach_name = "API capability-based"
                else:
                    input_tags = input("Enter mashup style tags (what you're building, comma-separated): ").strip()
                    approach_name = "Mashup style-based"
                
                if input_tags.lower() == 'quit':
                    print("Bye lol!")
                    break
                
                if not input_tags:
                    print("Please enter at least one tag.\n")
                    continue
                
                input_description = input("Describe your project: ").strip()
                
                try:
                    top_k = int(input("Number of API recommendations (1-10, default 5): ") or "5")
                    top_k = max(1, min(10, top_k))  # Clamp between 1-10
                except ValueError:
                    top_k = 5
                
                print(f"\n{'='*60}")
                print(f"Using {approach_name} approach")
                
                # Get recommendations based on selected approach
                if approach_is_api:
                    # API capability-based recommendations
                    recommendations = self.get_recommendations_with_explanations(
                        input_tags=input_tags,
                        input_description=input_description,
                        top_k=top_k,
                        explainability_mode=True
                    )
                else:
                    # Mashup style-based recommendations
                    recommendations = self.get_recommendations_by_mashup_style(
                        mashup_tags=input_tags,
                        mashup_description=input_description,
                        top_k=top_k,
                        debug=True,
                        explainability=True
                    )
                
                if not recommendations:
                    print("No recommendations generated. Please try different tags or descriptions.\n")
                    continue

                print(f"\nGenerated {len(recommendations)} recommendations with explanations!")
                for i, rec in enumerate(recommendations, 1):
                    print(f"\n{i}. {rec['name']}")
                    print(f"   Score: {rec['final_score']:.2f}")
                    print(f"   Tags: {rec['tags']}")
                    print(f"   Explanation: {rec['explanation']}")
                    if rec['description']:
                        print(f"   Description: {rec['description'][:100]}..." if len(rec['description']) > 100 else f"   Description: {rec['description']}")

                print(f"\n{'='*60}\n")
                
            except KeyboardInterrupt:
                print("\nSession interrupted. Goodbye!")
                break
            except Exception as e:
                print(f"Error: {e}")
                print("Please try again.\n")


    def _create_tag_vectorizer(self):
        """Create a tag vectorizer based on the dataset's tag vocabulary"""
        from sklearn.feature_extraction.text import CountVectorizer
        
        # Collect all tags from the dataset
        all_tag_texts = []
        
        # Get tags from APIs
        for _, api_row in self.apis_df.iterrows():
            api_tags_raw = api_row.get('tags', '')
            if pd.notna(api_tags_raw) and str(api_tags_raw).strip() != '[]':
                # Clean and format tags
                tags = [tag.strip().lower() for tag in str(api_tags_raw).split(",") if tag.strip()]
                tag_text = " ".join(tags)
                all_tag_texts.append(tag_text)
        
        # Get tags from mashups
        for _, mashup_row in self.mashups_df.iterrows():
            mashup_tags_raw = mashup_row.get('tags', '')
            if pd.notna(mashup_tags_raw) and str(mashup_tags_raw).strip() != '[]':
                tags = [tag.strip().lower() for tag in str(mashup_tags_raw).split(",") if tag.strip()]
                tag_text = " ".join(tags)
                all_tag_texts.append(tag_text)
        
        # Create and fit the tag vectorizer
        self.tag_vectorizer = CountVectorizer(
            max_features=500,  # Match the expected dimension
            binary=True,       # Binary presence/absence
            ngram_range=(1, 1) # Single words only
        )
        
        if all_tag_texts:
            self.tag_vectorizer.fit(all_tag_texts)
            print(f"Tag vectorizer created with {len(self.tag_vectorizer.vocabulary_)} features")
        else:
            print("Warning: No tags found for tag vectorizer")
            self.tag_vectorizer = None

    def save_tag_vectorizer(self, path: str = "tag_vectorizer.pkl"):
        """Save the tag vectorizer for future use"""
        if self.tag_vectorizer is not None:
            with open(path, 'wb') as f:
                pickle.dump(self.tag_vectorizer, f)
            print(f"Tag vectorizer saved to {path}")
        else:
            print("No tag vectorizer to save")

    def load_tag_vectorizer(self, path: str = "tag_vectorizer.pkl"):
        """Load a previously saved tag vectorizer"""
        try:
            with open(path, 'rb') as f:
                self.tag_vectorizer = pickle.load(f)
            print(f"Tag vectorizer loaded from {path}")
            return True
        except FileNotFoundError:
            print(f"Tag vectorizer not found at {path}")
            return False

    def _get_expected_feature_dimensions(self):
        """Get the expected feature dimensions from the model configuration"""
        # The model expects 2458 input features based on the config
        expected_total = 2458
        
        # Try to get the actual dimensions from training data
        if hasattr(self.data, 'metadata_dict'):
            if 'tfidf_features' in self.data.metadata_dict:
                tfidf_dim = self.data.metadata_dict['tfidf_features']
            else:
                tfidf_dim = len(self.vectorizer.vocabulary_) if hasattr(self.vectorizer, 'vocabulary_') else 1958
            
            if 'num_tag_features' in self.data.metadata_dict:
                tag_dim = self.data.metadata_dict['num_tag_features']
            else:
                tag_dim = expected_total - tfidf_dim
        else:
            # Default fallback based on common configuration
            tfidf_dim = 1958  # Common TF-IDF dimension
            tag_dim = 500     # Default tag dimension
        
        return {
            'total': expected_total,
            'tfidf': tfidf_dim,
            'tag': tag_dim
        }

    def get_recommendations_by_mashup_style(self, mashup_tags: str, mashup_description: str = "", 
                                          top_k: int = 10, debug: bool = True, explainability: bool = True) -> List[Dict[str, Any]]:
        """Generates API recommendations based on mashup-style tags and descriptions (what you're building)"""
        if debug:
            print(f"\nAPI Recommendation based on Mashup Style: '{mashup_tags}'")
        print(f"Mashup description: '{mashup_description[:100]}...' " if len(mashup_description) > 100 else f"Mashup description: '{mashup_description}'")

        # Validate input tags against mashup vocabulary
        mashup_validation_result = self.validate_mashup_tags(mashup_tags, show_suggestions=debug)

        if not mashup_validation_result['valid']:
            print(f"\nMashup Tag Validation Failed!")
            for warning in mashup_validation_result['warnings']:
                print(f"   {warning}")
            if mashup_validation_result['suggestions']:
                print(f"   Try these mashup-style tags: {', '.join(mashup_validation_result['suggestions'][:5])}")
            return []

        if debug and mashup_validation_result['coverage'] < 1.0:
            print(f"\nMashup Tag Coverage Warning:")
            print(f"   Known mashup tags: {mashup_validation_result['known_tags']}")
            print(f"   Unknown tags: {mashup_validation_result['unknown_tags']}")
            if mashup_validation_result['suggestions']:
                print(f"   Suggestions: {', '.join(mashup_validation_result['suggestions'][:5])}")

        # Process input using the same feature processing as training
        input_features = self.process_input_(mashup_tags, mashup_description)
        input_tensor = torch.tensor(input_features, dtype=torch.float).unsqueeze(0)

        # Get input embedding (treating as a mashup node)
        with torch.no_grad():
            temp_data = self.data.clone()
            temp_mashup_x = torch.cat([temp_data['mashup'].x, input_tensor], dim=0)
            temp_data['mashup'].x = temp_mashup_x
            z_dict = self.model(temp_data.x_dict, temp_data.edge_index_dict)
            input_embedding = z_dict['mashup'][-1].cpu().numpy()

        # Calculate embedding similarities
        embedding_similarities = cosine_similarity([input_embedding], self.api_embeddings)[0]

        # Mashup-style scoring (different from API capability scoring)
        api_scores = []
        input_tag_set = set(tag.strip().lower() for tag in mashup_tags.split(",") if tag.strip())

        if debug:
            print(f"Building mashup with style: {input_tag_set}")
        
        for api_idx in range(len(self.apis_df)):
            api_id = self.data['api'].node_id[api_idx].item()
            api_row = self.apis_df[self.apis_df['api_id'] == api_id].iloc[0]
            
            # Get API tags
            api_tags_raw = api_row.get('tags', '')
            if pd.notna(api_tags_raw):
                api_tag_set = set(tag.strip().lower() for tag in str(api_tags_raw).split(",") if tag.strip())
            else:
                api_tag_set = set()
            
            # Style compatibility scoring (different from capability matching)
            style_overlap = len(input_tag_set & api_tag_set)
            style_compatibility = style_overlap / len(input_tag_set) if input_tag_set else 0
            api_style_relevance = style_overlap / len(api_tag_set) if api_tag_set else 0
            
            # Style-based bonus (more conservative than capability matching)
            style_bonus = 0
            if style_overlap > 0:
                # Base style overlap bonus
                style_bonus += style_overlap * 30  # Lower than capability matching
                
                # Style compatibility bonuses
                style_bonus += style_compatibility * 15
                style_bonus += api_style_relevance * 10
                
                # High compatibility bonus
                if style_compatibility > 0.5:
                    style_bonus += 20
            
            # Get base embedding score
            embedding_score = embedding_similarities[api_idx]
            
            # Combined score with moderate style emphasis
            final_score = embedding_score + (style_bonus / 100.0)
            
            api_scores.append({
                'api_id': api_id,
                'api_idx': api_idx,
                'name': api_row.get('title', f'API_{api_id}'),
                'tags': api_tags_raw,
                'api_tag_set': api_tag_set,
                'embedding_score': embedding_score,
                'style_overlap': style_overlap,
                'style_compatibility': style_compatibility,
                'style_bonus': style_bonus,
                'final_score': final_score,
                'description': api_row.get('description', '')
            })
        
        # Sort by final score
        api_scores.sort(key=lambda x: x['final_score'], reverse=True)
        top_recommendations = api_scores[:top_k]
        
        # Print the count before explainability
        if top_recommendations:
            print(f"\nFound {len(top_recommendations)} API recommendations!")
        
        # Add explainability to recommendations
        if explainability:
            for i, rec in enumerate(top_recommendations):
                rec['explanation'] = self.generate_mashup_style_explanation(rec, input_tag_set, i + 1)
                rec['validation_info'] = mashup_validation_result
        
        if debug:
            self.debug_utils.debug_mashup_style_recommendations(input_tag_set, top_recommendations, input_embedding)
        elif explainability:
            # Display user-friendly explanations when not in debug mode
            display_mashup_style_explainability(top_recommendations, input_tag_set)
        
        return top_recommendations

    def validate_mashup_tags(self, input_tags: str, show_suggestions: bool = True) -> dict:
        """Validate input tags against mashup tag vocabulary"""
        # Parse input tags
        if isinstance(input_tags, str):
            tag_list = [tag.strip().lower() for tag in input_tags.split(",") if tag.strip()]
        else:
            tag_list = input_tags
        
        input_tag_set = set(tag_list)
        
        # Build mashup tag vocabulary if not exists
        if not hasattr(self, 'mashup_tags'):
            self.mashup_tags = set()
            for _, mashup_row in self.mashups_df.iterrows():
                mashup_tags_raw = mashup_row.get('categories', '')  # Note: mashups use 'categories' field
                if pd.notna(mashup_tags_raw) and str(mashup_tags_raw).strip() != '[]':
                    tags = [tag.strip().lower() for tag in str(mashup_tags_raw).split(",") if tag.strip()]
                    self.mashup_tags.update(tags)
        
        known_tags = input_tag_set & self.mashup_tags
        unknown_tags = input_tag_set - self.mashup_tags
        
        validation_result = {
            'valid': len(known_tags) > 0,
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
            validation_result['warnings'].append("No known mashup-style tags found in input")
        
        if validation_result['coverage'] < 0.5:
            validation_result['warnings'].append(f"Low mashup tag coverage ({validation_result['coverage']:.1%})")
        
        # Find suggestions for unknown tags
        if unknown_tags and show_suggestions:
            all_suggestions = []
            for unknown_tag in unknown_tags:
                similar_tags = self._find_similar_mashup_tags(unknown_tag, limit=2)
                all_suggestions.extend(similar_tags)
            
            validation_result['suggestions'] = list(set(all_suggestions))[:5]
        
        return validation_result

    def _find_similar_mashup_tags(self, target_tag: str, limit: int = 3) -> list:
        """Find similar mashup tags using string similarity"""
        from difflib import SequenceMatcher
        
        if not hasattr(self, 'mashup_tags'):
            return []
        
        similarities = []
        for mashup_tag in self.mashup_tags:
            if mashup_tag == '[]':
                continue
            similarity = SequenceMatcher(None, target_tag.lower(), mashup_tag.lower()).ratio()
            similarities.append((mashup_tag, similarity))
        
        # Sort by similarity
        similarities.sort(key=lambda x: x[1], reverse=True)
        return [tag for tag, sim in similarities[:limit] if sim > 0.3]

    def generate_mashup_style_explanation(self, recommendation: dict, input_tag_set: set, rank: int) -> str:
        """Generate explanation for mashup-style recommendations"""
        api_tags = recommendation['api_tag_set']
        style_overlap = recommendation['style_overlap']
        total_input_tags = len(input_tag_set)
        embedding_score = recommendation['embedding_score']
        
        # Style overlap explanation
        if style_overlap > 0:
            style_explanation = f"Shares {style_overlap}/{total_input_tags} style elements"
        else:
            style_explanation = "No direct style overlap"
        
        # Semantic similarity explanation
        if embedding_score >= 0.8:
            similarity_explanation = f"High semantic similarity ({embedding_score:.2f})"
            quality = "Excellent"
        elif embedding_score >= 0.7:
            similarity_explanation = f"Good semantic similarity ({embedding_score:.2f})"
            quality = "Good"
        elif embedding_score >= 0.6:
            similarity_explanation = f"Moderate semantic similarity ({embedding_score:.2f})"
            quality = "Fair"
        else:
            similarity_explanation = f"Low semantic similarity ({embedding_score:.2f})"
            quality = "Weak"
        
        # Style compatibility explanation
        compatibility = recommendation['style_compatibility']
        if compatibility >= 0.5:
            compatibility_explanation = f"High style compatibility ({compatibility:.0%})"
        elif compatibility >= 0.3:
            compatibility_explanation = f"Moderate style compatibility ({compatibility:.0%})"
        else:
            compatibility_explanation = f"Low style compatibility ({compatibility:.0%})"
        
        # Combine explanations
        explanation = f"{style_explanation} • {similarity_explanation} • {compatibility_explanation} • {quality} fit for your mashup style"
        
        return explanation

# End of RecommendationSystem class
