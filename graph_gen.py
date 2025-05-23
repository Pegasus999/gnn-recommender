# graph.py with tag handling and data balancing
import pandas as pd
import torch
from torch_geometric.data import HeteroData
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import defaultdict, Counter
import random
import pickle
import numpy as np

# === Config ===
MAX_API_USAGE = 25  # Reduced from 40 for stricter filtering
MASHUP_CSV = "./csv/mashup_nodes.csv"
API_CSV = "./csv/api_nodes.csv"
EDGES_CSV = "./csv/mashup_api_edges.csv"
OUTPUT_PATH = "dataset.pt"
VECTORIZER_PATH = "tfidf_vectorizer.pkl"
TAG_ANALYSIS_PATH = "tag_analysis.pkl"

print("🚀 Creating graph with tag importance...")

# === Load CSVs ===
mashups = pd.read_csv(MASHUP_CSV)
apis = pd.read_csv(API_CSV)
edges = pd.read_csv(EDGES_CSV)

print(f"📊 Data loaded: {len(mashups)} mashups, {len(apis)} APIs, {len(edges)} edges")

# === Analyze tag distribution ===
def analyze_tag_distribution():
    """Analyze tag distribution to understand data balance"""
    print("\n🔍 Analyzing tag distribution...")
    
    # Extract all tags
    mashup_tags = []
    api_tags = []
    
    for _, row in mashups.iterrows():
        if pd.notna(row['categories']):
            tags = [tag.strip().lower() for tag in row['categories'].split(',') if tag.strip()]
            mashup_tags.extend(tags)
    
    for _, row in apis.iterrows():
        if pd.notna(row['tags']):
            tags = [tag.strip().lower() for tag in row['tags'].split(',') if tag.strip()]
            api_tags.extend(tags)
    
    mashup_tag_counts = Counter(mashup_tags)
    api_tag_counts = Counter(api_tags)
    
    print(f"📈 Unique mashup tags: {len(mashup_tag_counts)}")
    print(f"📈 Unique API tags: {len(api_tag_counts)}")
    print(f"📈 Top 10 mashup tags: {mashup_tag_counts.most_common(10)}")
    print(f"📈 Top 10 API tags: {api_tag_counts.most_common(10)}")
    
    # Find common tags
    common_tags = set(mashup_tag_counts.keys()) & set(api_tag_counts.keys())
    print(f"🔗 Common tags between mashups and APIs: {len(common_tags)}")
    
    return {
        'mashup_tag_counts': mashup_tag_counts,
        'api_tag_counts': api_tag_counts,
        'common_tags': common_tags
    }

tag_analysis = analyze_tag_distribution()

# ===  edge filtering with category balancing ===
print("\n⚖️  Applying  edge filtering...")

# Group edges by API and analyze their tag distribution
api_edge_dict = defaultdict(list)
api_tag_diversity = defaultdict(set)

for _, row in edges.iterrows():
    api_id = row['api_id']
    mashup_id = row['mashup_id']
    
    # Get mashup tags for this edge
    mashup_row = mashups[mashups['mashup_id'] == mashup_id]
    if not mashup_row.empty and pd.notna(mashup_row.iloc[0]['categories']):
        mashup_tags = set(tag.strip().lower() for tag in mashup_row.iloc[0]['categories'].split(',') if tag.strip())
        api_tag_diversity[api_id].update(mashup_tags)
    
    api_edge_dict[api_id].append((mashup_id, api_id))

# Apply filtering with preference for tag diversity
filtered_edges = []
for api_id, edge_list in api_edge_dict.items():
    if len(edge_list) > MAX_API_USAGE:
        # For APIs with too many edges, prefer diverse tag combinations
        # Shuffle for randomness but maintain some diversity
        random.shuffle(edge_list)
        
        # If we have tag diversity info, try to maintain it
        if api_id in api_tag_diversity and len(api_tag_diversity[api_id]) > 3:
            # Take a balanced sample
            filtered_edges.extend(edge_list[:MAX_API_USAGE])
        else:
            # For APIs with low tag diversity, take fewer edges
            filtered_edges.extend(edge_list[:MAX_API_USAGE // 2])
    else:
        filtered_edges.extend(edge_list)

print(f"🔧 Filtered edges: {len(edges)} → {len(filtered_edges)} (reduction: {(1-len(filtered_edges)/len(edges))*100:.1f}%)")

# ===  ID mapping ===
mashup_ids = mashups['mashup_id'].tolist()
api_ids = apis['api_id'].tolist()
mashup_id_map = {mid: i for i, mid in enumerate(mashup_ids)}
api_id_map = {aid: i for i, aid in enumerate(api_ids)}

edge_index = torch.tensor([
    [mashup_id_map[mid] for mid, _ in filtered_edges],
    [api_id_map[aid] for _, aid in filtered_edges]
], dtype=torch.long)

# ===  TF-IDF Features with richer text processing ===
def process_tags_(series, include_descriptions=False):
    """ tag processing with optional description inclusion"""
    processed = []
    for text in series.fillna(""):
        if text:
            # Split by comma and clean
            tags = [tag.strip().lower() for tag in str(text).split(",") if tag.strip()]
            processed.append(" ".join(tags))
        else:
            processed.append("")
    return pd.Series(processed)

def process_descriptions(series):
    """Process descriptions for additional context"""
    processed = []
    for desc in series.fillna(""):
        if desc and len(str(desc)) > 10:  # Only use substantial descriptions
            # Simple preprocessing: lowercase, remove extra spaces
            clean_desc = " ".join(str(desc).lower().split())
            processed.append(clean_desc)
        else:
            processed.append("")
    return pd.Series(processed)

# Process tags
mashup_docs = process_tags_(mashups['categories'])
api_docs = process_tags_(apis['tags'])

# Process descriptions for additional context
mashup_descriptions = process_descriptions(mashups['description'])
api_descriptions = process_descriptions(apis['description'])

# Combine tags and descriptions with different weights
mashup_combined = []
api_combined = []

for i in range(len(mashup_docs)):
    # Tags get higher weight (repeated 3 times)
    tag_text = " ".join([mashup_docs.iloc[i]] * 3)
    desc_text = mashup_descriptions.iloc[i]
    combined = f"{tag_text} {desc_text}".strip()
    mashup_combined.append(combined)

for i in range(len(api_docs)):
    # Tags get higher weight (repeated 3 times)
    tag_text = " ".join([api_docs.iloc[i]] * 3)
    desc_text = api_descriptions.iloc[i]
    combined = f"{tag_text} {desc_text}".strip()
    api_combined.append(combined)

# Create  TF-IDF vectorizer
print("\n🧠 Creating  TF-IDF features...")
vectorizer = TfidfVectorizer(
    max_features=2000,  # Increase vocabulary size
    min_df=2,          # Minimum document frequency
    max_df=0.8,        # Maximum document frequency
    ngram_range=(1, 2), # Include bigrams
    stop_words='english'
)

# Fit on combined corpus
all_docs = mashup_combined + api_combined
vectorizer.fit(all_docs)

# Save  vectorizer
with open(VECTORIZER_PATH, 'wb') as f:
    pickle.dump(vectorizer, f)
print(f"✅ Saved  TF-IDF vectorizer to {VECTORIZER_PATH}")

# Transform to features
mashup_tfidf = vectorizer.transform(mashup_combined).toarray()
api_tfidf = vectorizer.transform(api_combined).toarray()

# Create tag-only features for concatenation
tag_vectorizer = TfidfVectorizer(
    max_features=500,
    min_df=1,
    ngram_range=(1, 1),
    stop_words='english'
)

# Fit tag-only vectorizer
tag_vectorizer.fit(list(mashup_docs) + list(api_docs))
mashup_tag_features = tag_vectorizer.transform(mashup_docs).toarray()
api_tag_features = tag_vectorizer.transform(api_docs).toarray()

# Concatenate TF-IDF features with tag-only features for enriched embeddings
print(f"🔗 Concatenating features: TF-IDF({mashup_tfidf.shape[1]}) + Tags({mashup_tag_features.shape[1]})")

mashup_x = torch.tensor(
    np.concatenate([mashup_tfidf, mashup_tag_features], axis=1), 
    dtype=torch.float
)
api_x = torch.tensor(
    np.concatenate([api_tfidf, api_tag_features], axis=1), 
    dtype=torch.float
)

print(f"📐 Final feature dimensions: Mashup {mashup_x.shape}, API {api_x.shape}")

# ===  HeteroData with metadata ===
data = HeteroData()
data['mashup'].x = mashup_x
data['mashup'].node_id = torch.tensor(mashup_ids)
data['mashup'].raw_tags = list(mashup_docs)  # Store raw tags for debugging

data['api'].x = api_x
data['api'].node_id = torch.tensor(api_ids)
data['api'].raw_tags = list(api_docs)  # Store raw tags for debugging

data['mashup', 'uses', 'api'].edge_index = edge_index
rev_edge_index = edge_index[[1, 0]]  # flip direction
data['api', 'rev_uses', 'mashup'].edge_index = rev_edge_index

# Store metadata for analysis
metadata = {
    'num_tfidf_features': mashup_tfidf.shape[1],
    'num_tag_features': mashup_tag_features.shape[1],
    'total_features': mashup_x.shape[1],
    'max_api_usage': MAX_API_USAGE,
    'original_edges': len(edges),
    'filtered_edges': len(filtered_edges),
    'vectorizer_vocab_size': len(vectorizer.vocabulary_),
    'tag_analysis': tag_analysis
}

# Store metadata as a node attribute
data.metadata_dict = metadata

# Save tag analysis for later use
with open(TAG_ANALYSIS_PATH, 'wb') as f:
    pickle.dump(tag_analysis, f)

# === Save ===
torch.save(data, OUTPUT_PATH)
print(f"✅ Saved  graph to {OUTPUT_PATH}")
print(f"📊 Graph statistics:")
print(f"   - Total features per node: {data['mashup'].x.shape[1]}")
print(f"   - TF-IDF features: {metadata['num_tfidf_features']}")
print(f"   - Tag-only features: {metadata['num_tag_features']}")
print(f"   - Edges after filtering: {len(filtered_edges)}")
print(f"   - Average edges per API: {len(filtered_edges) / len(api_ids):.1f}")

print("\n🎉  graph creation completed!")
