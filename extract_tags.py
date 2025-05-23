#!/usr/bin/env python3
"""
Extract unique tags from API and Mashup CSV files.
This script will create a comprehensive list of all valid tags that can be used 
in the web interface for tag validation.
"""

import pandas as pd
import json
import pickle
from collections import Counter
import re

# File paths
MASHUP_CSV = "./csv/mashup_nodes.csv"
API_CSV = "./csv/api_nodes.csv"
OUTPUT_JSON = "./valid_tags.json"
OUTPUT_PKL = "./valid_tags.pkl"

def extract_tags_from_string(tag_string):
    """Extract and clean tags from a comma-separated string."""
    if pd.isna(tag_string) or tag_string == '[]' or not tag_string.strip():
        return []
    
    # Split by comma and clean up each tag
    tags = [tag.strip().lower() for tag in tag_string.split(',') if tag.strip()]
    
    # Remove any remaining brackets or special characters
    cleaned_tags = []
    for tag in tags:
        # Remove brackets, quotes, and other unwanted characters
        cleaned_tag = re.sub(r'[^\w\s&-]', '', tag).strip()
        if cleaned_tag and cleaned_tag not in ['', 'nan', 'null']:
            cleaned_tags.append(cleaned_tag)
    
    return cleaned_tags

def main():
    print("🏷️  Extracting unique tags from CSV files...")
    
    # Load CSV files
    try:
        mashups = pd.read_csv(MASHUP_CSV)
        apis = pd.read_csv(API_CSV)
        print(f"📊 Loaded {len(mashups)} mashups and {len(apis)} APIs")
    except FileNotFoundError as e:
        print(f"❌ Error loading CSV files: {e}")
        return
    
    # Extract tags from mashups
    mashup_tags = []
    mashup_tag_counter = Counter()
    
    # Check both 'categories' and 'tags' columns for mashups
    for col in ['categories', 'tags']:
        if col in mashups.columns:
            print(f"📈 Processing mashup {col}...")
            for _, row in mashups.iterrows():
                tags = extract_tags_from_string(row[col])
                mashup_tags.extend(tags)
                mashup_tag_counter.update(tags)
    
    # Extract tags from APIs
    api_tags = []
    api_tag_counter = Counter()
    
    if 'tags' in apis.columns:
        print("📈 Processing API tags...")
        for _, row in apis.iterrows():
            tags = extract_tags_from_string(row['tags'])
            api_tags.extend(tags)
            api_tag_counter.update(tags)
    
    # Combine all tags and get unique set
    all_tags = set(mashup_tags + api_tags)
    all_tag_counter = Counter(mashup_tags + api_tags)
    
    # Remove empty strings or invalid tags
    valid_tags = {tag for tag in all_tags if tag and len(tag) > 1}
    
    # Create detailed statistics
    stats = {
        'total_unique_tags': len(valid_tags),
        'mashup_unique_tags': len(set(mashup_tags)),
        'api_unique_tags': len(set(api_tags)),
        'common_tags': len(set(mashup_tags) & set(api_tags)),
        'mashup_only_tags': len(set(mashup_tags) - set(api_tags)),
        'api_only_tags': len(set(api_tags) - set(mashup_tags)),
        'top_10_tags': all_tag_counter.most_common(10),
        'rare_tags_count': len([tag for tag, count in all_tag_counter.items() if count == 1])
    }
    
    # Sort tags alphabetically for easier use
    sorted_tags = sorted(list(valid_tags))
    
    # Create output data structure
    output_data = {
        'valid_tags': sorted_tags,
        'statistics': stats,
        'tag_frequencies': dict(all_tag_counter),
        'mashup_tag_frequencies': dict(mashup_tag_counter),
        'api_tag_frequencies': dict(api_tag_counter)
    }
    
    # Save as JSON (for web frontend)
    with open(OUTPUT_JSON, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    # Save as pickle (for Python backend)
    with open(OUTPUT_PKL, 'wb') as f:
        pickle.dump(output_data, f)
    
    # Print statistics
    print(f"\n📊 Tag Extraction Results:")
    print(f"   Total unique valid tags: {stats['total_unique_tags']}")
    print(f"   Mashup unique tags: {stats['mashup_unique_tags']}")
    print(f"   API unique tags: {stats['api_unique_tags']}")
    print(f"   Common tags (both mashups & APIs): {stats['common_tags']}")
    print(f"   Mashup-only tags: {stats['mashup_only_tags']}")
    print(f"   API-only tags: {stats['api_only_tags']}")
    print(f"   Rare tags (appear only once): {stats['rare_tags_count']}")
    
    print(f"\n🔥 Top 10 most frequent tags:")
    for tag, count in stats['top_10_tags']:
        print(f"   {tag}: {count}")
    
    print(f"\n✅ Valid tags saved to:")
    print(f"   JSON: {OUTPUT_JSON}")
    print(f"   Pickle: {OUTPUT_PKL}")
    
    # Preview some tags
    print(f"\n🏷️  Sample tags (first 20):")
    for tag in sorted_tags[:20]:
        print(f"   {tag}")
    
    if len(sorted_tags) > 20:
        print(f"   ... and {len(sorted_tags) - 20} more")

if __name__ == "__main__":
    main()
