#!/usr/bin/env python3
"""
Flask API Server for GNN Recommender System

This Flask server provides a REST API endpoint for getting API recommendations.

Endpoints:
    POST /recommend - Get API recommendations
    GET /health - Health check endpoint

Usage:
    python flask_server.py
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import logging
from input import RecommendationSystem
import pandas as pd
import sys
import traceback
import json
import os

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize the recommendation system once at startup
print("🚀 Initializing GNN Recommendation System...")
try:
    rec_sys = RecommendationSystem()
    print("✅ Recommendation system loaded successfully!")
except Exception as e:
    print(f"❌ Failed to load recommendation system: {e}")
    sys.exit(1)

def format_recommendation_response(recommendations, apis_df, validation_result=None):
    """Format recommendations for API response with descriptions and URLs"""
    formatted_recommendations = []
    
    logger.info(f"Formatting {len(recommendations)} recommendations")
    
    for i, rec in enumerate(recommendations):
        try:
            api_id = rec['api_id']
            logger.debug(f"Processing recommendation {i+1}: API ID {api_id}")
            
            # Get additional API info from CSV
            api_row = apis_df[apis_df['api_id'] == api_id].iloc[0]
            
            # Safely get API tags
            api_tags_raw = api_row.get('tags', '')
            if isinstance(api_tags_raw, str) and api_tags_raw:
                api_tags = [tag.strip() for tag in api_tags_raw.split(',') if tag.strip()]
            else:
                api_tags = []
            
            # Safely get API tag set from recommendation
            api_tag_set = rec.get('api_tag_set', set())
            if not isinstance(api_tag_set, set):
                logger.warning(f"api_tag_set is not a set for API {api_id}: {type(api_tag_set)}")
                api_tag_set = set()
            
            # Get input tags from validation result if available
            input_tags = set()
            if validation_result and isinstance(validation_result.get('known_tags'), (list, set)):
                input_tags = set(tag.strip().lower() for tag in validation_result['known_tags'] if isinstance(tag, str))
            
            # Calculate matching and additional capabilities
            matching_capabilities = list(api_tag_set & input_tags)
            additional_capabilities = list(api_tag_set - input_tags)[:10]  # Limit to 10
            
            formatted_rec = {
                'api_id': int(api_id),
                'name': rec.get('name', f'API_{api_id}'),
                'description': api_row.get('description', ''),
                'url': api_row.get('url', ''),
                'tags': api_tags,
                'explanation': rec.get('explanation', ''),
                'scores': {
                    'final_score': float(rec.get('final_score', 0)),
                    'embedding_score': float(rec.get('embedding_score', 0)),
                    'tag_bonus': float(rec.get('tag_bonus', 0)),
                    'tag_overlap': int(rec.get('tag_overlap', 0)),
                    'tag_coverage': float(rec.get('tag_coverage', 0))
                },
                'matching_capabilities': matching_capabilities,
                'additional_capabilities': additional_capabilities
            }
            
            formatted_recommendations.append(formatted_rec)
            
        except Exception as e:
            logger.error(f"Error formatting recommendation {i+1}: {e}")
            logger.error(f"Recommendation data: {rec}")
            # Continue with next recommendation rather than failing completely
            continue
    
    logger.info(f"Successfully formatted {len(formatted_recommendations)} recommendations")
    return formatted_recommendations

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'service': 'GNN Recommender API',
        'version': '1.0.0'
    })

@app.route('/valid-tags', methods=['GET'])
def get_valid_tags():
    """
    Endpoint that returns the list of valid tags extracted from the dataset
    
    Returns:
    {
        "valid_tags": ["tag1", "tag2", ...],
        "statistics": {
            "total_unique_tags": 440,
            "mashup_unique_tags": 415,
            "api_unique_tags": 373,
            "common_tags": 348
        },
        "tag_frequencies": {"tag": count, ...}
    }
    """
    try:
        # Load valid tags from JSON file
        valid_tags_path = './valid_tags.json'
        
        if not os.path.exists(valid_tags_path):
            logger.error(f"Valid tags file not found: {valid_tags_path}")
            return jsonify({
                'error': 'Valid tags data not available',
                'status': 'error'
            }), 404
        
        with open(valid_tags_path, 'r') as f:
            tags_data = json.load(f)
        
        logger.info(f"Serving {len(tags_data.get('valid_tags', []))} valid tags")
        
        return jsonify({
            'valid_tags': tags_data.get('valid_tags', []),
            'statistics': tags_data.get('statistics', {}),
            'tag_frequencies': tags_data.get('tag_frequencies', {}),
            'status': 'success'
        })
        
    except Exception as e:
        logger.error(f"Error loading valid tags: {str(e)}")
        
        return jsonify({
            'error': f'Internal server error: {str(e)}',
            'status': 'error'
        }), 500

@app.route('/recommend', methods=['POST'])
def get_recommendations():
    """
    Endpoint that returns API recommendations based on input tags and description
    
    Request body:
    {
        "tags": "comma,separated,tags",
        "description": "optional description",
        "top_k": 5,
        "explainability_mode": true
    }
    """
    try:
        # Parse request data
        data = request.get_json()
        
        if not data:
            return jsonify({
                'error': 'No JSON data provided',
                'status': 'error'
            }), 400
        
        # Extract parameters (same as CLI)
        tags = data.get('tags', '').strip()
        description = data.get('description', '').strip()
        top_k = data.get('top_k', 5)
        explainability_mode = data.get('explainability_mode', True)
        
        # Validate required parameters
        if not tags:
            return jsonify({
                'error': 'Tags parameter is required',
                'status': 'error'
            }), 400
        
        # Validate top_k
        try:
            top_k = int(top_k)
            if top_k < 1 or top_k > 20:
                top_k = 5  # Default
        except (ValueError, TypeError):
            top_k = 5  # Default
        
        logger.info(f"Getting recommendations for tags: '{tags}', description: '{description[:50]}...', top_k: {top_k}")
        
        # Get recommendations using the same method as CLI
        recommendations = rec_sys.get_recommendations_with_explanations(
            input_tags=tags,
            input_description=description,
            top_k=top_k,
            explainability_mode=explainability_mode
        )
        
        logger.info(f"Received {len(recommendations) if recommendations else 0} recommendations")
        
        if not recommendations:
            # Still provide validation info even if no recommendations
            validation_result = rec_sys.validate_input_tags(tags, show_suggestions=True)
            return jsonify({
                'recommendations': [],
                'validation': {
                    'valid': validation_result['valid'],
                    'coverage': validation_result['coverage'],
                    'known_tags': list(validation_result.get('known_tags', set())),
                    'unknown_tags': list(validation_result.get('unknown_tags', set())),
                    'suggestions': validation_result.get('suggestions', []),
                    'warnings': validation_result.get('warnings', [])
                },
                'request_info': {
                    'tags': tags,
                    'description': description,
                    'top_k': top_k,
                    'explainability_mode': explainability_mode
                },
                'status': 'success',
                'message': 'No recommendations found for the given tags'
            })
        
        # Load API data for additional info
        apis_df = pd.read_csv('./csv/api_nodes.csv')
        
        # Get validation info
        validation_result = rec_sys.validate_input_tags(tags, show_suggestions=True)
        
        # Format recommendations with descriptions and URLs
        formatted_recommendations = format_recommendation_response(recommendations, apis_df, validation_result)
        
        # Construct response
        response = {
            'recommendations': formatted_recommendations,
            'validation': {
                'valid': validation_result['valid'],
                'coverage': validation_result['coverage'],
                'known_tags': list(validation_result.get('known_tags', set())) if isinstance(validation_result.get('known_tags'), (set, list)) else [],
                'unknown_tags': list(validation_result.get('unknown_tags', set())) if isinstance(validation_result.get('unknown_tags'), (set, list)) else [],
                'suggestions': validation_result.get('suggestions', []) if isinstance(validation_result.get('suggestions'), list) else [],
                'warnings': validation_result.get('warnings', []) if isinstance(validation_result.get('warnings'), list) else []
            },
            'request_info': {
                'tags': tags,
                'description': description,
                'top_k': top_k,
                'explainability_mode': explainability_mode
            },
            'status': 'success',
            'count': len(formatted_recommendations)
        }
        
        logger.info(f"Successfully generated {len(formatted_recommendations)} recommendations")
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Error processing recommendation request: {str(e)}")
        logger.error(traceback.format_exc())
        
        return jsonify({
            'error': f'Internal server error: {str(e)}',
            'status': 'error'
        }), 500

@app.route('/validate', methods=['POST'])
def validate_tags():
    """
    Validate input tags and get suggestions
    
    Request body:
    {
        "tags": "comma,separated,tags"
    }
    """
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({
                'error': 'No JSON data provided',
                'status': 'error'
            }), 400
        
        tags = data.get('tags', '').strip()
        
        if not tags:
            return jsonify({
                'error': 'Tags parameter is required',
                'status': 'error'
            }), 400
        
        # Validate tags
        validation_result = rec_sys.validate_input_tags(tags, show_suggestions=True)
        
        response = {
            'validation': {
                'valid': validation_result['valid'],
                'coverage': validation_result['coverage'],
                'known_tags': list(validation_result.get('known_tags', set())),
                'unknown_tags': list(validation_result.get('unknown_tags', set())),
                'suggestions': validation_result.get('suggestions', []),
                'warnings': validation_result.get('warnings', [])
            },
            'request_info': {
                'tags': tags
            },
            'status': 'success'
        }
        
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Error validating tags: {str(e)}")
        
        return jsonify({
            'error': f'Internal server error: {str(e)}',
            'status': 'error'
        }), 500

@app.route('/', methods=['GET'])
def index():
    """API documentation endpoint"""
    return jsonify({
        'service': 'GNN Recommender API',
        'version': '1.0.0',
        'description': 'API recommendation system using Graph Neural Networks',
        'endpoints': {
            'POST /recommend': {
                'description': 'Get API recommendations',
                'parameters': {
                    'tags': 'comma-separated tags (required)',
                    'description': 'optional description',
                    'top_k': 'number of recommendations (1-20, default: 5)',
                    'explainability_mode': 'enable explanations (default: true)'
                }
            },
            'POST /validate': {
                'description': 'Validate input tags',
                'parameters': {
                    'tags': 'comma-separated tags (required)'
                }
            },
            'GET /valid-tags': {
                'description': 'Get list of all valid tags from the dataset',
                'parameters': {}
            },
            'GET /health': {
                'description': 'Health check endpoint'
            }
        },
        'examples': {
            'recommend': {
                'url': '/recommend',
                'method': 'POST',
                'body': {
                    'tags': 'social,mapping,location',
                    'description': 'Social mapping application with location features',
                    'top_k': 5,
                    'explainability_mode': True
                }
            },
            'validate': {
                'url': '/validate',
                'method': 'POST',
                'body': {
                    'tags': 'social,mapping,unknown_tag'
                }
            }
        }
    })

if __name__ == '__main__':
    print("\n🌐 Starting GNN Recommender Flask Server...")
    print("📚 Available endpoints:")
    print("  • POST /recommend - Get API recommendations")
    print("  • POST /validate - Validate input tags")
    print("  • GET /valid-tags - Get all valid tags")
    print("  • GET /health - Health check")
    print("  • GET / - API documentation")
    print("\n🔗 Example usage:")
    print("curl -X POST http://localhost:5000/recommend \\")
    print("  -H 'Content-Type: application/json' \\")
    print("  -d '{\"tags\": \"social,mapping,location\", \"description\": \"Social app\", \"top_k\": 3}'")
    print("\n✨ Server starting on http://localhost:5000")
    
    app.run(host='0.0.0.0', port=5000, debug=True)
