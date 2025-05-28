#!/usr/bin/env python3
"""
CLI for API Recommendation System

This script provides a command-line interface for the API recommendation system.

Usage:
    python cli.py                    # Interactive mode
    python cli.py --tags "web,api"   # Direct recommendation
    python cli.py --demo             # Run demo
    python cli.py --validate "invalid,tags"  # Test validation
"""

import argparse
import sys
from input import RecommendationSystem

def main():
    parser = argparse.ArgumentParser(
        description="API Recommendation System with Explainability",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument(
        "--tags", 
        type=str,
        help="Comma-separated tags for recommendation"
    )
    
    parser.add_argument(
        "--description",
        type=str,
        default="",
        help="Description of what you need"
    )
    
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of recommendations to show (default: 5)"
    )
    
    parser.add_argument(
        "--approach",
        type=str,
        choices=["api", "mashup", "capability", "style"],
        default="api",
        help="Recommendation approach: 'api'/'capability' for API capability-based (what you need), 'mashup'/'style' for mashup style-based (what you're building) (default: api)"
    )
    
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Run the full demo"
    )
    
    parser.add_argument(
        "--quick-demo",
        action="store_true",
        help="Run a quick demo"
    )
    
    parser.add_argument(
        "--validate",
        type=str,
        help="Test tag validation with these tags"
    )
    
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Start interactive session"
    )
    
    parser.add_argument(
        "--no-explainability",
        action="store_true",
        help="Disable explainability mode"
    )
    
    parser.add_argument(
        "--dev",
        action="store_true",
        help="Start in developer mode with detailed debug information"
    )
    
    args = parser.parse_args()
    
    try:
        print("Loading API Recommendation System...")
        rec_sys = RecommendationSystem()
        
        if args.demo:
            from input import demo__recommendations
            demo__recommendations()
            
        elif args.quick_demo:
            from input import quick_demo
            quick_demo()
            
        elif args.validate:
            print(f"\nTesting tag validation for: '{args.validate}'")
            validation_result = rec_sys.validate_input_tags(args.validate, show_suggestions=True)
            
            print(f"\nValidation Results:")
            print(f"   Valid: {validation_result['valid']}")
            print(f"   Coverage: {validation_result['coverage']:.1%}")
            print(f"   Known tags: {validation_result.get('known_tags', set())}")
            print(f"   Unknown tags: {validation_result.get('unknown_tags', set())}")
            
            if validation_result.get('suggestions'):
                print(f"   Suggestions: {validation_result['suggestions'][:5]}")
                
        elif args.interactive:
            rec_sys.interactive_recommendation_session(dev_mode=args.dev)
            
        elif args.tags:
            print(f"\nGetting recommendations for: '{args.tags}'")
            print(f"Approach: {args.approach} ({'API capability-based' if args.approach in ['api', 'capability'] else 'Mashup style-based'})")
            
            # Select recommendation method based on approach
            if args.approach in ['mashup', 'style']:
                # Mashup style-based recommendations
                print("Using mashup style-based approach (what you're building)")
                recommendations = rec_sys.get_recommendations_by_mashup_style(
                    mashup_tags=args.tags,
                    mashup_description=args.description,
                    top_k=args.top_k,
                    debug=args.dev,
                    explainability=not args.no_explainability
                )
            else:
                # API capability-based recommendations (default)
                print("Using API capability-based approach (what you need)")
                recommendations = rec_sys.get_recommendations_with_explanations(
                    input_tags=args.tags,
                    input_description=args.description,
                    top_k=args.top_k,
                    explainability_mode=not args.no_explainability
                )
            
            if recommendations:
                print(f"\nGenerated {len(recommendations)} recommendations with explanations!")
            else:
                print("No recommendations generated.")
                
        else:
            # Default to interactive mode
            print("Starting interactive session...")
            if args.dev:
                print("Developer mode enabled - detailed debug information will be shown")
            print("Use --help for other options.")
            rec_sys.interactive_recommendation_session()
            
    except FileNotFoundError as e:
        print(f"Error: Required files not found: {e}")
        print("Make sure you're in the correct directory with the model files.")
        sys.exit(1)
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()