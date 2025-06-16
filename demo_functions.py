# Demo functions for the API recommendation system
# This module contains demonstration and testing functions separated from production code

import traceback
from typing import List, Dict, Any


def demo_system_recommendations(rec_sys):
    """Demo the recommendation system with explainability"""
    
    print("RECOMMENDATION SYSTEM DEMO WITH EXPLAINABILITY")
    print("=" * 60)
    
    try:
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
        print("EXPLAINABILITY MODE DEMO")
        print(f"{'='*60}")
        
        for i, test_case in enumerate(test_cases[:3], 1):  # Test first 3 cases
            print(f"\nTest Case {i}: {test_case['name']}")
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
        print("INPUT VALIDATION DEMO")
        print(f"{'='*60}")
        
        # Test with unknown tags
        print(f"\nTesting with unknown tags...")
        recommendations = rec_sys.get_recommendations_with_explanations(
            input_tags="invalidtag, nonexistent, fakecategory",
            input_description="This should trigger validation warnings",
            top_k=3,
            explainability_mode=True
        )
        
        # Test with mixed known/unknown tags
        print(f"\nTesting with mixed known/unknown tags...")
        recommendations = rec_sys.get_recommendations_with_explanations(
            input_tags="web, invalidtag, api, nonexistent",
            input_description="Mix of valid and invalid tags",
            top_k=3,
            explainability_mode=True
        )
        
        # Demo debug mode vs explainability mode
        print(f"\n{'='*60}")
        print("DEBUG MODE vs EXPLAINABILITY MODE")
        print(f"{'='*60}")
        
        test_case = test_cases[0]  # Use first test case
        
        print(f"\nDEBUG MODE (Technical Details):")
        print("-" * 40)
        debug_recommendations = rec_sys.get_recommendations(
            input_tags=test_case['tags'],
            input_description=test_case['description'],
            top_k=3,
            debug=True,
            explainability=False
        )
        
        print(f"\nEXPLAINABILITY MODE (User-Friendly):")
        print("-" * 40)
        explainable_recommendations = rec_sys.get_recommendations(
            input_tags=test_case['tags'],
            input_description=test_case['description'],
            top_k=3,
            debug=False,
            explainability=True
        )
        
        # Show interactive session option
        print(f"\n{'='*60}")
        print("INTERACTIVE SESSION AVAILABLE")
        print(f"{'='*60}")
        print("To start an interactive recommendation session, uncomment the line below:")
        print("# rec_sys.interactive_recommendation_session()")
        
        print(f"\nDemo completed successfully!")
        print("Key features demonstrated:")
        print("   • Input tag validation with suggestions")
        print("   • Human-readable explanations")
        print("   • Tag overlap and similarity scoring")
        print("   • Quality indicators (Excellent/Good/Fair/Weak)")
        print("   • Coverage and match analysis")
        
    except Exception as e:
        print(f"ERROR: Demo failed: {e}")
        traceback.print_exc()


def quick_demo(rec_sys):
    """Quick demo showing just the explainability features"""
    
    print("QUICK EXPLAINABILITY DEMO")
    print("=" * 30)
    
    try:
        # Single test case
        recommendations = rec_sys.get_recommendations_with_explanations(
            input_tags="maps, location, gps",
            input_description="Mobile app needs location services",
            top_k=3
        )
        
        print("\nQuick demo completed!")
        
    except Exception as e:
        print(f"ERROR: Quick demo failed: {e}")


def demo_mashup_vs_api_querying(rec_sys):
    """Demo showing the difference between API capability querying vs Mashup style querying"""
    
    print("API CAPABILITY vs MASHUP STYLE QUERYING DEMO")
    print("=" * 60)
    
    try:
        # Example scenarios showing both approaches
        scenarios = [
            {
                "scenario": "User wants mapping functionality",
                "api_style_query": "maps, geolocation, navigation, gps",
                "mashup_style_query": "mapping, location, travel, mobile",
                "description": "Building a travel app that shows nearby attractions"
            },
            {
                "scenario": "User wants social media features", 
                "api_style_query": "social, sharing, authentication, posts",
                "mashup_style_query": "social, photos, sharing, community",
                "description": "Creating a photo sharing social platform"
            },
            {
                "scenario": "User wants payment processing",
                "api_style_query": "payment, transaction, billing, checkout",
                "mashup_style_query": "ecommerce, shopping, business, finance",
                "description": "Building an online marketplace"
            }
        ]
        
        for i, scenario in enumerate(scenarios, 1):
            print(f"\n{'='*60}")
            print(f"SCENARIO {i}: {scenario['scenario']}")
            print(f"{'='*60}")
            print(f"Project: {scenario['description']}")
            
            print(f"\nAPI CAPABILITY APPROACH:")
            print(f"Query: '{scenario['api_style_query']}'")
            print("(What specific capabilities do you need?)")
            print("-" * 50)
            
            api_recommendations = rec_sys.get_recommendations(
                input_tags=scenario['api_style_query'],
                input_description=scenario['description'],
                top_k=3,
                debug=False,
                explainability=True
            )
            
            print(f"\nMASHUP STYLE APPROACH:")
            print(f"Query: '{scenario['mashup_style_query']}'")
            print("(What type of app/mashup are you building?)")
            print("-" * 50)
            
            mashup_recommendations = rec_sys.get_recommendations_by_mashup_style(
                mashup_tags=scenario['mashup_style_query'],
                mashup_description=scenario['description'],
                top_k=3,
                debug=False,
                explainability=True
            )
            
            # Compare results
            print(f"\nCOMPARISON:")
            print("-" * 30)
            
            api_results = [r['name'] for r in api_recommendations[:3]]
            mashup_results = [r['name'] for r in mashup_recommendations[:3]]
            common_results = set(api_results) & set(mashup_results)
            
            print(f"API Capability Results: {api_results}")
            print(f"Mashup Style Results: {mashup_results}")
            print(f"Common APIs: {list(common_results) if common_results else 'None'}")
            print(f"Overlap: {len(common_results)}/3 APIs")
        
        print(f"\n{'='*60}")
        print("KEY INSIGHTS")
        print(f"{'='*60}")
        print("API Capability Querying:")
        print("   • Focus on specific technical needs")
        print("   • Uses API capability tags (what APIs can do)")
        print("   • Better for technical developers")
        print("   • Higher precision for specific functionality")
        print()
        print("Mashup Style Querying:")
        print("   • Focus on application category/domain")
        print("   • Uses mashup category tags (what you're building)")
        print("   • Better for conceptual planning")
        print("   • Broader discovery of related APIs")
        print()
        print("Both approaches use the SAME trained model!")
        print("   • No retraining required")
        print("   • Same embeddings and similarity calculations")
        print("   • Different tag vocabularies and scoring strategies")
        
    except Exception as e:
        print(f"ERROR: Demo failed: {e}")
        traceback.print_exc()


def display_mashup_style_explainability(recommendations: List[Dict], input_tag_set: set):
    """Display user-friendly explanations for mashup style API recommendations"""
    print(f"\nWhy These APIs Fit Your Mashup Style:")
    print("-" * 60)
    
    for i, rec in enumerate(recommendations, 1):
        print(f"\n{i}. {rec['name']}")
        print(f"   {rec['explanation']}")
        
        # Additional context
        if rec['api_tag_set']:
            shared_style_elements = sorted(input_tag_set & rec['api_tag_set'])
            additional_elements = sorted(rec['api_tag_set'] - input_tag_set)
            
            if shared_style_elements:
                print(f"   Shared style elements: {', '.join(shared_style_elements)}")
            if additional_elements:
                print(f"   Additional capabilities: {', '.join(additional_elements[:5])}")
        
        # Score breakdown
        print(f"   Style compatibility: {rec['style_compatibility']:.1%} | Relevance: {rec['embedding_score']:.2f} | Final: {rec['final_score']:.2f}")
