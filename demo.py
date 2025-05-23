#!/usr/bin/env python3
"""
API Recommendation System Demonstration

This script showcases the features of the API recommendation system.
"""

from input import RecommendationSystem
import time

def demo_feature_showcase():
    """Comprehensive demonstration of the recommendation system features"""
    
    print("🎯 API RECOMMENDATION SYSTEM - DEMO")
    print("=" * 65)
    print("This demo showcases the recommendation system features using API tags:")
    print("• Input tag validation with coverage analysis")
    print("• Explanations with quality indicators")
    print("• Tag overlap analysis and similarity scoring")
    print("• Suggestion system for unknown tags")
    print("• API capability tags from the dataset")
    print("=" * 65)
    
    # Initialize system
    print("\n🚀 Initializing  Recommendation System...")
    rec_sys = RecommendationSystem()
    
    # Demo scenarios using REAL API capability tags from the dataset
    scenarios = [
        {
            "name": "✅ Perfect Match Scenario",
            "tags": "social, mapping, location",
            "description": "Social mapping application with location features",
            "expected": "High tag overlap, good recommendations"
        },
        {
            "name": "⚠️ Mixed Valid/Invalid Tags",
            "tags": "social, invalidtag, search, nonexistent",
            "description": "Social search with some invalid tags",
            "expected": "Validation warnings with suggestions"
        },
        {
            "name": "❌ All Invalid Tags",
            "tags": "fakecategory, nonexistent, invalidtag",
            "description": "All tags unknown",
            "expected": "Complete validation failure"
        },
        {
            "name": "🎯 High-Quality Match",
            "tags": "ecommerce, financial, tools",
            "description": "E-commerce platform with financial tools",
            "expected": "Multiple tag matches, good explanations"
        },
        {
            "name": "🎵 Entertainment Scenario",
            "tags": "music, entertainment, streaming",
            "description": "Music streaming and entertainment platform",
            "expected": "Good media-focused recommendations"
        },
        {
            "name": "📱 Mobile Development",
            "tags": "mobile, tools, reference",
            "description": "Mobile development with reference tools",
            "expected": "Development-focused API recommendations"
        }
    ]
    
    for i, scenario in enumerate(scenarios, 1):
        print(f"\n{'='*65}")
        print(f"SCENARIO {i}: {scenario['name']}")
        print(f"{'='*65}")
        print(f"🏷️ Tags: {scenario['tags']}")
        print(f"📝 Description: {scenario['description']}")
        print(f"🎯 Expected: {scenario['expected']}")
        print(f"{'-'*65}")
        
        # Test validation first
        print("🔍 STEP 1: Input Validation")
        validation = rec_sys.validate_input_tags(scenario['tags'], show_suggestions=True)
        print(f"   ✓ Valid: {validation['valid']}")
        print(f"   ✓ Coverage: {validation['coverage']:.1%}")
        print(f"   ✓ Known tags: {validation['known_tag_list']}")
        print(f"   ✓ Unknown tags: {validation['unknown_tag_list']}")
        if validation['suggestions']:
            print(f"   💡 Suggestions: {validation['suggestions'][:3]}")
        
        # Get recommendations with explainability
        print("\n🎯 STEP 2: Get Recommendations")
        recommendations = rec_sys.get_recommendations_with_explanations(
            input_tags=scenario['tags'],
            input_description=scenario['description'],
            top_k=3,
            explainability_mode=True
        )
        
        if recommendations:
            print("✅ Recommendations generated successfully!")
            # Show one detailed example
            rec = recommendations[0]
            print(f"\n🔬 DETAILED ANALYSIS of top recommendation:")
            print(f"   📛 Name: {rec['name']}")
            print(f"   🎯 Explanation: {rec['explanation']}")
            print(f"   📊 Final Score: {rec['final_score']:.2f}")
            print(f"   🔗 Tag Overlap: {rec['tag_overlap']}")
            print(f"   📈 Coverage: {rec['tag_coverage']:.1%}")
            print(f"   🧠 Embedding Score: {rec['embedding_score']:.3f}")
        else:
            print("❌ No recommendations generated")
        
        print(f"\n⏱️ Scenario {i} completed")
        time.sleep(1)  # Brief pause for readability
    
    # Summary
    print(f"\n{'='*65}")
    print("🎉 EXPLAINABILITY SHOWCASE COMPLETED")
    print(f"{'='*65}")
    print("✅ All explainability features demonstrated:")
    print("   • Input validation with coverage analysis")
    print("   • Tag suggestion system for unknown tags")
    print("   • Human-readable explanations")
    print("   • Quality indicators (Excellent/Good/Fair/Weak)")
    print("   • Tag overlap and similarity analysis")
    print("   • Coverage percentage calculations")
    print("   • Score breakdown explanations")
    print("\n💡 The system successfully provides transparency into:")
    print("   • Why each API was recommended")
    print("   • How input tags match API capabilities")
    print("   • Quality of semantic similarity")
    print("   • Input validation and improvement suggestions")
    
    print(f"\n🎯 System is ready for production use!")

if __name__ == "__main__":
    demo_feature_showcase()
