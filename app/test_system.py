"""
🧪 AI Medical Coding System - Enhanced Multi-Stage Test Suite

Tests the enhanced multi-stage AI validation process:
1. Focused initial selection (8-15 primary codes)
2. Hierarchy enrichment (±3 codes around selected)
3. Bulk retrieval of enriched codes
4. Final clinical refinement
"""

import asyncio
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.title_enricher import TitleEnricher
from app.ai_validator import AIValidator
from app.vectorstore import VectorStore


async def test_enhanced_multi_stage_process():
    """Test the enhanced multi-stage AI validation process"""
    
    print("🧪 Testing Enhanced Multi-Stage AI Validation Process...")
    print("=" * 70)
    
    # Initialize components
    enricher = TitleEnricher()
    validator = AIValidator()
    vectorstore = VectorStore()
    
    # Test data - Depression example to verify root code focus
    test_text = "Depression in Teens: Recognizing the Signs"
    
    try:
        print(f"📄 Test Input: {test_text}")
        print()
        
        # Step 1: Title enrichment
        print("🔍 Step 1: Title Enrichment...")
        enrichment = enricher.enrich_title(test_text)
        search_text = f"{test_text} {enrichment.enriched_keywords}"
        print(f"✅ Enhanced search text: {search_text}")
        print()
        
        # Step 2: Vector search
        print("📊 Step 2: Vector Search...")
        candidates = vectorstore.search_all_codes(search_text, top_k=450)
        print(f"✅ Found {len(candidates)} candidate codes")
        print()
        
        # Step 3: Enhanced Multi-Stage Validation
        print("🎯 Step 3: Enhanced Multi-Stage Validation...")
        print("   📌 Stage 1: Focused primary condition identification...")
        print("   🔍 Stage 2: Hierarchy enrichment (±3 code range)...")
        print("   📋 Stage 3: Bulk retrieval of enriched codes...")
        print("   🩺 Stage 4: Final clinical refinement...")
        
        refinement_result = await validator.enhanced_multi_stage_validation(
            medical_text=search_text,
            initial_candidates=candidates,
            vectorstore=vectorstore
        )
        
        print(f"✅ Enhanced validation complete: {len(refinement_result.refined_codes)} final codes")
        print()
        
        # Display results with analysis
        print("📋 Enhanced Results Analysis:")
        print("-" * 50)
        
        # Analyze root code distribution
        root_codes = {}
        for code in refinement_result.refined_codes:
            root = code.icd_code.split('.')[0] if '.' in code.icd_code else code.icd_code
            if root not in root_codes:
                root_codes[root] = []
            root_codes[root].append(code.icd_code)
        
        print(f"🎯 Root Code Distribution ({len(root_codes)} families):")
        for root, codes in root_codes.items():
            print(f"   • {root}: {len(codes)} codes → {', '.join(codes)}")
        
        print()
        print("🔍 Top 5 Enhanced Codes:")
        for i, code in enumerate(refinement_result.refined_codes[:5], 1):
            confidence_pct = int(code.confidence_score * 100)
            print(f"   {i}. {code.icd_code} ({confidence_pct}%): {code.enhanced_description[:80]}...")
        
        print()
        print(f"🏥 Clinical Summary: {refinement_result.clinical_summary}")
        
        # Success criteria analysis
        print()
        print("✅ Enhanced Process Success Metrics:")
        print(f"   • Root Code Families: {len(root_codes)} (Target: 1-2)")
        print(f"   • Total Final Codes: {len(refinement_result.refined_codes)} (Target: 8-15)")
        print(f"   • Primary Focus: {'✅ ACHIEVED' if len(root_codes) <= 2 else '❌ TOO BROAD'}")
        
        print("\n🚀 Enhanced multi-stage process completed successfully!")
        
    except Exception as e:
        print(f"❌ Enhanced test failed: {str(e)}")
        raise


async def test_hierarchy_enrichment():
    """Test the hierarchy enrichment functionality specifically"""
    
    print("\n🧪 Testing Hierarchy Enrichment...")
    print("=" * 50)
    
    vectorstore = VectorStore()
    
    # Test data
    selected_codes = ["F32.1", "F33.0"]
    excluded_codes = {"F32.0", "F32.9", "F33.1", "F43.21"}  # Simulate initial round results
    
    print(f"📌 Selected Codes: {selected_codes}")
    print(f"🚫 Excluded Codes: {excluded_codes}")
    print()
    
    # Test enrichment
    enriched = vectorstore.enrich_code_hierarchy(
        selected_codes=selected_codes,
        excluded_codes=excluded_codes,
        range_size=3
    )
    
    print(f"🔍 Enriched Codes Generated: {sorted(enriched)}")
    print(f"✅ Successfully generated {len(enriched)} new codes")
    
    # Verify no excluded codes are included
    overlap = enriched.intersection(excluded_codes)
    if overlap:
        print(f"❌ ERROR: Enriched codes overlap with excluded: {overlap}")
    else:
        print("✅ No overlap with excluded codes - Perfect!")


if __name__ == "__main__":
    asyncio.run(test_enhanced_multi_stage_process())
    asyncio.run(test_hierarchy_enrichment()) 