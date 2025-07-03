"""Test script for the AI Medical Coding System with structured outputs"""

import asyncio
from .chapter_classifier import ChapterClassifier
from .vectorstore import VectorStore
from .ai_validator import AIValidator


async def test_structured_outputs():
    """Test the complete workflow with structured outputs"""
    
    # Sample medical text for testing
    test_text = """
    Patient presents with chest pain and shortness of breath. 
    History of diabetes mellitus type 2. 
    Physical examination reveals elevated blood pressure and irregular heartbeat.
    EKG shows signs consistent with myocardial infarction.
    Blood glucose levels are elevated at 250 mg/dL.
    Patient reports recent weight loss and frequent urination.
    """
    
    print("🧪 Testing AI Medical Coding System with Structured Outputs")
    print("=" * 60)
    
    try:
        # Test 1: Chapter Classification
        print("\n1️⃣ Testing Chapter Classification...")
        classifier = ChapterClassifier()
        chapter_result = classifier.classify_chapters(test_text)
        
        print(f"✅ Found {len(chapter_result.predictions)} chapter predictions:")
        for pred in chapter_result.predictions:
            prob_percent = pred.probability * 100
            print(f"   📋 {pred.chapter_name}")
            print(f"      Probability: {prob_percent:.1f}%")
            print(f"      Reasoning: {pred.reasoning}")
            print()
        
        # Get high probability chapters
        target_chapters = classifier.get_high_probability_chapters(chapter_result, 0.5)
        print(f"🎯 Target chapters (>50%): {len(target_chapters)}")
        
        # Test 2: Vector Search
        print("\n2️⃣ Testing Vector Search...")
        vectorstore = VectorStore()
        
        if target_chapters:
            search_results = vectorstore.search_codes_by_chapter(
                test_text, target_chapters, top_k=10
            )
            
            total_candidates = sum(len(results) for results in search_results.values())
            print(f"✅ Found {total_candidates} candidate codes across {len(search_results)} chapters")
            
            # Flatten for validation
            all_candidates = []
            for chapter_name, codes in search_results.items():
                print(f"   📂 {chapter_name}: {len(codes)} codes")
                all_candidates.extend(codes[:5])  # Top 5 per chapter for testing
        else:
            print("⚠️  No high-probability chapters found, searching all chapters...")
            all_candidates = vectorstore.search_all_codes(test_text, top_k=20)
            print(f"✅ Found {len(all_candidates)} candidate codes (global search)")
        
        # Show sample candidates
        print(f"\n📋 Sample candidate codes:")
        for i, candidate in enumerate(all_candidates[:3]):
            print(f"   {i+1}. {candidate['icd_code']} - {candidate['description']}")
            print(f"      Similarity: {candidate['score']:.3f}")
        
        # Test 3: AI Validation
        print("\n3️⃣ Testing AI Validation...")
        validator = AIValidator()
        
        if all_candidates:
            validation_result = validator.validate_codes(test_text, all_candidates)
            
            print(f"✅ Validation complete!")
            print(f"📊 Evaluated {len(validation_result.validated_codes)} codes")
            
            # Get high confidence codes
            final_codes = validator.get_high_confidence_codes(validation_result, 0.5)
            print(f"🏆 High confidence codes (>50%): {len(final_codes)}")
            
            print(f"\n🎯 Final Recommendations:")
            for code in final_codes:
                conf_percent = code.confidence_score * 100
                print(f"   ✓ {code.icd_code} - {code.description}")
                print(f"     Confidence: {conf_percent:.1f}%")
                print(f"     Reasoning: {code.reasoning}")
                print()
            
            print(f"💡 Overall Recommendation:")
            print(f"   {validation_result.overall_recommendation}")
        else:
            print("⚠️  No candidates available for validation")
        
        print("\n" + "=" * 60)
        print("🎉 Test completed successfully! All structured outputs working.")
        
    except Exception as e:
        print(f"❌ Test failed: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(test_structured_outputs()) 