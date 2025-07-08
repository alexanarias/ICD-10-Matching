"""Test script for spreadsheet processing functionality"""

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.document_reader import extract_title_from_file
from app.title_enricher import TitleEnricher
from app.models import SpreadsheetRow


def test_title_extraction():
    """Test title extraction from different file types"""
    print("🧪 Testing title extraction...")
    
    # Test with sample text
    sample_text = b"Heart Attack\nThis is about heart attacks and myocardial infarction."
    
    # Test PDF title extraction (simulated)
    title = extract_title_from_file(sample_text, "test.txt")
    print(f"✅ Extracted title: '{title}'")
    
    # Test with HTML content
    html_content = b"<html><body><h1>Diabetes Management</h1><p>This is about diabetes.</p></body></html>"
    title_html = extract_title_from_file(html_content, "test.html")
    print(f"✅ Extracted HTML title: '{title_html}'")


def test_title_enrichment():
    """Test AI title enrichment"""
    print("\n🤖 Testing AI title enrichment...")
    
    try:
        enricher = TitleEnricher()
        
        # Test with sample title
        title = "Heart Attack"
        enrichment = enricher.enrich_title(title)
        print(f"✅ Original: {title}")
        print(f"✅ Enriched: {enrichment.enriched_keywords}")
        print(f"✅ Reasoning: {enrichment.reasoning}")
        
    except Exception as e:
        print(f"❌ Error in title enrichment: {str(e)}")


def test_metadata_generation():
    """Test AI metadata generation"""
    print("\n📋 Testing AI metadata generation...")
    
    try:
        enricher = TitleEnricher()
        
        # Test with sample title
        title = "Pregnancy Complications"
        metadata = enricher.generate_metadata(title)
        print(f"✅ Title: {title}")
        print(f"✅ Gender: {metadata.gender}")
        print(f"✅ Keywords: {metadata.keywords}")
        print(f"✅ Reasoning: {metadata.reasoning}")
        
    except Exception as e:
        print(f"❌ Error in metadata generation: {str(e)}")


def test_spreadsheet_row():
    """Test SpreadsheetRow model"""
    print("\n📊 Testing SpreadsheetRow model...")
    
    row = SpreadsheetRow(
        filepath="test.pdf",
        title="Heart Attack",
        gender="Both",
        unique_name="Heart_Attack",
        keywords="heart, myocardial infarction, cardiac",
        diagnosis_codes="I21.9, I25.10",
        cpt_codes="",
        language="English",
        source="AI Medical Coding System",
        document_type="Patient Education"
    )
    
    print(f"✅ Created spreadsheet row:")
    print(f"   Title: {row.title}")
    print(f"   Gender: {row.gender}")
    print(f"   Diagnosis Codes: {row.diagnosis_codes}")


if __name__ == "__main__":
    print("🚀 Testing Spreadsheet Processing Functionality\n")
    
    test_title_extraction()
    test_title_enrichment()
    test_metadata_generation()
    test_spreadsheet_row()
    
    print("\n✅ All tests completed!")
    print("\n📌 Next Steps:")
    print("1. Start the FastAPI server: python -m uvicorn app.main:app --reload")
    print("2. Test the new endpoint: POST /process-spreadsheet")
    print("3. Frontend can send files one by one to build the spreadsheet") 