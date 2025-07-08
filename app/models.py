from pydantic import BaseModel, Field, ConfigDict
from typing import List


class InitialSelectionResponse(BaseModel):
    """Step 1: Simple code selection response"""
    model_config = ConfigDict(extra='forbid')
    
    selected_codes: List[str] = Field(
        description="List of selected ICD codes for clinical review",
        max_items=60  # Allow flexibility around target of 50
    )


class RefinedCodeValidation(BaseModel):
    """Enhanced code validation with refined descriptions"""
    model_config = ConfigDict(extra='forbid')
    
    icd_code: str = Field(description="The ICD-10-CM code")
    original_description: str = Field(description="Original official code description")
    enhanced_description: str = Field(description="AI-enhanced, clinically clear description")
    confidence_score: float = Field(description="Clinical confidence score from 0.0 to 1.0")


class ClinicalRefinementResponse(BaseModel):
    """Step 2: Clinical refinement and enhancement response"""
    model_config = ConfigDict(extra='forbid')
    
    refined_codes: List[RefinedCodeValidation] = Field(
        description="Clinically relevant codes with enhanced descriptions"
    )
    clinical_summary: str = Field(
        description="Summary of clinical refinement and code relevance"
    )


class TitleEnrichment(BaseModel):
    """AI Title Enrichment Response"""
    model_config = ConfigDict(extra='forbid')
    
    enriched_keywords: str = Field(description="Additional medical keywords for better vector search")
    reasoning: str = Field(description="Brief explanation of enrichment")


class DocumentMetadata(BaseModel):
    """AI Document Metadata Generation Response"""
    model_config = ConfigDict(extra='forbid')
    
    gender: str = Field(description="Gender applicability: Male, Female, or Both")
    keywords: str = Field(description="Comma-separated medical keywords extracted from content")
    reasoning: str = Field(description="Brief explanation of metadata generation")


class SpreadsheetRow(BaseModel):
    """Single row for spreadsheet output - Clean and Simple"""
    filepath: str
    title: str
    icd_code_root: str = Field(description="Root ICD codes only: Z0, R34, F84")
    icd_code_hierarchy: str = Field(description="Full hierarchy codes only: Z12.344, R34.3, F84.0")
    details_description: str = Field(description="Code with descriptions: Z12.44: blah blah, R34.3: another description")
    details_score: str = Field(description="Code with confidence: Z123.2: 60%, R34.3: 75%")
    gender: str = ""
    unique_name: str = ""
    keywords: str = ""
    diagnosis_codes: str = ""  # For backward compatibility
    cpt_codes: str = ""
    language: str = "English"
    source: str = "AI Medical Coding System"
    document_type: str = "Patient Education" 