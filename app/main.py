"""
🏥 AI Medical Coding System 
1. vector search 
2. Initial selection (~50 relevant codes)
3. Clinical refinement (enhanced descriptions + confidence)
"""

from fastapi import FastAPI, HTTPException, File, UploadFile, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from typing import Dict, Any, List

from .models import SpreadsheetRow, RefinedCodeValidation
from .document_reader import extract_title_from_file, extract_text_from_file
from .title_enricher import TitleEnricher
from .ai_validator import AIValidator
from .vectorstore import VectorStore

# Initialize components
app = FastAPI(title="AI Medical Coding System", version="2.0")
templates = Jinja2Templates(directory="app/templates")
app.mount("/static", StaticFiles(directory="app/static"), name="static")

# Initialize AI components
title_enricher = TitleEnricher()
validator = AIValidator()
vectorstore = VectorStore()

@app.get("/", response_class=HTMLResponse)
async def upload_form(request: Request):
    """Single document analysis interface"""
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/spreadsheet", response_class=HTMLResponse)
async def spreadsheet_interface(request: Request):
    """Bulk spreadsheet processing interface"""
    return templates.TemplateResponse("spreadsheet.html", {"request": request})

@app.post("/analyze")
async def analyze_document(file: UploadFile = File(...)) -> Dict[str, Any]:
    """
    Same two-step AI process as spreadsheet but simplified for single documents
    """
    try:
        # Step 1: Extract clean filename (strip any path)
        clean_filename = file.filename.split('/')[-1] if '/' in file.filename else file.filename
        
        # Step 2: Extract title
        file_content = await file.read()
        title = extract_title_from_file(file_content, clean_filename)
        
        if not title:
            raise HTTPException(
                status_code=400,
                detail=f"Could not extract title from file: {clean_filename}"
            )
        
        # Step 2: AI Title Enrichment
        enrichment_result = title_enricher.enrich_title(title)
        search_text = f"{title} {enrichment_result.enriched_keywords}"
        
        # Step 3: Direct Vector Search (No Chapter Limitations!)
        all_candidates = vectorstore.search_all_codes(search_text, top_k=450)
        
        if not all_candidates:
            raise HTTPException(
                status_code=404,
                detail="No relevant codes found"
            )
        
        # Step 4: Enhanced Multi-Stage AI Validation Process
        
        # NEW: Enhanced multi-stage process with hierarchy enrichment
        refinement_result = await validator.enhanced_multi_stage_validation(
            medical_text=search_text,
            initial_candidates=all_candidates,
            vectorstore=vectorstore
        )
        
        if not refinement_result.refined_codes:
            raise HTTPException(
                status_code=404,
                detail="No codes validated after clinical review"
            )
        
        # Step 5: Format Clean Response
        diagnosis_codes = [code.icd_code for code in refinement_result.refined_codes]
        
        # Enhanced code details
        code_details = []
        for code in refinement_result.refined_codes:
            confidence_pct = int(code.confidence_score * 100)
            code_details.append({
                "code": code.icd_code,
                "enhanced_description": code.enhanced_description,
                "confidence": f"{confidence_pct}%"
            })
        
        return {
            "title": title,
            "enriched_keywords": enrichment_result.enriched_keywords,
            "diagnosis_codes": diagnosis_codes,
            "total_codes_found": len(refinement_result.refined_codes),
            "code_details": code_details,
            "clinical_summary": refinement_result.clinical_summary
        }
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")


@app.post("/process-spreadsheet")
async def process_spreadsheet_document(file: UploadFile = File(...)) -> SpreadsheetRow:
    """
    NEW ENHANCED ENDPOINT: Clean two-step AI validation process
    
    Workflow:
    1. Extract title and AI enrichment
    2. Direct vector search (400-500 codes, no chapter limits)
    3. Step 1: Initial selection (~50 relevant codes)
    4. Step 2: Clinical refinement (enhanced descriptions + confidence)
    5. Return structured spreadsheet row
    """
    try:
        # Step 1: Extract clean filename (strip any path)
        clean_filename = file.filename.split('/')[-1] if '/' in file.filename else file.filename
        
        # Step 2: Extract title
        file_content = await file.read()
        title = extract_title_from_file(file_content, clean_filename)
        
        if not title:
            raise HTTPException(
                status_code=400,
                detail=f"Could not extract title from file: {clean_filename}"
            )
        
        # Step 3: Extract full document content for comprehensive keyword analysis
        full_content = extract_text_from_file(file_content, clean_filename)
        
        # Step 4: AI Title Enrichment for better vector search
        enrichment_result = title_enricher.enrich_title(title)
        search_text = f"{title} {enrichment_result.enriched_keywords}"
        
        # Step 5: AI Metadata Generation with full document content
        metadata_result = title_enricher.generate_metadata(title, full_content)
        
        # Step 6: Vector Search 
        # Search all codes directly - let AI decide what's relevant
        all_candidates = vectorstore.search_all_codes(search_text, top_k=150)
        
        if not all_candidates:
            raise HTTPException(
                status_code=404,
                detail="No relevant codes found in vector search"
            )
        
        # Step 7: Enhanced Multi-Stage AI Validation Process
        
        # NEW: Enhanced multi-stage process with hierarchy enrichment
        refinement_result = await validator.enhanced_multi_stage_validation(
            medical_text=search_text,
            initial_candidates=all_candidates,
            vectorstore=vectorstore
        )
        
        if not refinement_result.refined_codes:
            raise HTTPException(
                status_code=404,
                detail="No codes validated in clinical refinement step"
            )
        
        # Step 8: Format Results for Spreadsheet Output
        root_codes = extract_root_codes_simple(refinement_result.refined_codes)
        hierarchy_codes = extract_hierarchy_codes_simple(refinement_result.refined_codes)
        code_descriptions = format_enhanced_descriptions(refinement_result.refined_codes)
        code_scores = format_confidence_scores(refinement_result.refined_codes)
        
        # NEW: Generate structured data for enhanced export capabilities
        structured_codes = format_structured_codes(refinement_result.refined_codes)
        
        # Generate unique name
        unique_name = title.replace(" ", "_").replace(",", "").replace("(", "").replace(")", "")
        
        # Step 9: Return Clean Spreadsheet Row with Enhanced Data
        return SpreadsheetRow(
            filepath=file.filename,
            title=title,
            icd_code_root=root_codes,
            icd_code_hierarchy=hierarchy_codes,
            details_description=code_descriptions,
            details_score=code_scores,
            gender=metadata_result.gender,
            unique_name=unique_name,
            keywords=metadata_result.keywords,
            diagnosis_codes=hierarchy_codes,  # For backward compatibility
            cpt_codes="",  # Leave 
            language="English",
            source="AI Medical Coding System v2.0",
            document_type="Patient Education",
            icd_codes_structured=structured_codes  # NEW: Enhanced structured data
        )
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")


# =====  HELPER FUNCTIONS =====

def extract_root_codes_simple(codes: List[RefinedCodeValidation]) -> str:
    """Extract unique root codes (first 3 characters)"""
    root_codes = set()
    for code in codes:
        if len(code.icd_code) >= 3:
            root_code = code.icd_code[:3]
            root_codes.add(root_code)
    return ", ".join(sorted(root_codes))

def extract_hierarchy_codes_simple(codes: List[RefinedCodeValidation]) -> str:
    """Extract full hierarchy codes"""
    hierarchy_codes = [code.icd_code for code in codes]
    return ", ".join(hierarchy_codes)

def format_enhanced_descriptions(codes: List[RefinedCodeValidation]) -> str:
    """Format enhanced descriptions as 'CODE: Enhanced Description'"""
    descriptions = []
    for code in codes:
        descriptions.append(f"{code.icd_code}: {code.enhanced_description}")
    return ", ".join(descriptions)

def format_confidence_scores(codes: List[RefinedCodeValidation]) -> str:
    """Format confidence scores as 'CODE: XX%'"""
    scores = []
    for code in codes:
        percentage = int(code.confidence_score * 100)
        scores.append(f"{code.icd_code}: {percentage}%")
    return ", ".join(scores)

def format_structured_codes(codes: List[RefinedCodeValidation]) -> List[Dict[str, Any]]:
    """Format codes as structured data for enhanced export capabilities"""
    structured_codes = []
    for code in codes:
        structured_codes.append({
            "icd_code": code.icd_code,
            "root_code": code.icd_code[:3] if len(code.icd_code) >= 3 else code.icd_code,
            "enhanced_description": code.enhanced_description,
            "original_description": code.original_description,
            "confidence_score": code.confidence_score,
            "confidence_percentage": f"{int(code.confidence_score * 100)}%"
        })
    return structured_codes


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 