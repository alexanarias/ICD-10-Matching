from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi import Request
import medspacy
from medspacy.target_matcher import TargetRule
import pandas as pd
import pdfplumber
import spacy
from pathlib import Path
import json
from typing import List, Dict
import csv
from io import BytesIO
import os
import fitz
from collections import defaultdict
import re

app = FastAPI(
    title="Medical Text Analysis API",
    description="API for extracting medical terms from PDFs and matching ICD-10 codes",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount templates directory
templates = Jinja2Templates(directory="templates")

# Initialize medspacy with custom configuration
nlp = medspacy.load()

# Load ICD-10 codes from CSV
def load_icd10_codes():
    icd10_df = pd.read_csv("diagnosis.csv", encoding='utf-8')
    return icd10_df

# Initialize ICD-10 codes
icd10_data = load_icd10_codes()

def preprocess_text(text: str) -> str:
    """Preprocess text for better matching"""
    # Convert to lowercase
    text = text.lower()
    # Normalize whitespace
    text = ' '.join(text.split())
    # Handle basic medical abbreviations
    text = text.replace('w/', 'with ')
    text = text.replace('h/o', 'history of ')
    text = text.replace('s/p', 'status post ')
    # Remove special characters but keep word boundaries
    text = re.sub(r'[^a-z0-9\s/-]', ' ', text)
    return text.strip()

def is_valid_term_match(text: str, term: str) -> bool:
    """
    Validate if a term match is legitimate by checking word boundaries and context
    """
    text = text.lower()
    term = term.lower()
    
    # Create word boundary pattern
    pattern = r'\b' + re.escape(term) + r'(?:s|es)?\b'
    match = re.search(pattern, text)
    
    if not match:
        return False
        
    # Get context around the match
    start = max(0, match.start() - 30)  # Look at 30 chars before
    end = min(len(text), match.end() + 30)  # and 30 chars after
    context = text[start:end]
    
    # Check for negation words
    negation_words = ['no', 'not', 'none', 'negative', 'denies', 'without', 'ruled out']
    before_term = context[:context.find(term)].split()
    
    # Check last 3 words before term for negations
    if any(word in negation_words for word in before_term[-3:]):
        return False
        
    return True

def detect_document_domain(text: str) -> str:
    """Detect the medical domain of the document based on its content"""
    rules_dir = Path("rules")
    domains = {}
    
    preprocessed_text = preprocess_text(text)
    
    for rule_file in rules_dir.glob("*.json"):
        with open(rule_file, 'r') as f:
            rule_data = json.load(f)
            domain_name = rule_data['name']
            term_count = 0
            term_weights = {
                'DIAGNOSIS': 2.0,  # Give more weight to diagnosis terms
                'PROCEDURE': 1.5,
                'SYMPTOMS': 1.2,
                'TREATMENT': 1.0
            }
            
            # Count weighted occurrences of terms from each domain
            for category in rule_data['rules']:
                category_name = category.get('category', '')
                weight = term_weights.get(category_name, 1.0)
                
                for term in category['terms']:
                    if is_valid_term_match(preprocessed_text, term):
                        term_count += weight
            
            if term_count > 0:
                domains[rule_file.stem] = term_count
    
    # Return the domain with the most weighted matches
    if domains:
        best_domain = max(domains.items(), key=lambda x: x[1])
        return best_domain[0]
    return None

def get_intro_between_h1_and_h2(pdf_bytes: bytes) -> str:
    """Extract meaningful text from the first page of a PDF.
    
    This function attempts to extract text in a smart way:
    1. Gets all text blocks from the first page
    2. Identifies the title (largest font)
    3. Includes the title in the output
    4. Collects all text after the title that's not a header
    5. Removes periods from the text
    """
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    if len(doc) == 0:
        doc.close()
        return ""
    
    # Get first page
    page = doc[0]
    blocks = page.get_text("dict")["blocks"]
    
    # Collect all text spans with their properties
    spans_with_props = []
    for block in blocks:
        for line in block.get("lines", []):
            for span in line.get("spans", []):
                text = span["text"].strip()
                if text:
                    # Remove periods from the text
                    text = text.replace('.', '')
                    spans_with_props.append({
                        "text": text,
                        "size": span["size"],
                        "font": span["font"],
                        "color": span["color"],
                        "flags": span["flags"]  # Bold, italic, etc.
                    })
    
    if not spans_with_props:
        doc.close()
        return ""

    # Find the title (largest font size)
    sizes = sorted({s["size"] for s in spans_with_props}, reverse=True)
    title_size = sizes[0]

    # Collect meaningful text including title
    content = []
    title_found = False
    current_section = []
    title_text = ""

    for span in spans_with_props:
        text = span["text"]
        size = span["size"]

        if size == title_size and not title_found:
            title_text = text
            title_found = True
            continue

        if title_found:
            # Skip if it looks like a header (significantly larger than average text)
            if size > title_size * 0.7:
                if current_section:
                    content.append(" ".join(current_section))
                    current_section = []
            else:
                current_section.append(text)

    if current_section:
        content.append(" ".join(current_section))

    # Combine title with content
    final_text = title_text + " - " + " ".join(content) if content else title_text

    doc.close()
    return final_text.strip()

def extract_whole_content(pdf_bytes: bytes) -> str:
    """Extract all text content from a PDF while preserving structure.
    
    This function:
    1. Extracts text from all pages
    2. Preserves text hierarchy (headers, body text)
    3. Maintains paragraph structure
    4. Handles multiple columns
    5. Removes redundant whitespace
    """
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    if len(doc) == 0:
        doc.close()
        return ""
    
    full_content = []
    
    for page_num in range(len(doc)):
        page = doc[page_num]
        
        # Get blocks with their properties
        blocks = page.get_text("dict")["blocks"]
        
        # Sort blocks by vertical position (y0) and then horizontal position (x0)
        blocks.sort(key=lambda b: (b["bbox"][1], b["bbox"][0]))
        
        page_content = []
        current_y = None
        current_line = []
        
        for block in blocks:
            if block.get("type") == 0:  # Text block
                for line in block.get("lines", []):
                    y_pos = line["bbox"][1]  # Vertical position
                    
                    # Start a new line if y position changes significantly
                    if current_y is not None and abs(y_pos - current_y) > 5:
                        if current_line:
                            page_content.append(" ".join(current_line))
                            current_line = []
                    
                    current_y = y_pos
                    
                    # Extract text from spans
                    line_text = []
                    for span in line.get("spans", []):
                        text = span["text"].strip()
                        if text:
                            # Check if this is likely a header based on font size
                            if span.get("size", 0) > 12:  # Adjust threshold as needed
                                text = f"\n{text}\n"
                            line_text.append(text)
                    
                    if line_text:
                        current_line.extend(line_text)
        
        # Add any remaining line
        if current_line:
            page_content.append(" ".join(current_line))
        
        # Join page content with appropriate spacing
        if page_content:
            page_text = "\n".join(page_content)
            # Add page break if not the last page
            if page_num < len(doc) - 1:
                page_text += "\n\n--- Page Break ---\n\n"
            full_content.append(page_text)
    
    doc.close()
    
    # Join all pages and clean up whitespace
    final_text = "\n".join(full_content)
    # Clean up multiple newlines while preserving paragraph structure
    final_text = re.sub(r'\n{3,}', '\n\n', final_text)
    # Clean up multiple spaces
    final_text = re.sub(r' {2,}', ' ', final_text)
    
    return final_text.strip()

def load_rules_from_json(domain: str) -> list:
    """Load and convert rules from JSON to TargetRule format"""
    rules = []
    try:
        with open(f"rules/{domain}.json", 'r') as f:
            data = json.load(f)
            for category in data['rules']:
                category_name = category['category']
                for term in category['terms']:
                    # Create pattern with word boundaries
                    pattern = r'\b' + re.escape(term.lower()) + r'(?:s|es)?\b'
                    rules.append(TargetRule(
                        literal=term,
                        category=category_name,
                        pattern=pattern
                    ))
    except Exception as e:
        print(f"Error loading rules for domain {domain}: {str(e)}")
        return []
    return rules

def extract_medical_terms(text: str, domain: str = None) -> list:
    """Extract medical terms using medspacy"""
    # Preprocess the text
    preprocessed_text = preprocess_text(text)
    print(preprocessed_text)
    
    # Remove default pipelines that might interfere
    if "medspacy_pyrush" in nlp.pipe_names:
        nlp.remove_pipe("medspacy_pyrush")
    if "medspacy_context" in nlp.pipe_names:
        nlp.remove_pipe("medspacy_context")
    
    # Ensure we have the target matcher
    if "medspacy_target_matcher" not in nlp.pipe_names:
        target_matcher = nlp.add_pipe("medspacy_target_matcher")
    else:
        target_matcher = nlp.get_pipe("medspacy_target_matcher")
    
    # Detect domain if not provided
    if domain is None:
        domain = detect_document_domain(preprocessed_text)
    
    # Load and add rules
    rules = load_rules_from_json(domain)
    target_matcher.add(rules)
    
    # Process the text
    doc = nlp(preprocessed_text)
    
    # Track unique terms
    unique_terms = {}
    
    # Process all entities
    for ent in doc.ents:
        # Create a unique key for the term
        term_key = (ent.text.lower(), ent.label_)
        
        # Simple confidence scoring
        confidence = 1.0
        
        # Boost multi-word terms
        word_count = len(ent.text.split())
        if word_count > 1:
            confidence *= 1.2
        
        # Category weights
        category_weights = {
            'DIAGNOSIS': 1.2,
            'PROCEDURE': 1.1,
            'SYMPTOMS': 1.0,
            'TREATMENT': 1.0
        }
        confidence *= category_weights.get(ent.label_, 0.9)
        
        # Only add terms that pass validation
        if is_valid_term_match(preprocessed_text, ent.text):
            if term_key not in unique_terms or confidence > unique_terms[term_key]['confidence']:
                unique_terms[term_key] = {
                    'text': ent.text,
                    'label': ent.label_,
                    'start': ent.start_char,
                    'end': ent.end_char,
                    'confidence': confidence
                }
    
    # Convert to list and sort by confidence
    entities = list(unique_terms.values())
    entities.sort(key=lambda x: x['confidence'], reverse=True)
    
    return entities

def match_icd10_codes(term: str) -> List[Dict]:
    """Match medical terms to ICD-10 codes"""
    try:
        matches = []
        
        try:
            # Search in both short and long descriptions
            exact_matches_short = icd10_data[icd10_data['ShortDescription'].str.lower() == term.lower()]
            exact_matches_long = icd10_data[icd10_data['LongDescription'].str.lower() == term.lower()]
            exact_matches = pd.concat([exact_matches_short, exact_matches_long]).drop_duplicates()
            
            for _, row in exact_matches.iterrows():
                match = {
                    "code": str(row['CodeWithSeparator']),
                    "description": str(row['LongDescription']),
                    "match_type": "exact"
                }
                matches.append(match)
            
            # If no exact matches, search for partial matches
            if not matches:
                # Search in both descriptions using case-insensitive partial matching
                partial_matches_short = icd10_data[icd10_data['ShortDescription'].str.lower().str.contains(term.lower(), na=False)]
                partial_matches_long = icd10_data[icd10_data['LongDescription'].str.lower().str.contains(term.lower(), na=False)]
                partial_matches = pd.concat([partial_matches_short, partial_matches_long]).drop_duplicates()
                
                # Limit the number of partial matches to avoid overwhelming results
                max_partial_matches = 10
                for _, row in partial_matches.head(max_partial_matches).iterrows():
                    match = {
                        "code": str(row['CodeWithSeparator']),
                        "description": str(row['LongDescription']),
                        "match_type": "partial"
                    }
                    matches.append(match)
        
        except AttributeError as e:
            return []
            
        return matches
        
    except Exception as e:
        import traceback
        return []

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Serve the main HTML page"""
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/analyze")
async def analyze_pdf(file: UploadFile = File(...)):
    """
    Analyze uploaded PDF file and extract medical terms with ICD-10 codes
    """
    if not file.filename.endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are allowed")
    
    try:
        # Read PDF content
        contents = await file.read()
        
        text = extract_whole_content(contents)
        
        # Extract medical terms
        medical_terms = extract_medical_terms(text)
        
        # Match ICD-10 codes for each term
        results = []
        for term in medical_terms:
            icd10_matches = match_icd10_codes(term['text'])
            result = {
                "term": term['text'],
                "type": term['label'],
                "icd10_matches": icd10_matches if icd10_matches else [],  # Ensure it's always a list
                "validation_status": "pending"  # Initial status
            }
            results.append(result)
        
        return JSONResponse(content={
            "filename": file.filename,
            "total_terms": len(medical_terms),
            "results": results
        })
        
    except Exception as e:
        import traceback
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 