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
from extract_subect import extract_subject_from_title, medical_abbreviations

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
    """Load and preprocess ICD-10 codes from CSV file"""
    icd10_df = pd.read_csv("diagnosis.csv", encoding='utf-8')
    
    # Create normalized versions of descriptions for better matching
    icd10_df['normalized_short'] = icd10_df['ShortDescription'].str.lower().str.strip()
    icd10_df['normalized_long'] = icd10_df['LongDescription'].str.lower().str.strip()
    
    # Create word sets for each description
    icd10_df['short_words'] = icd10_df['normalized_short'].fillna('').apply(lambda x: set(x.split()))
    icd10_df['long_words'] = icd10_df['normalized_long'].fillna('').apply(lambda x: set(x.split()))
    
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
    """Extract text between H1 and H2 tags from a PDF.
    
    This function:
    1. Gets all text blocks from the first page
    2. Identifies H1 (largest font size) as the title (can be multiple lines)
    3. Finds the next largest font size as potential H2 or image/section break
    4. Returns the title (H1) and text between H1 and first section break
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
        # Skip image blocks
        if block.get("type") == 1:  # Image block
            spans_with_props.append({
                "text": "IMAGE_BREAK",
                "size": 0,
                "y": block["bbox"][1],
                "block_id": id(block),
                "is_image": True
            })
            continue
            
        for line in block.get("lines", []):
            for span in line.get("spans", []):
                text = span["text"].strip()
                if text:
                    spans_with_props.append({
                        "text": text,
                        "size": span["size"],
                        "font": span["font"],
                        "color": span["color"],
                        "flags": span["flags"],
                        "y": line["bbox"][1],
                        "block_id": id(block),
                        "is_image": False,
                        "bbox": line["bbox"]  # Store full bounding box for better spacing detection
                    })
    
    if not spans_with_props:
        doc.close()
        return ""

    # Sort spans by Y position to maintain reading order
    spans_with_props.sort(key=lambda x: x["y"])

    # Find unique font sizes and sort them
    text_sizes = sorted({s["size"] for s in spans_with_props if not s.get("is_image")}, reverse=True)
    if len(text_sizes) < 2:
        doc.close()
        return " ".join(s["text"] for s in spans_with_props if not s.get("is_image"))

    # H1 is the largest font size
    h1_size = text_sizes[0]

    # Extract title and content
    title_parts = []
    content_blocks = []
    current_block = []
    found_h1 = False
    last_block_id = None
    last_y = None
    last_bbox = None

    for span in spans_with_props:
        text = span["text"]
        size = span.get("size", 0)
        block_id = span["block_id"]
        y = span["y"]
        is_image = span.get("is_image", False)
        bbox = span.get("bbox")

        # Handle H1 (title)
        if not is_image and size >= h1_size * 0.95:
            if not title_parts or block_id == last_block_id:
                title_parts.append(text)
                found_h1 = True
                last_block_id = block_id
            continue

        # After title, collect content until we hit an image or a clear section break
        if found_h1:
            if is_image:
                # Stop at first image
                if current_block:
                    content_blocks.append(current_block)
                break
            elif text.startswith("There are") or text.startswith("These are") or text.startswith("Types of"):
                # Stop at section transitions
                if current_block:
                    content_blocks.append(current_block)
                break
            else:
                # Check if this is a continuation of the current block or a new block
                if last_bbox is not None:
                    y_gap = bbox[1] - last_bbox[3]  # Distance between bottom of last line and top of current line
                    if y_gap > 15:  # Large gap indicates new paragraph
                        if current_block:
                            content_blocks.append(current_block)
                            current_block = []
                
                current_block.append(text)
                last_bbox = bbox
                last_y = y

    # Add any remaining block
    if current_block:
        content_blocks.append(current_block)

    doc.close()

    # Combine title parts and content blocks
    result = " ".join(title_parts) + "\n\n"
    
    # Join blocks with appropriate spacing
    content_text = []
    for block in content_blocks:
        block_text = " ".join(block)
        if block_text.strip():
            content_text.append(block_text)
    
    result += "\n\n".join(content_text)
    return result

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

def extract_keywords_from_paragraph(text: str) -> List[str]:
    """Extract specific medical terms, symptoms, and conditions from text using defined rules
    
    Args:
        text (str): The paragraph text to analyze
        
    Returns:
        List[str]: List of specific medical terms and symptoms
    """
    keywords = []
    
    # Clean and normalize text
    text = text.lower().strip()
    
    # Load rules from all JSON files in rules directory
    rules_dir = Path("rules")
    for rule_file in rules_dir.glob("*.json"):
        try:
            with open(rule_file, 'r') as f:
                rule_data = json.load(f)
                # Process each category in the rules
                for category in rule_data['rules']:
                    category_name = category['category']
                    # Match each term in the category
                    for term in category['terms']:
                        # Create word boundary pattern
                        pattern = r'\b' + re.escape(term.lower()) + r'(?:s|es)?\b'
                        if re.search(pattern, text):
                            keywords.append(term)
        except Exception as e:
            print(f"Error loading rules from {rule_file}: {str(e)}")
            continue
    
    # Clean up extracted keywords
    cleaned_keywords = []
    for keyword in keywords:
        # Remove leading/trailing special characters and spaces
        keyword = re.sub(r'^[^a-zA-Z0-9]+|[^a-zA-Z0-9]+$', '', keyword)
        # Remove extra whitespace
        keyword = ' '.join(keyword.split())
        # Only keep keywords with 2 or more characters
        if len(keyword) >= 2:
            cleaned_keywords.append(keyword)
    
    # Remove duplicates while preserving order
    seen = set()
    unique_keywords = []
    for keyword in cleaned_keywords:
        if keyword not in seen:
            seen.add(keyword)
            unique_keywords.append(keyword)
    
    return unique_keywords

def extract_medical_terms(text: str, domain: str = None) -> list:
    """Extract medical terms using medspacy"""
    # Preprocess the text
    preprocessed_text = preprocess_text(text)
    print(preprocessed_text)
    
    # Extract keywords first
    keywords = extract_keywords_from_paragraph(preprocessed_text)
    
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
        # If still None, use general_medical as default
        if domain is None:
            domain = "general_medical"
    
    # Load and add rules
    rules = load_rules_from_json(domain)
    target_matcher.add(rules)
    
    # Process the text
    doc = nlp(preprocessed_text)
    
    # Track unique terms
    unique_terms = {}
    
    # Add extracted keywords as terms
    for keyword in keywords:
        term_key = (keyword.lower(), 'SYMPTOM')
        if term_key not in unique_terms:
            unique_terms[term_key] = {
                'text': keyword,
                'label': 'SYMPTOM',
                'confidence': 0.8
            }
    
    # Process all entities from medspacy
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
                    'confidence': confidence
                }
    
    # Convert to list and sort by confidence
    entities = list(unique_terms.values())
    entities.sort(key=lambda x: x['confidence'], reverse=True)
    
    return entities

def normalize_medical_term(term: str) -> List[str]:
    """Normalize medical term and generate variants for matching
    
    Args:
        term (str): Medical term to normalize
        
    Returns:
        List[str]: List of normalized term variants
    """
    if not term:
        return []
        
    term = term.lower().strip()
    variants = [term]
    
    # Common word replacements in medical terminology
    replacements = {
        'disorder': ['disease', 'condition', 'syndrome'],
        'disease': ['disorder', 'condition', 'syndrome'],
        'syndrome': ['disorder', 'disease', 'condition'],
        'acute': ['chronic', 'recurrent'],
        'chronic': ['acute', 'recurrent'],
        'recurring': ['recurrent', 'chronic'],
        'inability': ['impairment', 'difficulty'],
        'impairment': ['inability', 'difficulty'],
        'surgery': ['surgical procedure', 'operation', 'surgical treatment'],
        'surgical': ['surgery', 'operative'],
        'arthritis': ['arthritic condition', 'joint inflammation'],
        'joint': ['articular'],
        'basal': ['base', 'basilar']
    }
    
    # Generate variants with common word replacements
    words = term.split()
    for i, word in enumerate(words):
        if word in replacements:
            for replacement in replacements[word]:
                new_words = words.copy()
                new_words[i] = replacement
                variants.append(' '.join(new_words))
    
    # Handle compound terms
    if 'basal joint' in term:
        variants.append('carpometacarpal joint')
        variants.append('cmc joint')
        variants.append('thumb joint')
        variants.append('thumb base joint')
    
    if 'joint arthritis' in term:
        variants.append('osteoarthritis')
        variants.append('degenerative joint disease')
        variants.append('arthritic joint')
    
    # Handle surgical terms
    if any(word in term for word in ['surgery', 'surgical', 'operation']):
        surgical_variants = [
            'surgical treatment',
            'surgical management',
            'operative treatment',
            'surgical procedure',
            'operative procedure'
        ]
        for variant in surgical_variants:
            if 'options' in term:
                variants.append(f"{variant} options")
            else:
                variants.append(variant)
    
    # Remove common medical prefixes/suffixes for additional variants
    prefixes = ['acute ', 'chronic ', 'recurrent ', 'severe ']
    suffixes = [' disorder', ' disease', ' syndrome', ' condition', ' options', ' treatment']
    
    # Generate core medical term variants
    for prefix in prefixes:
        if term.startswith(prefix):
            base_term = term[len(prefix):]
            variants.append(base_term)
            # Also add other prefixes to the base term
            for other_prefix in prefixes:
                if other_prefix != prefix:
                    variants.append(other_prefix + base_term)
    
    for suffix in suffixes:
        if term.endswith(suffix):
            base_term = term[:-len(suffix)]
            variants.append(base_term)
            # Also add other suffixes to the base term
            for other_suffix in suffixes:
                if other_suffix != suffix:
                    variants.append(base_term + other_suffix)
    
    # Clean and normalize variants
    cleaned_variants = set()
    for variant in variants:
        # Remove extra whitespace
        cleaned = ' '.join(variant.split())
        # Remove any empty strings
        if cleaned:
            cleaned_variants.add(cleaned)
    
    return list(cleaned_variants)

def match_icd10_codes(term: str) -> List[Dict]:
    """Match medical terms to ICD-10 codes with improved accuracy and coverage
    
    Args:
        term (str): The medical term to match against ICD-10 codes
        
    Returns:
        List[Dict]: List of matching ICD-10 codes with their descriptions and match types
    """
    if not term or not isinstance(term, str):
        return []
        
    try:
        matches = []
        term = term.lower().strip()
        
        # Generate variants for better matching
        term_variants = normalize_medical_term(term)
        all_term_words = set()
        for variant in term_variants:
            all_term_words.update(variant.split())
        
        # 1. Direct exact matches with variants
        exact_matches = icd10_data[
            icd10_data['normalized_long'].isin(term_variants) |
            icd10_data['normalized_short'].isin(term_variants)
        ]
        
        for _, row in exact_matches.iterrows():
            matches.append({
                "code": str(row['CodeWithSeparator']),
                "description": str(row['LongDescription']),
                "match_type": "exact",
                "confidence": 1.0
            })
        
        # 2. Specific handling for surgical/treatment terms
        if any(surg_term in term.lower() for surg_term in ['surgery', 'surgical', 'operation', 'procedure']):
            # Look for procedure codes related to the condition
            condition_terms = [w for w in term.split() if w not in ['surgery', 'surgical', 'operation', 'procedure', 'options', 'for']]
            if condition_terms:
                condition_search = ' '.join(condition_terms)
                procedure_matches = icd10_data[
                    (icd10_data['normalized_long'].str.contains('|'.join(condition_terms), case=False, na=False)) &
                    (icd10_data['CodeWithSeparator'].str.startswith(('0', '3', '4')))  # Procedure code ranges
                ]
                
                for _, row in procedure_matches.iterrows():
                    matches.append({
                        "code": str(row['CodeWithSeparator']),
                        "description": str(row['LongDescription']),
                        "match_type": "procedure",
                        "confidence": 0.9
                    })
        
        # 3. Partial matches with improved scoring
        for _, row in icd10_data.iterrows():
            desc_words = set(row['normalized_long'].split())
            word_overlap = len(all_term_words.intersection(desc_words))
            
            if word_overlap > 0:
                # Calculate base confidence
                confidence = 0.3 + (word_overlap / len(all_term_words)) * 0.6
                
                # Boost confidence for specific conditions
                if 'arthritis' in term and 'arthritis' in row['normalized_long']:
                    confidence = min(1.0, confidence + 0.2)
                if 'joint' in term and 'joint' in row['normalized_long']:
                    confidence = min(1.0, confidence + 0.1)
                if 'basal' in term and any(w in row['normalized_long'] for w in ['basal', 'base', 'thumb', 'carpometacarpal']):
                    confidence = min(1.0, confidence + 0.2)
                
                # Add match if confidence is high enough
                if confidence > 0.3:
                    matches.append({
                        "code": str(row['CodeWithSeparator']),
                        "description": str(row['LongDescription']),
                        "match_type": "partial",
                        "confidence": round(confidence, 2)
                    })
        
        # 4. Related condition matches
        code_prefix = None
        for match in matches:
            if match['confidence'] >= 0.8:
                code = match['code']
                prefix = code.split('.')[0]
                if len(prefix) >= 3:
                    code_prefix = prefix[:3]
                    break
        
        if code_prefix:
            related_matches = icd10_data[
                icd10_data['CodeWithSeparator'].str.startswith(code_prefix)
            ]
            
            for _, row in related_matches.iterrows():
                code = str(row['CodeWithSeparator'])
                if not any(m['code'] == code for m in matches):
                    matches.append({
                        "code": code,
                        "description": str(row['LongDescription']),
                        "match_type": "related",
                        "confidence": 0.7
                    })
        
        # Sort by confidence and remove duplicates
        matches.sort(key=lambda x: x['confidence'], reverse=True)
        unique_matches = []
        seen_codes = set()
        
        for match in matches:
            if match['code'] not in seen_codes:
                seen_codes.add(match['code'])
                unique_matches.append(match)
        
        return unique_matches[:20]
        
    except Exception as e:
        print(f"Error in match_icd10_codes: {str(e)}")
        return []

def match_subject_to_icd10(subject: str) -> List[Dict]:
    """Match medical subject directly to ICD-10 codes with comprehensive category matching
    
    Args:
        subject (str): The medical subject/category to match
        
    Returns:
        List[Dict]: List of matching ICD-10 codes with their descriptions and match types
    """
    if not subject or not isinstance(subject, str):
        return []
        
    try:
        all_matches = []
        subject = subject.lower().strip()
        
        # 1. First try direct keyword matching
        direct_matches = match_icd10_codes(subject)
        all_matches.extend(direct_matches)
        
        # 2. Try matching individual words if subject has multiple words
        subject_words = subject.split()
        if len(subject_words) > 1:
            for word in subject_words:
                if len(word) > 3:  # Only try matching significant words
                    word_matches = match_icd10_codes(word)
                    # Add matches with reduced confidence
                    for match in word_matches:
                        match['confidence'] = round(match['confidence'] * 0.8, 2)  # Reduce confidence for word matches
                        all_matches.append(match)
        
        # 3. Try category-based matching
        category_matches = []
        for category, info in direct_mappings.items():
            for code_range, keywords in info['codes'].items():
                if any(kw in subject for kw in keywords):
                    start_code, end_code = code_range.split('-')
                    matches_in_range = icd10_data[
                        (icd10_data['CodeWithSeparator'] >= start_code) & 
                        (icd10_data['CodeWithSeparator'] <= end_code)
                    ]
                    
                    for _, row in matches_in_range.iterrows():
                        # Calculate match quality
                        desc_words = set(row['LongDescription'].lower().split())
                        subject_words = set(subject.split())
                        word_overlap = len(subject_words.intersection(desc_words))
                        
                        # Calculate confidence
                        confidence = info['confidence']
                        if word_overlap > 0:
                            confidence = min(1.0, confidence + (word_overlap * 0.05))
                        
                        category_matches.append({
                            "code": str(row['CodeWithSeparator']),
                            "description": str(row['LongDescription']),
                            "match_type": "category",
                            "confidence": round(confidence, 2),
                            "category": category
                        })
        
        all_matches.extend(category_matches)
        
        # Sort by confidence and remove duplicates
        all_matches.sort(key=lambda x: x['confidence'], reverse=True)
        unique_matches = []
        seen_codes = set()
        
        for match in all_matches:
            if match['code'] not in seen_codes:
                seen_codes.add(match['code'])
                unique_matches.append(match)
        
        return unique_matches[:20]  # Return more matches
        
    except Exception as e:
        print(f"Error in match_subject_to_icd10: {str(e)}")
        return []

def extract_subjects_from_pdf(pdf_bytes: bytes) -> Dict[str, str]:
    """Extract subjects from PDF title and first paragraph"""
    # Get title and first paragraph
    intro_text = get_intro_between_h1_and_h2(pdf_bytes)
    if not intro_text:
        return {"error": "Could not extract text from PDF"}
    
    # Split into lines to get title
    lines = intro_text.split('\n')
    if not lines:
        return {"error": "No text found in PDF"}
    
    title = lines[0].strip()
    
    # Extract subject from title
    title_subject = extract_subject_from_title(title, get_full_form=True)
    
    # Look for abbreviations in the first paragraph
    first_paragraph = ' '.join(lines[1:]).strip()
    found_abbreviations = []
    
    for abbrev, full_form in medical_abbreviations.items():
        if abbrev in first_paragraph and f"{abbrev} (" not in first_paragraph:
            # Check if the full form appears before the abbreviation
            full_form_pos = first_paragraph.lower().find(full_form.lower())
            abbrev_pos = first_paragraph.find(abbrev)
            
            if full_form_pos != -1 and full_form_pos < abbrev_pos:
                found_abbreviations.append(f"{abbrev} ({full_form})")
    
    return {
        "title": title,
        "title_subject": title_subject,
        "found_abbreviations": found_abbreviations,
        "first_paragraph": first_paragraph
    }

def validate_icd10_correlation(text: str, icd10_matches: List[Dict]) -> List[Dict]:
    """Validate if the ICD-10 codes actually correlate with the article content
    
    Args:
        text (str): The article text
        icd10_matches (List[Dict]): List of potential ICD-10 matches
        
    Returns:
        List[Dict]: List of validated ICD-10 matches with correlation scores
    """
    validated_matches = []
    text = text.lower()
    
    for match in icd10_matches:
        correlation_score = 0.0
        code_desc = match['description'].lower()
        
        # Check if code description keywords appear in text
        desc_words = set(code_desc.split())
        text_words = set(text.split())
        word_overlap = len(desc_words.intersection(text_words))
        
        if word_overlap > 0:
            # Calculate correlation score based on word overlap and position
            correlation_score = min(0.3 + (word_overlap * 0.1), 1.0)
            
            # Check if description appears near the beginning of the text
            if any(sent.strip().startswith(code_desc[:20]) for sent in text.split('.')):
                correlation_score += 0.2
            
            # Add validation result
            validated_match = match.copy()
            validated_match['correlation_score'] = round(correlation_score, 2)
            validated_match['is_validated'] = correlation_score >= 0.5
            validated_matches.append(validated_match)
    
    # Sort by correlation score
    validated_matches.sort(key=lambda x: x['correlation_score'], reverse=True)
    return validated_matches

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Serve the main HTML page"""
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/analyze")
async def analyze_pdf(file: UploadFile = File(...)):
    """
    Analyze uploaded PDF file following the process:
    1. Extract subject from title
    2. Match ICD-10 codes for subject
    3. Extract keywords from first paragraph
    4. Match ICD-10 codes for keywords
    5. Validate correlations
    """
    if not file.filename.endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are allowed")
    
    try:
        # Read PDF content
        contents = await file.read()
        
        # Step 1: Extract subject from title
        subjects_info = extract_subjects_from_pdf(contents)
        title_subject = subjects_info.get("title_subject", "")
        first_paragraph = subjects_info.get("first_paragraph", "")
        
        # Step 2: Match ICD-10 codes for subject
        subject_icd10_matches = []
        if title_subject:
            subject_icd10_matches = match_subject_to_icd10(title_subject)
        
        # Step 3: Extract keywords from first paragraph
        keywords = extract_keywords_from_paragraph(first_paragraph)
        
        # Step 4: Match ICD-10 codes for keywords
        keyword_matches = []
        for keyword in keywords:
            matches = match_icd10_codes(keyword)
            if matches:
                keyword_matches.extend(matches)
        
        # Combine subject and keyword matches
        all_matches = subject_icd10_matches + keyword_matches
        
        # Step 5: Validate correlations
        validated_matches = validate_icd10_correlation(first_paragraph, all_matches)
        
        # Organize results by confidence and correlation
        high_confidence_matches = [m for m in validated_matches if m['correlation_score'] >= 0.7]
        medium_confidence_matches = [m for m in validated_matches if 0.5 <= m['correlation_score'] < 0.7]
        low_confidence_matches = [m for m in validated_matches if m['correlation_score'] < 0.5]
        
        return JSONResponse(content={
            "filename": file.filename,
            "title": subjects_info["title"],
            "title_subject": title_subject,
            "extracted_keywords": keywords,
            "icd10_matches": {
                "high_confidence": high_confidence_matches[:5],
                "medium_confidence": medium_confidence_matches[:3],
                "low_confidence": low_confidence_matches[:2]
            },
            "first_paragraph": first_paragraph,
            "analysis_summary": {
                "total_keywords_found": len(keywords),
                "total_matches_found": len(validated_matches),
                "high_confidence_count": len(high_confidence_matches),
                "validation_status": "complete"
            }
        })
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 