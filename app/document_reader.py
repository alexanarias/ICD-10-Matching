"""Document text extraction utilities"""

import io
from typing import Optional


def extract_text_from_file(file_content: bytes, filename: str) -> Optional[str]:
    """
    Extract text from uploaded document
    
    Args:
        file_content: Raw file bytes
        filename: Original filename with extension
        
    Returns:
        str: Extracted text or None if unsupported format
    """
    file_extension = filename.lower().split('.')[-1]
    
    try:
        if file_extension == 'txt':
            return _extract_from_txt(file_content)
        elif file_extension == 'pdf':
            return _extract_from_pdf(file_content)
        elif file_extension in ['doc', 'docx']:
            return _extract_from_docx(file_content)
        else:
            return None
    except Exception as e:
        print(f"Error extracting text from {filename}: {str(e)}")
        return None


def _extract_from_txt(file_content: bytes) -> str:
    """Extract text from plain text file"""
    try:
        return file_content.decode('utf-8')
    except UnicodeDecodeError:
        return file_content.decode('latin-1')


def _extract_from_pdf(file_content: bytes) -> str:
    """Extract text from PDF file"""
    try:
        import PyPDF2
        pdf_file = io.BytesIO(file_content)
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() + "\n"
        
        return text.strip()
    except ImportError:
        raise Exception("PyPDF2 not installed. Install with: pip install PyPDF2")


def _extract_from_docx(file_content: bytes) -> str:
    """Extract text from Word document"""
    try:
        from docx import Document
        doc_file = io.BytesIO(file_content)
        doc = Document(doc_file)
        
        text = ""
        for paragraph in doc.paragraphs:
            text += paragraph.text + "\n"
        
        return text.strip()
    except ImportError:
        raise Exception("python-docx not installed. Install with: pip install python-docx")


def validate_and_clean_text(text: str) -> str:
    """
    Validate and clean extracted text
    
    Args:
        text: Raw extracted text
        
    Returns:
        str: Cleaned text
    """
    if not text or not text.strip():
        raise ValueError("No text content found in document")
    
    # Basic cleaning
    cleaned = text.strip()
    
    # Remove excessive whitespace
    import re
    cleaned = re.sub(r'\s+', ' ', cleaned)
    
    # Check minimum length
    if len(cleaned) < 10:
        raise ValueError("Document text too short (minimum 10 characters)")
    
    return cleaned 