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
        elif file_extension in ['html', 'htm']:
            return _extract_from_html(file_content)
        else:
            return None
    except Exception as e:
        print(f"Error extracting text from {filename}: {str(e)}")
        return None


def extract_title_from_file(file_content: bytes, filename: str) -> Optional[str]:
    """
    Extract title from filename by removing date and extension
    
    Expected formats:
    - PDF: "Macular Degeneration, Age-Related 05-24-2025.pdf" → "Macular Degeneration, Age-Related"
    - HTML: "Allergens, Bedroom Dust Mites.html" → "Allergens, Bedroom Dust Mites"
    
    Args:
        file_content: Raw file bytes (unused but kept for compatibility)
        filename: Original filename with extension
        
    Returns:
        str: Clean title without date and extension
    """
    try:
        # Remove file extension
        name_without_ext = filename.rsplit('.', 1)[0]
        
        # Remove date pattern (MM-DD-YYYY) from the end
        import re
        # Pattern matches: space + MM-DD-YYYY at the end
        clean_title = re.sub(r'\s+\d{2}-\d{2}-\d{4}$', '', name_without_ext)
        
        return clean_title.strip() if clean_title else name_without_ext
        
    except Exception as e:
        print(f"Error extracting title from filename {filename}: {str(e)}")
        return filename.rsplit('.', 1)[0]  # Fallback: just remove extension





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


def _extract_from_html(file_content: bytes) -> str:
    """Extract text from HTML file"""
    try:
        from bs4 import BeautifulSoup
        
        # Try different encodings
        for encoding in ['utf-8', 'latin-1', 'cp1252']:
            try:
                html_content = file_content.decode(encoding)
                break
            except UnicodeDecodeError:
                continue
        else:
            return ""
        
        soup = BeautifulSoup(html_content, 'html.parser')
        return soup.get_text().strip()
        
    except ImportError:
        raise Exception("BeautifulSoup not installed. Install with: pip install beautifulsoup4")


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