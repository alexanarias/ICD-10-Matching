import xml.etree.ElementTree as ET
import json
import re
from typing import Dict, List, Optional, Set
from dataclasses import dataclass, asdict
import argparse
from pathlib import Path


@dataclass
class ICDCode:
    """Structure to hold all ICD-10-CM code information"""
    code: str
    description: str
    chapter_number: Optional[int] = None
    chapter_name: str = ""
    chapter_range: str = ""
    section_id: str = ""
    section_name: str = ""
    category_code: str = ""
    category_description: str = ""
    parent_codes: List[str] = None
    
    # Text content for keywords
    includes_notes: List[str] = None
    excludes1_notes: List[str] = None
    excludes2_notes: List[str] = None
    inclusion_terms: List[str] = None
    use_additional_code: List[str] = None
    code_first: List[str] = None
    code_also: List[str] = None
    
    # Derived fields
    all_keywords: List[str] = None
    rich_text: str = ""  # For vector search - NO exclude terms
    is_category: bool = False
    is_subcategory: bool = False
    
    def __post_init__(self):
        if self.parent_codes is None:
            self.parent_codes = []
        if self.includes_notes is None:
            self.includes_notes = []
        if self.excludes1_notes is None:
            self.excludes1_notes = []
        if self.excludes2_notes is None:
            self.excludes2_notes = []
        if self.inclusion_terms is None:
            self.inclusion_terms = []
        if self.use_additional_code is None:
            self.use_additional_code = []
        if self.code_first is None:
            self.code_first = []
        if self.code_also is None:
            self.code_also = []
        if self.all_keywords is None:
            self.all_keywords = []


class ICD10CMParser:
    """Parser for ICD-10-CM Tabular XML file"""
    
    def __init__(self, xml_file_path: str):
        self.xml_file_path = xml_file_path
        self.codes: List[ICDCode] = []
        self.chapters: Dict[str, Dict] = {}  # Map section_id to chapter info
        self.code_map: Dict[str, ICDCode] = {} # Map code string to ICDCode object
        self.tree = None
        self.root = None
        
    def parse(self) -> List[ICDCode]:
        """Main parsing method"""
        print("Parsing %s..." % self.xml_file_path)
        
        # Parse XML
        self.tree = ET.parse(self.xml_file_path)
        self.root = self.tree.getroot()
        
        # --- Stage 1: Parse the XML hierarchically to build a context-rich map ---
        # First pass: Extract all chapter information and build section-to-chapter mapping
        self._extract_chapters()
        # Second pass: Process all sections to populate the initial code map
        self._process_sections()
            
        print(f"Parsed {len(self.code_map)} codes from XML with hierarchy.")

        # --- Stage 2: Augment with the complete list from the order file ---
        self._augment_with_order_file()
        print(f"Total codes after augmentation: {len(self.code_map)}")

        # --- Stage 3: Post-process all codes to generate keywords and rich text ---
        self._post_process_codes()

        self.codes = list(self.code_map.values())
        print("Extracted %d final codes" % len(self.codes))
        return self.codes
    
    def _extract_chapters(self):
        """Extract chapter information and build section mapping"""
        print("Extracting chapter information...")
        for chapter_elem in self.root.findall('chapter'):
            chapter_info = self._extract_chapter_info(chapter_elem)
            section_index = chapter_elem.find('sectionIndex')
            if section_index is not None:
                for section_ref in section_index.findall('sectionRef'):
                    section_id = section_ref.get('id', '')
                    if section_id:
                        self.chapters[section_id] = chapter_info
    
    def _process_sections(self):
        """Process all section elements, parsing the diag codes within them."""
        print("Processing sections from XML...")
        for section_elem in self.root.findall('.//section'):
            section_id = section_elem.get('id', '')
            chapter_info = self.chapters.get(section_id, {})
            
            if not chapter_info:
                continue
                
            section_info = self._extract_section_info(section_elem)
                
            for diag_elem in section_elem.findall('diag'):
                self._parse_diagnosis_xml(diag_elem, chapter_info, section_info)

    def _parse_diagnosis_xml(self, diag_elem: ET.Element, chapter_info: Dict, section_info: Dict, parent_codes: Optional[List[str]] = None):
        """Recursively parses a <diag> element from the XML and its children."""
        if parent_codes is None:
            parent_codes = []
            
        code_elem = diag_elem.find('name')
        desc_elem = diag_elem.find('desc')
        
        if code_elem is None or desc_elem is None or not code_elem.text:
            return
            
        code = code_elem.text.strip()
        description = desc_elem.text.strip()
        
        if code in self.code_map:
            return

        is_category = len(code) == 3
        is_subcategory = not is_category

        icd_code = ICDCode(
            code=code,
            description=description,
            chapter_number=chapter_info.get('number'),
            chapter_name=chapter_info.get('name', ''),
            section_id=section_info.get('id', ''),
            section_name=section_info.get('name', ''),
            parent_codes=parent_codes,
            is_category=is_category,
            is_subcategory=is_subcategory
        )
        
        icd_code.includes_notes = self._extract_notes(diag_elem, 'includes')
        icd_code.excludes1_notes = self._extract_notes(diag_elem, 'excludes1')
        icd_code.excludes2_notes = self._extract_notes(diag_elem, 'excludes2')
        icd_code.inclusion_terms = self._extract_inclusion_terms(diag_elem)
        icd_code.use_additional_code = self._extract_notes(diag_elem, 'useAdditionalCode')
        icd_code.code_first = self._extract_notes(diag_elem, 'codeFirst')
        icd_code.code_also = self._extract_notes(diag_elem, 'codeAlso')
        
        self.code_map[code] = icd_code

        for child_diag_elem in diag_elem.findall('diag'):
            self._parse_diagnosis_xml(child_diag_elem, chapter_info, section_info, parent_codes + [code])

    def _augment_with_order_file(self):
        """
        Parses the flat 'order' file to add any missing codes, inheriting context
        from parents found in the already-parsed XML data.
        """
        # This path logic is simplified assuming the script is run from the root
        order_file_path = 'raw-data/CM/icd10cm-order-2026.txt'
        if not Path(order_file_path).exists():
            print(f"Warning: Order file not found at {order_file_path}. The code list may be incomplete.")
            return
            
        print(f"Augmenting with data from {order_file_path}...")
        
        with open(order_file_path, 'r') as f:
            for line in f:
                match = re.match(r'^\d+\s+([A-Z0-9\.]+)\s+\d\s+(.+)$', line)
                if not match:
                    continue
                    
                code = match.group(1).strip()
                description = match.group(2).strip().split('   ')[0] # Clean up long descriptions

                # Standardize the code format to include a decimal point
                if len(code) > 3 and '.' not in code:
                    code = f"{code[:3]}.{code[3:]}"

                if code not in self.code_map:
                    parent_code_str = self._find_best_parent_code(code)
                    
                    if parent_code_str and parent_code_str in self.code_map:
                        parent_code_obj = self.code_map[parent_code_str]
                        
                        new_code = ICDCode(
                            code=code,
                            description=description,
                            chapter_number=parent_code_obj.chapter_number,
                            chapter_name=parent_code_obj.chapter_name,
                            section_id=parent_code_obj.section_id,
                            section_name=parent_code_obj.section_name,
                            parent_codes=[parent_code_str]
                        )
                        self.code_map[code] = new_code
                    else:
                        # Add it without hierarchy if no parent could be found
                        self.code_map[code] = ICDCode(code=code, description=description)

    def _find_best_parent_code(self, code_str: str) -> Optional[str]:
        """
        Finds the most likely parent for a given code by checking for progressively
        shorter versions of the code string in the existing code_map.
        """
        # Try finding parent by removing last character from sub-code
        if '.' in code_str:
            base, sub = code_str.split('.', 1)
            if len(sub) > 1:
                parent_candidate = f"{base}.{sub[:-1]}"
                if parent_candidate in self.code_map:
                    return parent_candidate
            # Check for the base code (e.g., A01.1 -> A01)
            if base in self.code_map:
                return base
        
        # Check for 3-character category (e.g., A011 -> A01)
        if len(code_str) > 3:
            parent_candidate = code_str[:3]
            if parent_candidate in self.code_map:
                return parent_candidate
                
        return None

    def _post_process_codes(self):
        """
        After all codes are parsed and augmented, iterate through them to build
        rich text and keywords.
        """
        print("Post-processing all codes to generate rich text and keywords...")
        for code_obj in self.code_map.values():
            parent_descriptions = [
                self.code_map[p_code].description for p_code in code_obj.parent_codes if p_code in self.code_map
            ]
            self._generate_rich_text(code_obj, parent_descriptions)
            self._generate_keywords(code_obj)
    
    def _extract_chapter_info(self, chapter_elem: ET.Element) -> Dict:
        """Extracts information from a <chapter> element."""
        name_elem = chapter_elem.find('name')
        desc_elem = chapter_elem.find('desc')
        chapter_name = desc_elem.text.strip() if desc_elem is not None else ""
        # Extract range from description, e.g., (A00-B99)
        range_match = re.search(r'\(([A-Z0-9\-]+)\)', chapter_name)

        return {
            'number': int(name_elem.text.strip()) if name_elem is not None and name_elem.text.strip().isdigit() else None,
            'name': chapter_name,
            'range': range_match.group(1) if range_match else ""
        }

    def _extract_section_info(self, section_elem: ET.Element) -> Dict:
        """Extracts information from a <section> element."""
        desc_elem = section_elem.find('desc')
        return {
            'id': section_elem.get('id', ''),
            'name': desc_elem.text.strip() if desc_elem is not None else ""
        }
    
    def _extract_notes(self, element: ET.Element, note_type: str) -> List[str]:
        """Extract notes of a specific type from an element"""
        notes = []
        
        # Find the note container
        note_container = element.find(note_type)
        if note_container is not None:
            # Extract all note elements
            for note_elem in note_container.findall('note'):
                if note_elem.text:
                    notes.append(note_elem.text.strip())
        
        return notes
    
    def _extract_inclusion_terms(self, diag_elem: ET.Element) -> List[str]:
        """Extracts <inclusionTerm> notes."""
        terms = []
        inclusion_term_elem = diag_elem.find('inclusionTerm')
        if inclusion_term_elem is not None:
            for note in inclusion_term_elem.findall('note'):
                if note.text:
                    terms.append(note.text.strip())
        return terms

    def _generate_keywords(self, icd_code: ICDCode):
        """Generates a list of keywords for the code."""
        keywords = set()
        
        # Add code and description
        keywords.add(icd_code.code)
        keywords.update(self._extract_keywords_from_text(icd_code.description))
        
        # Add notes
        for note in icd_code.includes_notes:
                keywords.update(self._extract_keywords_from_text(note))
        for term in icd_code.inclusion_terms:
            keywords.update(self._extract_keywords_from_text(term))
        
        icd_code.all_keywords = sorted(list(keywords))
    
    def _extract_keywords_from_text(self, text: str) -> Set[str]:
        """Extract meaningful keywords from text"""
        if not text:
            return set()
            
        # Clean text - remove parentheses content, codes, and special characters
        cleaned = re.sub(r'\([^)]*\)', '', text)  # Remove parentheses
        cleaned = re.sub(r'[A-Z]\d+[\.\-][A-Z0-9\.\-]*', '', cleaned)  # Remove codes
        cleaned = re.sub(r'[^\w\s-]', '', cleaned)  # Remove special chars
        
        # Split into words and filter
        words = cleaned.lower().split()
        
        # Filter out common stop words and short words
        stop_words = {
            'a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for', 'from', 
            'has', 'he', 'in', 'is', 'it', 'its', 'of', 'on', 'that', 'the', 
            'to', 'was', 'were', 'will', 'with', 'not', 'or', 'due', 'see',
            'nec', 'nos', 'unspecified', 'other', 'certain', 'diseases'
        }
        
        keywords = {
            word for word in words 
            if len(word) >= 3 and word not in stop_words and word.isalpha()
        }
        
        return keywords
    
    def _generate_rich_text(self, icd_code: ICDCode, parent_descriptions: List[str]):
        """
        Generates the rich text paragraph for vector search.
        """
        
        parts = []
        
        # Start with the code and its description
        parts.append(f"ICD-10-CM code {icd_code.code} is for '{icd_code.description}'.")
        
        # Add Chapter and Section context
        if icd_code.chapter_name:
            # Clean up chapter name for readability
            clean_chapter_name = icd_code.chapter_name.split('(')[0].strip()
            parts.append(f"It belongs to Chapter {icd_code.chapter_number}: {clean_chapter_name}.")
        if icd_code.section_name:
            parts.append(f"Specifically, it is in the section on '{icd_code.section_name}'.")

        # Add parent context
        if parent_descriptions:
            parent_context = " -> ".join(parent_descriptions)
            parts.append(f"The code hierarchy is: {parent_context}.")

        # Add inclusion notes
        if icd_code.includes_notes:
            notes = "; ".join(icd_code.includes_notes)
            parts.append(f"This category includes notes such as: {notes}.")

        if icd_code.inclusion_terms:
            terms = "; ".join(icd_code.inclusion_terms)
            parts.append(f"Related inclusion terms are: {terms}.")
        
        # Add coding guidelines
        if icd_code.code_first:
            guidelines = "; ".join(icd_code.code_first)
            parts.append(f"A 'Code First' guideline applies for: {guidelines}.")
        
        if icd_code.code_also:
            guidelines = "; ".join(icd_code.code_also)
            parts.append(f"A 'Code Also' guideline applies for: {guidelines}.")
            
        if icd_code.use_additional_code:
            guidelines = "; ".join(icd_code.use_additional_code)
            parts.append(f"A 'Use Additional Code' guideline applies for: {guidelines}.")

        icd_code.rich_text = " ".join(parts)
    
    def save_to_json(self, output_file: str):
        """Saves the parsed data to a JSON file."""
        print("Saving %d codes to %s..." % (len(self.codes), output_file))
        
        # Convert list of dataclass objects to list of dicts
        output_data = [asdict(code) for code in self.codes]
        
        try:
            with open(output_file, 'w') as f:
                json.dump(output_data, f, indent=2)
            print("Successfully saved data.")
        except IOError as e:
            print(f"Error writing to JSON file: {e}")


def main():
    parser = argparse.ArgumentParser(description="Parse ICD-10-CM XML file and generate a structured JSON output.")
    parser.add_argument(
        "--xml_file",
        default="raw-data/CM/icd10cm_tabular_2026.xml",
        help="Path to the main ICD-10-CM XML file (icd10cm_tabular_2026.xml)."
    )
    parser.add_argument(
        "--output_file",
        default="data-prep/icd10cm_parsed.json",
        help="Path to the output JSON file."
    )
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    Path(args.output_file).parent.mkdir(parents=True, exist_ok=True)

    icd10_parser = ICD10CMParser(xml_file_path=args.xml_file)
    parsed_codes = icd10_parser.parse()
    icd10_parser.save_to_json(output_file=args.output_file)


if __name__ == '__main__':
    exit(main()) 