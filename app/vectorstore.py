"""Pinecone vector database operations for ICD code retrieval"""

from pinecone import Pinecone
from openai import OpenAI
from .config import PINECONE_API_KEY, PINECONE_INDEX_NAME, OPENAI_API_KEY
from typing import List, Set, Dict


class VectorStore:
    """Handles vector operations with Pinecone for ICD code retrieval"""
    
    def __init__(self):
        self.pc = Pinecone(api_key=PINECONE_API_KEY)
        self.index = self.pc.Index(PINECONE_INDEX_NAME)
        self.openai_client = OpenAI(api_key=OPENAI_API_KEY)
    
    def _create_embedding(self, text: str) -> list:
        """Create embedding for text using OpenAI"""
        response = self.openai_client.embeddings.create(
            model="text-embedding-3-small",
            input=text
        )
        return response.data[0].embedding
    
    def search_all_codes(self, medical_text: str, top_k: int = 150) -> list:
        """
        Enhanced search for ICD codes across all chapters - optimized for two-step AI process
        
        Args:
            medical_text: The medical text to search for
            top_k: Number of results to return (optimized for 400-500 range)
            
        Returns:
            list: Comprehensive search results across all chapters
        """
        # Create embedding for the medical text
        query_embedding = self._create_embedding(medical_text)
        
        # Enhanced search with higher top_k for better coverage
        search_result = self.index.query(
            vector=query_embedding,
            top_k=min(top_k, 1000),  # Cap at 1000 for performance
            include_metadata=True
        )
        
        # Format results with enhanced metadata
        formatted_results = []
        for match in search_result.matches:
            formatted_results.append({
                'icd_code': match.id,
                'score': match.score,
                'description': match.metadata.get('description', ''),
                'rich_text': match.metadata.get('rich_text', ''),
                'chapter': match.metadata.get('chapter', ''),
                'section': match.metadata.get('section', '')
            })
        
        print(f"🔍 Vector search found {len(formatted_results)} codes for two-step AI processing")
        return formatted_results

    def get_codes_by_exact_match(self, icd_codes: List[str]) -> List[Dict]:
        """
        Efficiently retrieve codes by exact ICD code match using Pinecone fetch
        
        Args:
            icd_codes: List of exact ICD codes to retrieve
            
        Returns:
            List of code dictionaries with metadata
        """
        if not icd_codes:
            return []
        
        try:
            # Use Pinecone fetch for exact ID retrieval - much faster than query
            fetch_result = self.index.fetch(ids=icd_codes)
            
            formatted_results = []
            for code_id, vector_data in fetch_result.vectors.items():
                formatted_results.append({
                    'icd_code': code_id,
                    'score': 1.0,  # Exact match
                    'description': vector_data.metadata.get('description', ''),
                    'rich_text': vector_data.metadata.get('rich_text', ''),
                    'chapter': vector_data.metadata.get('chapter', ''),
                    'section': vector_data.metadata.get('section', '')
                })
            
            print(f"📋 Fetched {len(formatted_results)} codes by exact match")
            return formatted_results
            
        except Exception as e:
            print(f"Error fetching codes by exact match: {e}")
            return []

    def enrich_code_hierarchy(self, selected_codes: List[str], excluded_codes: Set[str], range_size: int = 3) -> Set[str]:
        """
        Generate enriched code set with ±range_size codes around selected codes
        Excludes codes that were already considered (selected or filtered out)
        
        Args:
            selected_codes: Codes selected by AI in first round
            excluded_codes: All codes from first round (selected + filtered out)
            range_size: Number of codes to expand up/down (default 3)
            
        Returns:
            Set of new enriched codes (excluding already processed ones)
        """
        enriched_codes = set()
        
        for code in selected_codes:
            if '.' in code:
                try:
                    base, suffix = code.split('.', 1)
                    
                    # Handle numeric suffixes (e.g., F32.1 → F32.0, F32.1, F32.2, etc.)
                    if suffix.isdigit():
                        num = int(suffix)
                        for i in range(max(0, num - range_size), num + range_size + 1):
                            candidate_code = f"{base}.{i}"
                            if candidate_code not in excluded_codes:
                                enriched_codes.add(candidate_code)
                    
                    # Handle alphanumeric suffixes (e.g., S72.001A)
                    elif len(suffix) >= 2 and suffix[:2].isdigit():
                        base_num = int(suffix[:2])
                        remainder = suffix[2:]
                        for i in range(max(0, base_num - range_size), base_num + range_size + 1):
                            candidate_code = f"{base}.{i:02d}{remainder}"
                            if candidate_code not in excluded_codes:
                                enriched_codes.add(candidate_code)
                    
                    # Handle other patterns - add base variations
                    else:
                        for i in range(range_size + 1):
                            candidate_code = f"{base}.{i}"
                            if candidate_code not in excluded_codes:
                                enriched_codes.add(candidate_code)
                                
                except ValueError:
                    # If parsing fails, just add the original code if not excluded
                    if code not in excluded_codes:
                        enriched_codes.add(code)
            else:
                # Handle codes without decimal (rare case)
                if code not in excluded_codes:
                    enriched_codes.add(code)
        
        print(f"🔍 Enriched {len(selected_codes)} codes to {len(enriched_codes)} new candidates (excluded {len(excluded_codes)} already processed)")
        return enriched_codes

    # ===== CHAPTER-SPECIFIC SEARCH METHODS =====

    def search_chapter_codes(self, search_text: str, chapter_list: list, top_k: int = 100) -> list:
        """
        Search for ICD codes in specific chapters using vector similarity
        
        Args:
            search_text: Text to search for
            chapter_list: List of chapter names to search within
            top_k: Maximum number of results to return
            
        Returns:
            List of matching codes with metadata
        """
        # Create embedding for the medical text
        query_embedding = self._create_embedding(search_text)
        
        results = []
        
        for chapter_name in chapter_list:
            # Use the EXACT chapter name for filtering
            filter_dict = {
                "chapter": {"$eq": chapter_name}
            }
            
            # Search in Pinecone with chapter filter
            search_result = self.index.query(
                vector=query_embedding,
                top_k=top_k,
                include_metadata=True,
                filter=filter_dict
            )
            
            # Format results
            formatted_results = []
            for match in search_result.matches:
                formatted_results.append({
                    'icd_code': match.id,
                    'score': match.score,
                    'description': match.metadata.get('description', ''),
                    'rich_text': match.metadata.get('rich_text', ''),
                    'chapter': match.metadata.get('chapter', ''),
                    'section': match.metadata.get('section', '')
                })
            
            results.extend(formatted_results)
        
        return results 