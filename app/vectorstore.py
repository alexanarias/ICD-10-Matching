"""Pinecone vector database operations for ICD code retrieval"""

from pinecone import Pinecone
from openai import OpenAI
from .config import PINECONE_API_KEY, PINECONE_INDEX_NAME, OPENAI_API_KEY


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
    

    
    def search_codes_by_chapter(self, medical_text: str, chapter_names: list, 
                               top_k: int = 50) -> dict:
        """
        Search for ICD codes in specific chapters using vector similarity
        
        Args:
            medical_text: The medical text to search for
            chapter_names: List of EXACT chapter names to filter by
            top_k: Number of results to return per chapter
            
        Returns:
            dict: Chapter-wise search results
        """
        # Create embedding for the medical text
        query_embedding = self._create_embedding(medical_text)
        
        results = {}
        
        for chapter_name in chapter_names:
            # Use the EXACT chapter name for filtering
            # This matches exactly what we stored in Pinecone metadata
            filter_dict = {
                "chapter": {"$eq": chapter_name}
            }
            
            print(f"🔍 Searching chapter: '{chapter_name}'")  # Debug logging
            
            # Search in Pinecone with chapter filter
            search_result = self.index.query(
                vector=query_embedding,
                top_k=top_k,
                include_metadata=True,
                filter=filter_dict
            )
            
            print(f"   Found {len(search_result.matches)} matches")  # Debug logging
            
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
            
            results[chapter_name] = formatted_results
        
        return results
    
    def search_all_codes(self, medical_text: str, top_k: int = 100) -> list:
        """
        Search for ICD codes across all chapters
        
        Args:
            medical_text: The medical text to search for
            top_k: Number of results to return
            
        Returns:
            list: Search results across all chapters
        """
        query_embedding = self._create_embedding(medical_text)
        
        search_result = self.index.query(
            vector=query_embedding,
            top_k=top_k,
            include_metadata=True
        )
        
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
        
        return formatted_results 