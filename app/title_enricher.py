"""Title enrichment and metadata generation for spreadsheet processing"""

from openai import OpenAI
from typing import Dict, Any
from .models import TitleEnrichment, DocumentMetadata
from .prompts import TITLE_ENRICHMENT_PROMPT, METADATA_GENERATION_PROMPT
from .config import OPENAI_API_KEY


class TitleEnricher:
    """Handle title enrichment and metadata generation using OpenAI"""
    
    def __init__(self):
        self.client = OpenAI(api_key=OPENAI_API_KEY)
        self.model = "gpt-4o-2024-08-06"
    
    def enrich_title(self, title: str) -> TitleEnrichment:
        """
        Enrich medical title with additional keywords for better vector search
        
        Args:
            title: Original medical document title
            
        Returns:
            TitleEnrichment: Enriched keywords and reasoning
        """
        try:
            prompt = TITLE_ENRICHMENT_PROMPT.format(title=title)
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a medical terminology expert specializing in keyword enrichment for vector search."
                    },
                    {"role": "user", "content": prompt}
                ],
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": "title_enrichment",
                        "strict": True,
                        "schema": TitleEnrichment.model_json_schema()
                    }
                },
                temperature=0.0,
                top_p=0.1,
                frequency_penalty=0.0,
                presence_penalty=0.0
            )
            
            import json
            from pydantic import ValidationError
            
            try:
                result_json = json.loads(response.choices[0].message.content)
                return TitleEnrichment(**result_json)
            except (json.JSONDecodeError, ValidationError) as e:
                raise ValueError(f"Failed to parse OpenAI response: {e}")
            
        except Exception as e:
            print(f"Error in title enrichment: {str(e)}")
            # Fallback: return original title
            return TitleEnrichment(
                enriched_keywords=title,
                reasoning="Fallback due to API error"
            )
    
    def generate_metadata(self, title: str, content: str = None) -> DocumentMetadata:
        """
        Generate document metadata (gender, keywords) from title and document content
        
        Args:
            title: Medical document title
            content: Full document content (optional, falls back to title only)
            
        Returns:
            DocumentMetadata: Generated gender and keywords
        """
        try:
            analysis_text = content if content else title
            
            prompt = METADATA_GENERATION_PROMPT.format(
                title=title, 
                content=analysis_text
            )
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a medical documentation expert specializing in metadata generation."
                    },
                    {"role": "user", "content": prompt}
                ],
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": "document_metadata",
                        "strict": True,
                        "schema": DocumentMetadata.model_json_schema()
                    }
                },
                temperature=0.0,
                top_p=0.1,
                frequency_penalty=0.0,
                presence_penalty=0.0
            )
            
            import json
            from pydantic import ValidationError
            
            try:
                result_json = json.loads(response.choices[0].message.content)
                return DocumentMetadata(**result_json)
            except (json.JSONDecodeError, ValidationError) as e:
                raise ValueError(f"Failed to parse OpenAI response: {e}")
            
        except Exception as e:
            print(f"Error in metadata generation: {str(e)}")
            # Fallback: return default values
            return DocumentMetadata(
                gender="Both",
                keywords=title,
                reasoning="Fallback due to API error"
            ) 