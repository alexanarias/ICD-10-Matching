"""AI validation of ICD codes using OpenAI structured outputs"""

from openai import OpenAI
from .config import OPENAI_API_KEY
from .models import ValidationResponse
from .prompts import VALIDATION_PROMPT


class AIValidator:
    """Validates ICD codes using AI with structured outputs"""
    
    def __init__(self):
        self.client = OpenAI(api_key=OPENAI_API_KEY)
        self.model = "gpt-4o-2024-08-06"
    
    def validate_codes(self, medical_text: str, candidate_codes: list) -> ValidationResponse:
        """
        Validate candidate ICD codes against medical text
        
        Args:
            medical_text: Original medical documentation
            candidate_codes: List of candidate codes from vector search
            
        Returns:
            ValidationResponse: Structured validation results
        """
        # Format candidate codes for prompt
        codes_formatted = self._format_candidate_codes(candidate_codes)
        
        # Create validation prompt
        prompt = VALIDATION_PROMPT.format(
            medical_text=medical_text,
            candidate_codes=codes_formatted
        )
        
        # Call OpenAI with SUPER STRICT parameters for validation accuracy
        completion = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "🚨 CRITICAL MEDICAL CODE VALIDATION 🚨 You are an expert medical coder and auditor. This validation affects patient care. Never guess - only validate with 100% certainty. Be extremely strict and conservative with confidence scores."},
                {"role": "user", "content": prompt}
            ],
            # SUPER STRICT PARAMETERS FOR MAXIMUM VALIDATION ACCURACY
            temperature=0.0,           # Zero creativity - pure logic only
            top_p=0.1,                # Extremely focused token selection
            frequency_penalty=0.0,    # No penalty for repetition
            presence_penalty=0.0,     # No penalty for detailed reasoning
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "validation_response",
                    "strict": True,
                    "schema": ValidationResponse.model_json_schema()
                }
            }
        )
        
        import json
        from pydantic import ValidationError
        
        try:
            result_json = json.loads(completion.choices[0].message.content)
            return ValidationResponse(**result_json)
        except (json.JSONDecodeError, ValidationError) as e:
            raise ValueError(f"Failed to parse OpenAI response: {e}")
    
    def _format_candidate_codes(self, candidate_codes: list) -> str:
        """Format candidate codes for the validation prompt"""
        formatted = []
        
        for code_data in candidate_codes:
            code_text = f"""
Code: {code_data['icd_code']}
Description: {code_data['description']}
Chapter: {code_data.get('chapter', 'Unknown')}
Vector Similarity Score: {code_data['score']:.3f}
Rich Context: {code_data.get('rich_text', '')[:200]}...
---"""
            formatted.append(code_text)
        
        return "\n".join(formatted)
    
    def get_high_confidence_codes(self, validation: ValidationResponse, 
                                 threshold: float = 0.5) -> list:
        """
        Filter codes with confidence above threshold
        
        Args:
            validation: The validation result
            threshold: Minimum confidence threshold
            
        Returns:
            list: High confidence code validations
        """
        return [
            code_val for code_val in validation.validated_codes 
            if code_val.confidence_score >= threshold
        ] 