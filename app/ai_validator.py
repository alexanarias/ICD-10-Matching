"""AI-powered validation of ICD-10-CM codes with enhanced multi-stage process"""

from openai import OpenAI, AsyncOpenAI
from .config import OPENAI_API_KEY
from .models import (
    InitialSelectionResponse, ClinicalRefinementResponse, RefinedCodeValidation
)
from .prompts import INITIAL_SELECTION_PROMPT, CLINICAL_REFINEMENT_PROMPT
import json

class AIValidator:
    """Enhanced AI validator with multi-stage process for medical accuracy"""
    
    def __init__(self):
        self.client = OpenAI(api_key=OPENAI_API_KEY)
        self.async_client = AsyncOpenAI(api_key=OPENAI_API_KEY)
        self.model = "gpt-4o-2024-08-06"
    
    def _format_simple_codes(self, candidate_codes: list) -> str:
        """Helper to format codes as simple list for initial selection"""
        return "\n".join([f"- {code['icd_code']}: {code['description']}" for code in candidate_codes])

    def _format_selected_codes(self, selected_codes: list, all_candidates: list) -> str:
        """Helper to format selected codes with their descriptions for refinement"""
        # Create lookup for descriptions
        code_lookup = {code['icd_code']: code['description'] for code in all_candidates}
        
        formatted = []
        for code in selected_codes:
            description = code_lookup.get(code, "Description not found")
            formatted.append(f"- {code}: {description}")
        
        return "\n".join(formatted)

    async def initial_selection(self, medical_text: str, candidate_codes: list) -> InitialSelectionResponse:
        """
        Step 1: Focused initial code selection - select 8-15 codes representing PRIMARY condition
        """
        if not candidate_codes:
            return InitialSelectionResponse(selected_codes=[])

        formatted_codes = self._format_simple_codes(candidate_codes)
        prompt = INITIAL_SELECTION_PROMPT.format(
            medical_text=medical_text,
            candidate_codes=formatted_codes
        )
        
        try:
            completion = await self.async_client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a medical coding expert performing focused primary condition identification."},
                    {"role": "user", "content": prompt}
                ],
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": "initial_selection",
                        "strict": True,
                        "schema": InitialSelectionResponse.model_json_schema()
                    }
                },
                temperature=0.05,  # Ultra-low temperature for strict adherence to instructions
                top_p=0.01
            )
            
            result_json = json.loads(completion.choices[0].message.content)
            return InitialSelectionResponse(**result_json)
            
        except Exception as e:
            print(f"Error during focused initial selection: {e}")
            return InitialSelectionResponse(selected_codes=[])

    async def enhanced_multi_stage_validation(self, medical_text: str, initial_candidates: list, vectorstore) -> ClinicalRefinementResponse:
        """
        ENHANCED MULTI-STAGE PROCESS:
        1. Focused initial selection (8-15 primary codes)
        2. Hierarchy enrichment (±3 codes around selected)
        3. Bulk retrieval of enriched codes
        4. Final clinical refinement with complete hierarchy
        """
        print("🎯 Starting enhanced multi-stage validation...")
        
        # Stage 1: Focused Initial Selection
        print("📌 Stage 1: Focused primary condition identification...")
        initial_result = await self.initial_selection(medical_text, initial_candidates)
        
        if not initial_result.selected_codes:
            print("❌ No codes selected in initial stage")
            return ClinicalRefinementResponse(
                refined_codes=[],
                clinical_summary="No relevant codes identified in primary condition analysis."
            )
        
        print(f"✅ Selected {len(initial_result.selected_codes)} primary codes: {initial_result.selected_codes}")
        
        # Stage 2: Create exclusion set (codes that were considered but NOT selected)
        initial_candidate_codes = {code['icd_code'] for code in initial_candidates}
        selected_code_set = set(initial_result.selected_codes)
        
        # Exclude codes that were in initial candidates but NOT selected (rejected codes)
        rejected_codes = initial_candidate_codes - selected_code_set
        
        print(f"🚫 Excluding {len(rejected_codes)} rejected codes from initial round")
        print(f"✅ Keeping {len(selected_code_set)} selected codes for enrichment")
        
        # Stage 3: Hierarchy Enrichment around selected codes (excluding rejected ones)
        print("🔍 Stage 2: Hierarchy enrichment (±3 code range)...")
        enriched_codes = vectorstore.enrich_code_hierarchy(
            selected_codes=initial_result.selected_codes,
            excluded_codes=rejected_codes,  # Only exclude rejected codes, not selected ones
            range_size=3
        )
        
        if not enriched_codes:
            print("⚠️ No new codes found in hierarchy enrichment")
            # Proceed with just initial codes
            enriched_candidates = []
        else:
            print(f"🔍 Generated {len(enriched_codes)} enriched codes")
            
            # Stage 4: Bulk Retrieval of Enriched Codes
            print("📋 Stage 3: Bulk retrieval of enriched codes...")
            enriched_candidates = vectorstore.get_codes_by_exact_match(list(enriched_codes))
            print(f"📋 Retrieved {len(enriched_candidates)} valid enriched codes")
        
        # Stage 5: Combine ONLY selected and enriched candidates (not rejected ones)
        # Create candidates list from selected codes
        selected_candidates = [code for code in initial_candidates if code['icd_code'] in selected_code_set]
        
        combined_candidates = selected_candidates + enriched_candidates
        print(f"🔗 Combined {len(selected_candidates)} selected + {len(enriched_candidates)} enriched = {len(combined_candidates)} total codes")
        print(f"✅ Properly excluded {len(rejected_codes)} rejected codes from final refinement")
        
        # Stage 6: Final Clinical Refinement with Complete Set
        print("🩺 Stage 4: Final clinical refinement with complete hierarchy...")
        final_result = await self.clinical_refinement(
            medical_text=medical_text,
            selected_codes=initial_result.selected_codes,
            all_candidates=combined_candidates
        )
        
        print(f"✅ Final validation complete: {len(final_result.refined_codes)} refined codes")
        return final_result

    async def clinical_refinement(self, medical_text: str, selected_codes: list, all_candidates: list) -> ClinicalRefinementResponse:
        """
        Step 2: Clinical refinement - remove irrelevant codes and enhance descriptions with confidence scores
        """
        if not selected_codes:
            return ClinicalRefinementResponse(
                refined_codes=[],
                clinical_summary="No codes provided for refinement."
            )

        formatted_codes = self._format_selected_codes(selected_codes, all_candidates)
        prompt = CLINICAL_REFINEMENT_PROMPT.format(
            medical_text=medical_text,
            selected_codes=formatted_codes
        )
        
        try:
            completion = await self.async_client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a senior medical coding specialist performing clinical validation and enhancement."},
                    {"role": "user", "content": prompt}
                ],
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": "clinical_refinement",
                        "strict": True,
                        "schema": ClinicalRefinementResponse.model_json_schema()
                    }
                },
                temperature=0.2,
                top_p=0.1
            )
            
            result_json = json.loads(completion.choices[0].message.content)
            return ClinicalRefinementResponse(**result_json)
            
        except Exception as e:
            print(f"Error during clinical refinement: {e}")
            return ClinicalRefinementResponse(
                refined_codes=[],
                clinical_summary=f"Clinical refinement failed: {e}"
            ) 