"""Prompt templates for AI classification and validation"""

CHAPTER_CLASSIFICATION_PROMPT = """
🚨🚨🚨 LIFE-CRITICAL MEDICAL CODING TASK 🚨🚨🚨
⚠️ CRITICAL INSTRUCTION: You MUST return EXACT chapter names from the list below with NO modifications, additions, or variations whatsoever.
🚫 NEVER GUESS - This affects patient care and billing accuracy
🚫 NEVER MODIFY chapter names in ANY way

TASK: Analyze medical text and identify the most relevant ICD-10-CM chapters.

🚫 FORBIDDEN:
- Do NOT modify chapter names in any way
- Do NOT add "Chapter 1:", "Chapter 2:" etc.
- Do NOT change punctuation, spacing, or wording
- Do NOT abbreviate or expand names
- Do NOT add explanatory text to chapter names

✅ REQUIRED:
- Copy chapter names EXACTLY as shown in the list
- Use only names from the provided list
- Return probability between 0.0 and 1.0
- Provide brief reasoning for each match

📋 EXACT CHAPTER NAMES (copy these EXACTLY):
{chapters_list}

🎯 ANALYSIS CRITERIA:
- Primary diagnoses and conditions
- Symptoms and clinical findings  
- Anatomical systems involved
- Disease processes described

📊 OUTPUT REQUIREMENTS:
- Return top 5 most relevant chapters
- Order by probability (highest first)
- Only include chapters with probability > 0.3
- Use EXACT chapter names from the list above

Medical text to analyze:
{medical_text}

🚨 FINAL WARNING: Copy chapter names EXACTLY as they appear in the list. Any modification will cause system failure.
🚫 NO GUESSING ALLOWED - Only use chapters where you have HIGH CONFIDENCE
🩺 Patient safety depends on your accuracy - BE PRECISE!
"""

VALIDATION_PROMPT = """
🚨🚨🚨 CRITICAL CODE VALIDATION - PATIENT SAFETY IMPACT 🚨🚨🚨

You are an expert medical coder specializing in ICD-10-CM code validation. Your task is to evaluate how well specific ICD codes match the given medical documentation.

🚫 NEVER GUESS OR ASSUME ANYTHING
🚫 BE EXTREMELY CONSERVATIVE WITH CONFIDENCE SCORES  
🚫 Only give >80% confidence if evidence is CRYSTAL CLEAR
🚫 Only give >90% confidence if evidence is OVERWHELMING
🩺 Remember: Incorrect codes affect patient care and insurance billing

Original medical text:
{medical_text}

Please evaluate each of the following ICD codes for accuracy and appropriateness:

{candidate_codes}

🔍 VALIDATION CRITERIA - BE EXTREMELY STRICT:

For each code, provide:
1. The ICD-10-CM code
2. The official description  
3. A confidence score (0.0 to 1.0) - BE VERY CONSERVATIVE:
   - 0.9-1.0: Perfect match with overwhelming evidence
   - 0.8-0.9: Strong match with clear evidence  
   - 0.7-0.8: Good match with solid evidence
   - 0.5-0.7: Moderate match with some evidence
   - 0.3-0.5: Weak match with minimal evidence
   - 0.0-0.3: Poor match or insufficient evidence
4. Clear reasoning explaining your assessment
5. Specific evidence from the medical text
6. Any missing information needed for higher confidence

Consider:
- Specificity and accuracy of the code
- Completeness of documentation  
- Clinical appropriateness
- Coding guidelines compliance
- Available supporting evidence

🚫 CRITICAL: Only recommend codes with confidence score > 0.5
📊 Order by confidence score (highest first)
🚨 When in doubt, use LOWER confidence scores - patient safety first!
""" 