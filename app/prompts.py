"""Prompt templates for AI medical coding"""


# ===== Clean Two-Step Process Prompts =====
INITIAL_SELECTION_PROMPT = """
🩺 PRIMARY CONDITION IDENTIFICATION - STRICT MEDICAL FOCUS

You are a senior medical coding specialist. Your CRITICAL task is to identify ONLY the PRIMARY medical conditions and select the MINIMUM relevant codes.

CLINICAL DOCUMENTATION:
{medical_text}

CANDIDATE CODES:
{candidate_codes}

🚨 STRICT REQUIREMENTS - FOLLOW EXACTLY:
🎯 Select 15-50 codes total
🎯 Focus on THE PRIMARY medical condition only
🎯 Choose 1-2 root code families maximum (e.g., ONLY F32/F33 for depression) or if you strongly believe the term includes mor than 2 include upto 4
🚫 include secondary, related, or differential diagnosis codes
🚫 DO NOT cast a wide net - be surgically precise


TARGET: 15-50 codes representing ONLY those PRIMARY conditions
AVOID: Comprehensive coverage, differential diagnosis, related conditions

CRITICAL: Count your selections. If over 25 codes, remove the least relevant ones.
"""

CLINICAL_REFINEMENT_PROMPT = """
🩺 CLINICAL CODE ENHANCEMENT & VALIDATION

You are a senior medical coding specialist performing final validation and enhancement of pre-selected relevant codes.

CLINICAL DOCUMENTATION:
{medical_text}

PRE-SELECTED CODES FOR ENHANCEMENT:
{selected_codes}

IMPORTANT CONTEXT:
- These codes have already been pre-filtered for relevance to the primary medical condition
- Related hierarchy codes have been intentionally added for completeness
- Your goal is to make sure irrelevant codes are not introduced during enrichemnt by removing them out 

REFINED TASK:
1. KEEP most codes that are clinically relevant to the documentation
2. Remove ONLY codes that are clearly unrelated or inappropriate
3. PRESERVE multiple similar codes within the same condition family 
4. Enhance descriptions to improve clinical clarity and usefulness

INCLUSION GUIDELINES:
✅ KEEP codes representing different severities of the same condition
✅ KEEP codes representing different episodes or variations  
✅ KEEP codes within the primary medical condition family
✅ KEEP codes that could be relevant for comprehensive documentation
🚫 Remove  codes that are clearly unrelated to the primary condition

TARGET: Aim to retain  codes from the pre-selected set
FOCUS: Enhancement of descriptions + confidence scoring

ENHANCEMENT REQUIREMENTS:
- Improve description clarity while maintaining medical accuracy
- Make descriptions more clinically specific and actionable
- Add context about when this code would be most appropriate
- Assign confidence scores based on clinical relevance to the documentation

DELIVERABLE: Enhanced codes with improved descriptions and confidence scores, preserving medical completeness.
"""





# ===== Spreadsheet functionality prompts =====
TITLE_ENRICHMENT_PROMPT = """
🎯 MEDICAL TITLE ENRICHMENT FOR VECTOR SEARCH

You are a medical terminology expert. Your task is to enrich a medical document title with additional relevant keywords to improve vector search accuracy.

CRITICAL REQUIREMENTS:
🚫 NEVER change the original meaning
🚫 NEVER add unrelated medical terms
✅ Only add synonyms, related terms, and medical variations
✅ Keep enrichment focused and relevant

Original title: {title}

TASK: Generate additional medical keywords that would help find relevant ICD-10-CM codes for this topic.

GUIDELINES:
- Add medical synonyms and alternative terms
- Include related anatomical terms if applicable
- Add common medical abbreviations if relevant
- Include related condition variations
- Focus on terms that would appear in ICD code descriptions

EXAMPLE:
Original: "Heart Attack"
Enriched: "myocardial infarction, MI, cardiac arrest, coronary thrombosis, acute coronary syndrome"

Provide your enriched keywords as a comma-separated list.
Keep the enrichment focused and medically accurate.
"""

METADATA_GENERATION_PROMPT = """
🩺 MEDICAL DOCUMENT METADATA GENERATION

You are a medical documentation expert. Analyze the medical title and document content to generate comprehensive metadata.

Title: {title}
Document Content: {content}

TASK: Generate the following metadata by analyzing BOTH the title and full document content:

1. GENDER APPLICABILITY:
   - "Male" - if condition primarily affects males
   - "Female" - if condition primarily affects females  
   - "Both" - if condition affects both genders equally

2. MEDICAL KEYWORDS:
   - Extract key medical terms from the content
   - Include relevant anatomical terms
   - Add symptoms, treatments, procedures
   - Include diagnostic terms and medical conditions
   - Include other relevant  medical keywords and synonyms
   - Add relevant medical specialties and care areas
   - Format as comma-separated list
   - Include synonyms and related terms
   - Focus on terms that would help in medical coding and search

GUIDELINES:
- Analyze the FULL document content, not just the title
- Be accurate about gender applicability based on content analysis
- Focus on clinically relevant keywords from the entire document
- Include both technical and common medical terms found in content and other relavant keywords too that might not be mentioned int he document
- Extract procedure names, medication categories, and treatment approaches
- Include anatomical terms and body systems mentioned
- You should provide at least 15-35 comprehensive keywords
- Prioritize terms that appear in or relate to the document content

Provide accurate and clinically appropriate metadata based on the complete document analysis.
"""