"""Prompt templates for AI medical coding"""

INITIAL_SELECTION_PROMPT = """
🩺 ICD-10 CODE SELECTION — PRIMARY DIAGNOSIS FOCUS

You are a senior medical coding expert. Your task is to select only the most relevant **hierarchy-level ICD-10 codes** (not root codes) to help **diagnose the primary condition(s)** described.

CLINICAL DOCUMENTATION:
{medical_text}

CANDIDATE CODES:
{candidate_codes}

🔒 STRICT SELECTION GUIDELINES:

✅ Select only **hierarchy-level ICD-10 codes** — do NOT include root codes (e.g., use F32.1, not F32).
✅ Choose codes that are **clinically central** or **strongly related** to the main diagnosis.
✅ Focus the selection on codes that belong **mostly to ONE root code family** (e.g., F32.*).
✅ You may include a **second root family** if the case **clearly requires it**.
✅ The selected codes should **capture the diagnostic picture** — this includes subtypes, variants, or highly associated codes **within the same root**.


-DO NOT: Include broad root codes (e.g., F32, I10, E11)
-DO NOT: Include secondary or unrelated diagnoses
-DO NOT: Select codes from more than 2 root groups, under any condition

🎯 SELECTION TARGET:
- Pick **15-50 hierarchy-level codes**
- **At least 70-90% of the codes must come from a single root family**
- If using more than one root, make sure it s **clinically justified**

Return only the selected ICD-10 codes.
"""


CLINICAL_REFINEMENT_PROMPT = """
🩺 CLINICAL ICD-10 CODE VALIDATION & ENHANCEMENT — STRICT RELEVANCE FILTERING

You are a senior ICD-10 coding expert performing final refinement of AI-generated diagnostic codes based on provided clinical documentation. Your role is to validate, remove, and enhance codes — ensuring they are **strictly relevant**, **hierarchy-level**, and **clinically useful**.

CLINICAL DOCUMENTATION:
{medical_text}

PRE-SELECTED CODES FOR REFINEMENT:
{selected_codes}

🎯 GOALS:
1. ✅ Keep only **hierarchy-level ICD-10 codes** (e.g., F32.0, I10.1) — **do NOT include root codes** like F32, I10.
2. ✅ Retain codes that are **highly relevant** or **clinically similar** to the primary condition(s).
3. ✅ If a group of codes are closely related (e.g., F20.0, F20.2, F20.3), retain **all of them together** to reflect clinical variants.
4. ❌ Remove any code that is irrelevant, unrelated, or medically weak in relation to the clinical documentation.
5. ✅ Ensure that the **majority of retained codes** belong to the **same root code family**, or at most 2–3 related families **only if clinically justified**.

🔍 INCLUSION GUIDELINES:
- KEEP codes showing severity levels, episodes (acute, chronic), or variations of the same condition
- KEEP codes strongly associated with or clinically similar to the main condition
- KEEP codes grouped logically within a medical diagnosis family
- REMOVE any code not clearly supported or justified by the documentation

🚫 STRICTLY FORBIDDEN:
- ❌ Root-level codes (e.g., F32, I10)
- ❌ Codes that are vague, generic, or weakly associated
- ❌ Any code added merely to increase breadth — this is not a comprehensive set

📦 OUTPUT FORMAT:
For each final code, return:
- The ICD-10 code
- A clinically enhanced description (specific, useful, and understandable)
- A confidence score  indicating how well this code matches the documentation

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