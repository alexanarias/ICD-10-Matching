import os
from dotenv import load_dotenv

load_dotenv()

# API Keys
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")

# EXACT ICD-10-CM Chapters - DO NOT MODIFY THESE NAMES!
ICD_CHAPTERS = [
    "Certain conditions originating in the perinatal period (P00-P96)",
    "Certain infectious and parasitic diseases (A00-B99)",
    "Codes for special purposes (U00-U85)",
    "Congenital malformations, deformations, chromosomal abnormalities, and genetic disorders (Q00-QA0)",
    "Diseases of the blood and blood-forming organs and certain disorders involving the immune mechanism (D50-D89)",
    "Diseases of the circulatory system (I00-I99)",
    "Diseases of the digestive system (K00-K95)",
    "Diseases of the ear and mastoid process (H60-H95)",
    "Diseases of the eye and adnexa (H00-H59)",
    "Diseases of the genitourinary system (N00-N99)",
    "Diseases of the musculoskeletal system and connective tissue (M00-M99)",
    "Diseases of the nervous system (G00-G99)",
    "Diseases of the respiratory system (J00-J99)",
    "Diseases of the skin and subcutaneous tissue (L00-L99)",
    "Endocrine, nutritional and metabolic diseases (E00-E89)",
    "External causes of morbidity (V00-Y99)",
    "Factors influencing health status and contact with health services (Z00-Z99)",
    "Injury, poisoning and certain other consequences of external causes (S00-T88)",
    "Mental, Behavioral and Neurodevelopmental disorders (F01-F99)",
    "Neoplasms (C00-D49)",
    "Pregnancy, childbirth and the puerperium (O00-O9A)",
    "Symptoms, signs and abnormal clinical and laboratory findings, not elsewhere classified (R00-R99)"
    "cls causes of morbidity (V00-Y99)"
] 