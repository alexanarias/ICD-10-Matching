import spacy
import re

# Load spaCy English model
nlp = spacy.load("en_core_web_sm")

# Articles to remove
articles = {"a", "an", "the"}

# Words to remove from the beginning of titles
title_prefixes = {"understanding", "treating", "controlling", "caring", "managing", "about"}

# Words that indicate context (to be removed)
context_words = {
    "during": "before",  # Keep text before "during"
    "in": "before",      # Keep text before "in"
    "with": "before",    # Keep text before "with"
    "and": "before",     # Keep text before "and"
    "between": "ignore"  # Special case: ignore this split
}

# Known medical abbreviations and their full forms
medical_abbreviations = {
    "CTA": "computed tomography angiography",
    "GAD": "generalized anxiety disorder",
    "ASD": "autism spectrum disorder",
    "RSV": "respiratory syncytial virus",
    "CO": "carbon monoxide",
    "DVT": "deep vein thrombosis",
    "COPD": "chronic obstructive pulmonary disease",
    "ED": "erectile dysfunction",
    "LDL": "low-density lipoprotein",
    "ICD": "implantable cardioverter defibrillator",
    "AMD": "age-related macular degeneration",
    "SIDS": "sudden infant death syndrome",
    "PAD": "peripheral artery disease",
    "PD": "peritoneal dialysis",
    "ESRD": "end-stage renal disease",
    "PPD": "postpartum depression",
    "CSF": "cerebrospinal fluid",
    "ADHD": "Attention Deficit Hyperactivity Disorder",
    "AIDS": "Acquired Immunodeficiency Syndrome",
    "ALS": "Amyotrophic Lateral Sclerosis",
    "ARDS": "Acute Respiratory Distress Syndrome",
    "BP": "Blood Pressure",
    "CHF": "Congestive Heart Failure",
    "CVD": "Cardiovascular Disease",
    "DM": "Diabetes Mellitus",
    "EKG": "Electrocardiogram",
    "GERD": "Gastroesophageal Reflux Disease",
    "GI": "Gastrointestinal",
    "HIV": "Human Immunodeficiency Virus",
    "HTN": "Hypertension",
    "IBD": "Inflammatory Bowel Disease",
    "IBS": "Irritable Bowel Syndrome",
    "MI": "Myocardial Infarction",
    "MS": "Multiple Sclerosis",
    "OCD": "Obsessive-Compulsive Disorder",
    "PE": "Pulmonary Embolism",
    "PTSD": "Post-Traumatic Stress Disorder",
    "RA": "Rheumatoid Arthritis",
    "TB": "Tuberculosis",
    "UTI": "Urinary Tract Infection"
}

def remove_leading_article(phrase):
    tokens = phrase.split()
    if tokens and tokens[0].lower() in articles:
        return ' '.join(tokens[1:])
    return phrase

def remove_title_prefix(text):
    words = text.lower().split()
    if words and words[0] in title_prefixes:
        return ' '.join(text.split()[1:])
    return text

def extract_subject_from_title(title: str, get_full_form: bool = False) -> str:
    """Extract the medical subject from a document title.
    
    Args:
        title: The document title
        get_full_form: Whether to return the full form of abbreviations
        
    Returns:
        str: The extracted subject
    """
    if not title:
        return ""
        
    # Clean the title
    title = title.strip()
    
    # Common title prefixes to remove
    prefixes_to_remove = [
        'Caring for',
        'Understanding',
        'About',
        'What is',
        'Guide to',
        'Information on',
        'Living with',
        'Managing',
        'Treatment of',
        'Treating',
        'Dealing with',
        'Coping with',
        'Overview of',
        'Introduction to',
        'Facts about',
        'All about'
    ]
    
    # Treatment-related terms to preserve
    treatment_terms = [
        'surgery',
        'surgical',
        'operation',
        'procedure',
        'treatment',
        'therapy',
        'management',
        'options',
        'approaches',
        'interventions'
    ]
    
    # Remove prefixes
    lower_title = title.lower()
    for prefix in prefixes_to_remove:
        if lower_title.startswith(prefix.lower()):
            title = title[len(prefix):].strip()
            break
    
    # Remove common words that don't add meaning, but preserve treatment-related terms
    words_to_remove = [
        'a', 'an', 'the', 'your', 'my', 'our', 'their',
        'this', 'that', 'these', 'those', 'some', 'with',
        'for', 'to', 'in', 'on', 'at', 'by', 'and'
    ]
    
    # Split into words and filter
    words = title.split()
    filtered_words = []
    has_treatment_term = any(term in lower_title for term in treatment_terms)
    
    for i, word in enumerate(words):
        word_lower = word.lower()
        # Keep the word if:
        # 1. It's a treatment term, or
        # 2. It's not in words_to_remove, or
        # 3. It's part of a medical term (check next/previous word)
        if (word_lower in treatment_terms or 
            word_lower not in words_to_remove or
            (i > 0 and words[i-1].lower() + " " + word_lower in lower_title) or
            (i < len(words)-1 and word_lower + " " + words[i+1].lower() in lower_title)):
            filtered_words.append(word)
    
    # Rejoin the words
    subject = ' '.join(filtered_words)
    
    # Special handling for surgical/treatment options
    if has_treatment_term:
        # Make sure we preserve the treatment type with the condition
        treatment_parts = []
        condition_parts = []
        
        for word in filtered_words:
            if word.lower() in treatment_terms:
                treatment_parts.append(word)
            else:
                condition_parts.append(word)
        
        if treatment_parts and condition_parts:
            # Combine them in a meaningful way
            subject = ' '.join(treatment_parts) + ' for ' + ' '.join(condition_parts)
    
    # Handle common medical abbreviations
    if get_full_form:
        for abbrev, full_form in medical_abbreviations.items():
            # Only replace if it's a standalone abbreviation
            pattern = r'\b' + re.escape(abbrev) + r'\b'
            if re.search(pattern, subject):
                subject = re.sub(pattern, full_form, subject)
    
    # Special handling for specific medical terms
    subject = subject.replace('basal joint', 'carpometacarpal joint')
    
    return subject.strip()

# Test cases
test_titles = [
    "Understanding Anemia During Cancer",
    "Treating a Mood Disorder",
    "Controlling Dust Mite Allergens in the Bedroom",
    "Understanding Childhood Asthma",
    "Understanding Generalized Anxiety Disorder in Children and Teens",
    "Understanding Basal Joint Arthritis",
    "Autism: Support for the Whole Family",
    "Caring for Ear Tubes",  # Should return "Ear Tubes"
    "Understanding Puberty: A Guide for Girls",
    "The Link Between PAD and Smoking",
    "Understanding Anemia During Cancer"
]

if __name__ == "__main__":
    # Example usage
    for t in test_titles:
        print(f"Title: {t}")
        if any(abbrev in t for abbrev in medical_abbreviations):
            print(f"Subject with full form: {extract_subject_from_title(t, get_full_form=True)}\n")
        else:
            print(f"Subject: {extract_subject_from_title(t)}\n")