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
    
    # Remove prefixes
    lower_title = title.lower()
    for prefix in prefixes_to_remove:
        if lower_title.startswith(prefix.lower()):
            title = title[len(prefix):].strip()
            break
    
    # Remove common words that don't add meaning
    words_to_remove = [
        'a', 'an', 'the', 'your', 'my', 'our', 'their',
        'this', 'that', 'these', 'those', 'some', 'with'
    ]
    
    # Split into words and filter
    words = title.split()
    filtered_words = []
    for word in words:
        if word.lower() not in words_to_remove:
            filtered_words.append(word)
    
    # Rejoin the words
    subject = ' '.join(filtered_words)
    
    # Handle common medical abbreviations
    if get_full_form:
        for abbrev, full_form in medical_abbreviations.items():
            # Only replace if it's a standalone abbreviation
            pattern = r'\b' + re.escape(abbrev) + r'\b'
            if re.search(pattern, subject):
                subject = re.sub(pattern, full_form, subject)
    
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