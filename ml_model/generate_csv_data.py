import pandas as pd
import spacy
from spacy.matcher import PhraseMatcher
import re
import ast
import os
import PyPDF2
from skills_data import skills_domain
from functools import lru_cache

# Global constants
nlp = spacy.load("en_core_web_sm")

KEYWORDS = [
    "experience", "project", "management", "responsibility", "responsible",
    "work history", "work experience", "job description", "summary",
    "role", "tasks", "positions", "certification", "abilities", "skills",
    "technical skills", "communication", "team", "agile", "collaborate",
    "led", "specialized", "handled", "implemented", "delivered"
]

SPECIAL_CHARS = "/&#+."

def clean_html(text):
    """Clean HTML tags and normalize whitespace."""
    return re.sub(r"<[^>]+>|\s+", " ", str(text)).strip()

def extract_text_from_pdf(pdf_path):
    """Extract text content from PDF file."""
    try:
        with open(pdf_path, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            return " ".join(page.extract_text() for page in reader.pages)
    except Exception as e:
        print(f"Error reading PDF {pdf_path}: {e}")
        return ""

def extract_relevant_sentences(text, min_word_count=5):
    """Extract sentences containing relevant keywords."""
    if pd.isna(text) or not text:
        return []

    text = clean_html(text)
    doc = nlp(text)
    keyword_lemmas = set(KEYWORDS)

    relevant_sents = []
    for sent in doc.sents:
        lemmas = {token.lemma_.lower() for token in sent if not token.is_stop and not token.is_punct}
        if (keyword_lemmas.intersection(lemmas) and 
            len(sent.text.split()) >= min_word_count):
            cleaned = sent.text.strip()
            if cleaned and not cleaned.isspace():
                relevant_sents.append(cleaned)

    return relevant_sents

def create_skill_variations(skill):
    """Create variations of a skill for better matching."""
    skill_clean = skill.strip().lower()
    variations = [skill_clean]
    
    # Add dotless version
    if "." in skill:
        dotless = skill.replace(".", "").lower()
        variations.append(dotless)
    
    # Add spaced version for special characters
    if any(c in skill for c in SPECIAL_CHARS):
        spaced = re.sub(r"[\/&#+.]", " ", skill_clean)
        spaced = re.sub(r"\s+", " ", spaced).strip()
        variations.append(spaced)
    
    return variations

@lru_cache(maxsize=1)
def build_skill_matcher(skill_list):
    """Build a phrase matcher for skills with caching."""
    temp_nlp = spacy.blank("en")
    matcher = PhraseMatcher(temp_nlp.vocab, attr="LOWER")
    
    patterns = []
    normalized_map = {}
    
    # Sort by length (longest first) to avoid partial matches
    sorted_skill_list = sorted(skill_list, key=len, reverse=True)

    for skill in sorted_skill_list:
        variations = create_skill_variations(skill)
        
        for variation in variations:
            if variation not in normalized_map:
                normalized_map[variation] = skill
                patterns.append(temp_nlp.make_doc(variation))

    matcher.add("SKILL", patterns)
    return matcher, temp_nlp, normalized_map

def clean_text_for_matching(text):
    """Clean and normalize text for skill matching."""
    processed = re.sub(r"[^\w\s.+#/&-]", " ", text)
    return re.sub(r"\s+", " ", processed).strip()

def add_skills_to_sentences(sentences, skill_list=skills_domain):
    """Extract skills from sentences using phrase matching and regex fallback."""
    if not sentences:
        return []

    matcher, temp_nlp, normalized_map = build_skill_matcher(tuple(skill_list))
    full_text = " ".join(sentences).lower()
    processed = clean_text_for_matching(full_text)
    doc = temp_nlp.make_doc(processed)

    # Primary matching using phrase matcher
    matches = matcher(doc)
    matched_skills = set()

    for _, start, end in matches:
        span_text = doc[start:end].text.strip().lower()
        if span_text in normalized_map:
            matched_skills.add(normalized_map[span_text])

    # Fallback: regex for flexible/partial matches
    if not matched_skills:
        matched_skills = find_skills_with_regex(processed, skill_list)

    return sorted(matched_skills)

def find_skills_with_regex(text, skills_list):
    """Find skills using regex patterns as fallback."""
    found_skills = set()
    text_lower = text.lower()
    
    for skill in skills_list:
        skill_lower = skill.lower()
        
        # Create regex pattern with word boundaries
        escaped_skill = re.escape(skill_lower)
        pattern = r'\b' + escaped_skill.replace(r'\ ', r'\s+') + r'\b'
        
        try:
            if re.search(pattern, text_lower):
                found_skills.add(skill)
        except re.error:
            # Fallback for complex patterns
            if skill_lower in text_lower:
                words = re.findall(r'\b\w+\b', text_lower)
                if (skill_lower in words or 
                    any(skill_lower in word for word in words)):
                    found_skills.add(skill)
    
    return found_skills

def clean_sentences(sentences):
    """Clean and join sentences into a single text."""
    if isinstance(sentences, list):
        cleaned_sentences = [re.sub(r'\s+', ' ', s.strip()) for s in sentences]
        return " ".join(cleaned_sentences)
    else:
        return re.sub(r'\s+', ' ', str(sentences).strip())

def clean_skills(skills):
    """Clean and validate skills list."""
    if isinstance(skills, list):
        return [re.sub(r'\\', '', str(s)) for s in skills]
    return []

def cleaned_text(raw):
    """Clean raw data ensuring it's in (text, skills_list) format."""
    try:
        parsed = ast.literal_eval(raw)
        if not (isinstance(parsed, tuple) and len(parsed) == 2):
            return ("", [])
        
        sentences, skills = parsed
        full_text = clean_sentences(sentences)
        cleaned_skills = clean_skills(skills)
        
        return (full_text, cleaned_skills)
        
    except (ValueError, SyntaxError, TypeError) as e:
        print(f"Skipping bad row: {raw} — {e}")
        return ("", [])

def process_csv_data(csv_path, output_csv):
    """Process CSV data and extract relevant sentences and skills."""
    if not os.path.exists(csv_path):
        print(f"CSV file not found at {csv_path}")
        return

    try:
        df = pd.read_csv(csv_path)
        print(f"📄 Processing {len(df)} entries from CSV: {csv_path}")

        df["Relevant Sentences"] = df["Resume_html"].apply(extract_relevant_sentences)
        df["matched_skills"] = df["Relevant Sentences"].apply(add_skills_to_sentences)
        df["raw_data"] = list(zip(df["Relevant Sentences"], df["matched_skills"]))

        df[["raw_data"]].to_csv(output_csv, index=False, mode="w", header=True)
        print(f"CSV data saved to {output_csv}")
    except Exception as e:
        print(f"Error processing CSV data: {e}")

def process_pdf_data(pdf_root_dir, output_csv):
    """Process PDF files and extract relevant sentences and skills."""
    if not os.path.exists(pdf_root_dir):
        print(f"PDF folder not found: {pdf_root_dir}")
        return

    first_write = not os.path.exists(output_csv) or pd.read_csv(output_csv).empty
    print(f"Scanning PDFs from {pdf_root_dir}...")

    for root, _, files in os.walk(pdf_root_dir):
        for file in sorted(f for f in files if f.endswith(".pdf")):
            try:
                path = os.path.join(root, file)
                print(f"📘 Reading {file}...")
                
                text = extract_text_from_pdf(path)
                if text:
                    sentences = extract_relevant_sentences(text)
                    skills = add_skills_to_sentences(sentences)
                    row = pd.DataFrame([{"raw_data": (sentences, skills)}])
                    row.to_csv(output_csv, mode="a", header=first_write, index=False)
                    first_write = False
            except Exception as e:
                print(f"Failed to process {file}: {e}")

def clean_raw_data_file(input_csv="relevant.csv", output_csv="raw_data.csv"):
    """Clean and validate raw data file."""
    if not os.path.exists(input_csv):
        print(f"Input file {input_csv} not found.")
        return None

    try:
        df = pd.read_csv(input_csv)
        print(f"Cleaning {len(df)} raw data rows...")

        df["raw_data"] = df["raw_data"].apply(cleaned_text)
        df.to_csv(output_csv, index=False)
        print(f"Cleaned data saved to {output_csv}")
        return output_csv
    except Exception as e:
        print(f"Error during cleaning: {e}")
        return None

def generate_and_process_data(
    csv_path="./data/Resume/Resume/Resume.csv",
    pdf_root_dir="./data/Resume/data/data/",
    intermediate_csv="relevant.csv",
    final_csv="raw_data.csv",
):
    """Main pipeline function to process CSV and PDF data."""
    print("=== Starting Resume Data Pipeline ===")
    
    process_csv_data(csv_path, intermediate_csv)
    process_pdf_data(pdf_root_dir, intermediate_csv)
    
    return clean_raw_data_file(intermediate_csv, final_csv)

if __name__ == "__main__":
    final_output = generate_and_process_data()
    print(f"\nFinal processed data file: {final_output if final_output else 'None'}")

