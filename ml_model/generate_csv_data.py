import pandas as pd
import spacy
from spacy.matcher import PhraseMatcher
import re
import ast
import os
import PyPDF2
from skills_data import skills_domain # Import from skills_data.py
from tqdm import tqdm
from functools import lru_cache

nlp = spacy.load("en_core_web_sm")

keywords = [
    "experience", "project", "management", "responsibility", "responsible",
    "work history", "work experience", "job description", "summary",
    "role", "tasks", "positions", "certification", "abilities", "skills",
    "technical skills", "communication", "team", "agile", "collaborate",
    "led", "specialized","handled","implemented","delivered"
]

def clean_html(text):
    return re.sub(r"<[^>]+>|\s+", " ", str(text)).strip()

def extract_text_from_pdf(pdf_path):
    text = ""
    try:
        with open(pdf_path, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            for page_num in range(len(reader.pages)):
                text += reader.pages[page_num].extract_text()
    except Exception as e:
        print(f"Error reading PDF {pdf_path}: {e}")
    return text

def extract_relevant_sentences(text,min_word_count=5):
    if pd.isna(text) or not text:
        return []

    text = clean_html(text)
    doc = nlp(text) # Use the general nlp model for sentence segmentation

    keyword_lemmas = set(keywords)

    relevant_sents = []
    for sent in doc.sents:
        lemmas = {token.lemma_.lower() for token in sent if not token.is_stop and not token.is_punct}
        if keyword_lemmas.intersection(lemmas) and len(sent.text.split()) >= min_word_count:
            cleaned = sent.text.strip()
            if cleaned and not cleaned.isspace():
                relevant_sents.append(cleaned)

    return relevant_sents
    

@lru_cache(maxsize=1)
def build_skill_matcher(skill_list):
    temp_nlp = spacy.blank("en")
    matcher = PhraseMatcher(temp_nlp.vocab, attr="LOWER")
    
    patterns = []
    normalized_map = {}


    sorted_skill_list = sorted(skill_list, key=len, reverse=True)

    for skill in sorted_skill_list:
        skill_clean = skill.strip().lower()
        normalized_map[skill_clean] = skill
        patterns.append(temp_nlp.make_doc(skill_clean))
        if "." in skill:
            dotless = skill.replace(".", "").lower()
            if dotless not in normalized_map:
                normalized_map[dotless] = skill
                patterns.append(temp_nlp.make_doc(dotless))

        if any(c in skill for c in "/&#+."):
            spaced = re.sub(r"[\/&#+.]", " ", skill_clean)
            spaced = re.sub(r"\s+", " ", spaced).strip()
            if spaced not in normalized_map:
                normalized_map[spaced] = skill
                patterns.append(temp_nlp.make_doc(spaced))

    matcher.add("SKILL", patterns)
    return matcher, temp_nlp, normalized_map

def add_skills_to_sentences(sentences, skill_list=skills_domain):
    if not sentences:
        return []

    matcher, temp_nlp, normalized_map = build_skill_matcher(tuple(skill_list))  # make hashable for cache
    full_text = " ".join(sentences).lower()
    
    # Clean text
    processed = re.sub(r"[^\w\s.+#/&-]", " ", full_text)
    processed = re.sub(r"\s+", " ", processed).strip()
    doc = temp_nlp.make_doc(processed)

    matches = matcher(doc)
    matched_skills = set()

    for _, start, end in matches:
        span_text = doc[start:end].text.strip().lower()
        if span_text in normalized_map:
            matched_skills.add(normalized_map[span_text])

    # Fallback: regex for flexible/partial matches (only if nothing found)
    if not matched_skills:
        for skill in skill_list:
            pattern = r"\b" + re.escape(skill.lower()).replace(r"\ ", r"\s+") + r"\b"
            if re.search(pattern, processed):
                matched_skills.add(skill)

    return sorted(matched_skills)

def find_skills_with_regex(text, skills_list):
    found_skills = set()
    text_lower = text.lower()
    
    for skill in skills_list:
        skill_lower = skill.lower()
        
        # Escape special characters in the skill name for regex
        escaped_skill = re.escape(skill_lower)
        
        # Replace escaped spaces with \s+ to match one or more whitespace characters
        # This helps with skills like "Machine Learning" where there might be varying spaces
        pattern = escaped_skill.replace(r'\ ', r'\s+')
        
        # Add word boundaries to ensure whole word matches
        pattern = r'\b' + pattern + r'\b'
        
        try:
            if re.search(pattern, text_lower):
                found_skills.add(skill)
        except re.error:
            # Fallback for complex regex patterns that might cause errors,
            # or simply check for substring presence if regex fails.
            if skill_lower in text_lower:
                # Further refine by checking if it's a "word" match (e.g., not part of another word)
                words = re.findall(r'\b\w+\b', text_lower)
                if skill_lower in words or any(skill_lower in word for word in words):
                    found_skills.add(skill)
    return found_skills

def cleaned_text(raw):
    """
    Cleans raw data extracted from CSV/PDF, ensuring it's in (text, skills_list) format.
    Handles various input formats and potential errors from literal_eval.
    """
    try:
        parsed = ast.literal_eval(raw)
        if isinstance(parsed, tuple) and len(parsed) == 2:
            sentences = parsed[0]
            skills = parsed[1]
            
            if isinstance(sentences, list):
                # Join list of sentences into a single string
                cleaned_sentences = [re.sub(r'\s+', ' ', s.strip()) for s in sentences]
                full_text = " ".join(cleaned_sentences)
            else:
                # If 'sentences' is already a string, just clean it
                full_text = re.sub(r'\s+', ' ', str(sentences).strip())
            
            if isinstance(skills, list):
                # Clean up any escaped characters in skills
                cleaned_skills = [re.sub(r'\\', '', str(s)) for s in skills]
            else:
                cleaned_skills = [] # Ensure skills is a list
            
            return (full_text, cleaned_skills)
        else:
            # If parsed content is not a 2-element tuple, return empty
            return ("", [])
    except (ValueError, SyntaxError, TypeError) as e:
        # Catch errors during literal_eval
        print(f"Skipping bad row: {raw} — {e}")
        return ("", [])

def process_csv_data(csv_path, output_csv):
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
    print("=== Starting Resume Data Pipeline ===")
    process_csv_data(csv_path, intermediate_csv)
    process_pdf_data(pdf_root_dir, intermediate_csv)
    return clean_raw_data_file(intermediate_csv, final_csv)

# Optional: Only run when this script is the entry point
if __name__ == "__main__":
    final_output = generate_and_process_data()
    print(f"\nFinal processed data file: {final_output if final_output else 'None'}")

