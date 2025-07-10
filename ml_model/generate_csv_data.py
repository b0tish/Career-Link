import pandas as pd
import spacy
from spacy.matcher import PhraseMatcher
import re
import ast
import os
import PyPDF2
from skills_data import skills_domain # Import from skills_data.py

# Initialize spacy model for sentence extraction and preliminary processing
# This is a general-purpose model, not the one being trained for NER
nlp = spacy.load("en_core_web_sm")

# Define keywords for sentence extraction
keywords = [
    "experience", "project", "management", "responsibility",
    "work history", "work experience", "job description",
    "role", "tasks", "positions", "certification", "abilities",
    "technical skills", "summary", "profile", "accomplishments"
]

def clean_html(text):
    """Function to clean HTML content."""
    return re.sub(r"<[^>]+>|\s+", " ", str(text)).strip()

def extract_text_from_pdf(pdf_path):
    """Function to extract text from PDF."""
    text = ""
    try:
        with open(pdf_path, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            for page_num in range(len(reader.pages)):
                text += reader.pages[page_num].extract_text()
    except Exception as e:
        print(f"Error reading PDF {pdf_path}: {e}")
    return text

def extract_relevant_sentences(text):
    """Function to extract relevant sentences based on keywords."""
    if pd.isna(text) or not text:
        return []
    text = clean_html(text)
    doc = nlp(text) # Use the general nlp model for sentence segmentation
    keyword_set = set(keywords)
    return [
        sent.text.strip()
        for sent in doc.sents
        if any(keyword in sent.text.lower() for keyword in keyword_set)
    ]

def add_skills_to_sentences(sentences, skill_list=skills_domain):
    """
    Identifies and extracts skills from a list of sentences using PhraseMatcher
    and additional regex patterns.
    """
    if not sentences:
        return []
    
    # Use a blank spaCy model for PhraseMatcher to avoid interference from existing pipes
    temp_nlp = spacy.blank("en")
    matcher = PhraseMatcher(temp_nlp.vocab, attr="LOWER")
    
    skill_patterns = []
    skill_map = {}
    
    for skill in skill_list:
        normalized_skill = skill.lower().strip()
        
        pattern_doc = temp_nlp.make_doc(normalized_skill)
        skill_patterns.append(pattern_doc)
        skill_map[normalized_skill] = skill
        
        # Add variants with removed dots (e.g., "C.S" -> "CS")
        if "." in skill:
            variant = skill.replace(".", "").lower().strip()
            if variant != normalized_skill:
                variant_doc = temp_nlp.make_doc(variant)
                skill_patterns.append(variant_doc)
                skill_map[variant] = skill
        
        # Add variants with spaces for special characters (e.g., "C/C++" -> "C C++")
        if any(char in skill for char in ["/", "&", "+", "#"]):
            clean_skill = re.sub(r'[/&+#]', ' ', skill).lower().strip()
            clean_skill = re.sub(r'\s+', ' ', clean_skill) # Normalize multiple spaces
            if clean_skill != normalized_skill and clean_skill:
                clean_doc = temp_nlp.make_doc(clean_skill)
                skill_patterns.append(clean_doc)
                skill_map[clean_skill] = skill
    
    matcher.add("SKILL", skill_patterns)
    
    full_text = " ".join(sentences)
    
    # Pre-process text for matching: lowercasing and standardizing spaces
    processed_text = full_text.lower()
    processed_text = re.sub(r'[^\w\s.#+&/-]', ' ', processed_text) # Keep relevant special chars for skills
    processed_text = re.sub(r'\s+', ' ', processed_text).strip()
    
    doc = temp_nlp.make_doc(processed_text)
    
    matches = matcher(doc)
    matched_skills = set()
    
    for match_id, start, end in matches:
        span = doc[start:end]
        matched_text = span.text.lower().strip()
        
        if matched_text in skill_map:
            matched_skills.add(skill_map[matched_text])
        else:
            # Fallback for exact match from skills_domain if not in map (should be rare)
            for skill in skill_list:
                if skill.lower() == matched_text:
                    matched_skills.add(skill)
                    break
    
    # Add skills found with additional regex patterns (e.g., handling partial matches or complex patterns)
    additional_matches = find_skills_with_regex(full_text, skill_list)
    matched_skills.update(additional_matches)
    
    return list(matched_skills)

def find_skills_with_regex(text, skills_list):
    """
    Finds skills in text using regex for more flexible matching,
    especially for multi-word skills with variations or specific patterns.
    """
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

def generate_and_process_data(csv_path="./data/Resume/Resume/Resume.csv", pdf_root_dir="data/Resume/data/data/", output_csv="relevant.csv"):
    """
    Main function to generate and process data from CSV and PDF files,
    saving the results to a specified CSV file.
    """
    print("=== Data Generation and Processing ===")

    # Process CSV files
    if os.path.exists(csv_path):
        try:
            df_to_process = pd.read_csv(csv_path).iloc[501:801] # Example slice
            print(f"Processing {len(df_to_process)} CSV data entries from {csv_path}.")
            df_to_process["Relevant Sentences"] = df_to_process["Resume_html"].apply(extract_relevant_sentences)
            df_to_process["matched_skills"] = df_to_process["Relevant Sentences"].apply(add_skills_to_sentences)
            df_to_process["raw_data"] = list(zip(df_to_process["Relevant Sentences"], df_to_process["matched_skills"]))
            
            df_to_process[["raw_data"]].to_csv(output_csv, mode="w", index=False, header=True)
            print(f"CSV data processed and saved to {output_csv}")
        except Exception as e:
            print(f"Error processing CSV file {csv_path}: {e}")
    else:
        print(f"CSV file not found at {csv_path}, skipping CSV processing...")

    # Process PDF files and append to the output_csv
    first_pdf_write = not os.path.exists(output_csv) or pd.read_csv(output_csv).empty

    if os.path.exists(pdf_root_dir):
        print(f"Processing PDF files from {pdf_root_dir}...")
        
        for root, dirs, files in os.walk(pdf_root_dir):
            pdf_files_in_folder = sorted([f for f in files if f.endswith(".pdf")])
            
            for file in pdf_files_in_folder:
                pdf_path = os.path.join(root, file)
                print(f"Processing PDF: {pdf_path}")
                pdf_text = extract_text_from_pdf(pdf_path)
                
                if pdf_text:
                    relevant_sentences = extract_relevant_sentences(pdf_text)
                    if relevant_sentences:
                        matched_skills = add_skills_to_sentences(relevant_sentences)
                        
                        pdf_df = pd.DataFrame([{
                            "raw_data": (relevant_sentences, matched_skills)
                        }])
                        
                        pdf_df[["raw_data"]].to_csv(
                            output_csv,
                            mode="a",
                            index=False,
                            header=first_pdf_write
                        )
                        first_pdf_write = False # Only write header for the very first entry
        print("PDF processing completed.")
    else:
        print(f"PDF directory not found at {pdf_root_dir}, skipping PDF processing...")

    # Clean the raw data
    if os.path.exists(output_csv):
        df = pd.read_csv(output_csv)
        print(f"Cleaning {len(df)} rows of data from {output_csv}...")
        df["raw_data"] = df["raw_data"].apply(cleaned_text)
        cleaned_output_csv = "raw_data.csv" # Name for the final cleaned data
        df.to_csv(cleaned_output_csv, index=False)
        print(f"Raw data cleaned and saved to {cleaned_output_csv}")
        return cleaned_output_csv
    else:
        print("No raw data file generated, skipping cleaning...")
        return None

if __name__ == "__main__":
    # Example usage when run directly
    generate_and_process_data()
