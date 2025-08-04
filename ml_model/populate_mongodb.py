import pandas as pd
from pymongo import MongoClient
import spacy
from dotenv import load_dotenv
import os
import json
from skills_data import skills_domain

load_dotenv()
MONGODB_URI = os.getenv("MONGODB_URI")

# Load trained spaCy model
nlp = spacy.load("./skill_ner_model")

# Normalize skills_domain
skills_domain_lower = [skill.lower() for skill in skills_domain]

def extract_skills(text: str):
    doc = nlp(text)
    ner_skills = {ent.text.lower().strip() for ent in doc.ents if ent.label_ == "SKILL"}

    # Lowercase text once for keyword matching
    text_lower = text.lower()
    matched_skills = {skill for skill in skills_domain_lower if skill in text_lower}

    return list(ner_skills.union(matched_skills))


# Connect to MongoDB
client = MongoClient(MONGODB_URI)
db = client["career_link"]
collection = db["jobs"]

# Load CSV
df = pd.read_csv("./data/postings.csv",on_bad_lines="skip", engine="python")

# Optional: Clear collection
collection.delete_many({})

# Process and insert
for _, row in df.iterrows():
    # Basic null and empty checks
    if pd.isna(row["company_name"]) or pd.isna(row["title"]) or pd.isna(row["description"]):
        continue

    if not str(row["skills_desc"]).strip():
        continue

    combined_text = f"{str(row['description']).strip()} {str(row['skills_desc']).strip()}"
    extracted_skills = extract_skills(combined_text)

    if not extracted_skills:
        print("Skipping row due to no skills found.")
        continue

    salary_str = "Competitive"
    if pd.notna(row["normalized_salary"]):
        salary_str = f"${int(row['normalized_salary'])}"
    elif pd.notna(row["max_salary"]):
        salary_str = f"Up to ${int(row['max_salary'])}"

    job_doc = {
        "company": row["company_name"],
        "job_title": row["title"],
        "description": row["description"],
        "required_skills": extracted_skills,
        "location": row["location"] if pd.notna(row["location"]) else "Not specified",
        "salary": salary_str
    }

    # print(json.dumps(job_doc, indent=2))
    # print("=" * 60)

    # Insert into MongoDB
    collection.insert_one(job_doc)
    print("✅ MongoDB has been populated.")
