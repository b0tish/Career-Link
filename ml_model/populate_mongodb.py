import pandas as pd
import re
from pymongo import MongoClient
import spacy
import json
from skills_data import skills_domain

# Load the trained spaCy model (adjust path if needed)
nlp = spacy.load("./skill_ner_model")

def extract_skills(text: str):
    doc = nlp(text)
    ner_skills = {ent.text.lower().strip() for ent in doc.ents if ent.label_ == "SKILL"}

    # Token-based matching from the skills_domain
    text_lower = text.lower()
    matched_skills = {skill for skill in skills_domain if skill in text_lower}

    return list(ner_skills.union(matched_skills))


# client = MongoClient("mongodb://localhost:27017/")
# db = client["career_link"]
# collection = db["jobs"]
#
# Load CSV

df = pd.read_csv("./data/postings.csv")


# Optional: Clear collection first
# collection.delete_many({})

# Process and insert
for _, row in df.iterrows():
    if pd.isna(row["company_name"]) or pd.isna(row["title"]) or pd.isna(row["description"]):
        print("Skipping row due to missing fields:", row.to_dict())
        continue

    combined_text = f"{str(row['description']).strip()} {str(row['skills_desc']).strip()}"
    extracted_skills = extract_skills(combined_text)

    if not extracted_skills:
        print("Skipping row due to no skills found.")
        continue

    job_doc = {
        "company": row["company_name"],
        "job_title": row["title"],
        "description": row["description"],
        "required_skills": extracted_skills
    }

    print(json.dumps(job_doc, indent=2))
    print("=" * 60)

    # collection.insert_one(job_doc)

# print("✅ MongoDB has been populated.")
