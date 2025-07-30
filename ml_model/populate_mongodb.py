import pandas as pd
import re
# from pymongo import MongoClient
import spacy

# Load the trained spaCy model (adjust path if needed)
nlp = spacy.load("./skill_ner_model")

def extract_skills(text: str):
    doc = nlp(text)
    skills = set(ent.text.lower() for ent in doc.ents if ent.label_ == "SKILL")
    return list(skills)

# client = MongoClient("mongodb://localhost:27017/")
# db = client["career_link"]
# collection = db["jobs"]
#
# Load CSV

df = pd.read_csv("./data/postings.csv")


# Optional: Clear collection first
# collection.delete_many({})

# Process and insert
for _, row in df.iloc[0:5].iterrows():
    combined_text = f"{row['description']} {row['skills_desc']}"
    extracted_skills = extract_skills(combined_text)

    job_doc = {
        "company": row["company_name"],
        "job_title": row["title"],
        "description": row["description"],
        "required_skills": list(set(extracted_skills))
    }


    print(job_doc)
    print()
    print()


    # collection.insert_one(job_doc)

# print("✅ MongoDB has been populated.")
