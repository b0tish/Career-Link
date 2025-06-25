import pandas as pd
import spacy
import re

nlp = spacy.load("en_core_web_sm")


keywords = [
    "experience", "project", "management", "responsibility",
    "work history", "work experience", "job description",
    "role", "tasks", "positions", "certification", "abilities",
    "technical skills", "summary", "profile", "accomplishments"
]

def clean_html(text):
    text = re.sub(r"<[^>]+>", " ", str(text))
    return re.sub(r"\s+", " ", text).strip()

def extract_relevant_sentences(text):
    if pd.isna(text): return []
    text = clean_html(text)
    doc = nlp(text)

    
    keywords = ["experience", "skills", "expertise", "knowledge", "familiar", "proficient"]
    relevant_sentences = []

    for sent in doc.sents:
        sentence_text = sent.text.strip()
        sentence_lower = sentence_text.lower()
        for keyword in keywords:
            if keyword in sentence_lower:
                relevant_sentences.append(sentence_text)
                break
    return relevant_sentences

batch_size = 50
for start in range(51, len(df), batch_size):
    end = min(start + batch_size, len(df))
    print(f"Processing {start} to {end - 1}")

    df.loc[start:end - 1, "Relevant Sentences"] = df.loc[start:end - 1, "Resume_html"].apply(extract_relevant_sentences)

#model save

df.to_csv(r"D:\Project7\Career-Link\ml_model\data\Resume_processed.csv", index=False)
print("New CSV successfully saved.")

#skills

skills_domain = [
    "python", "java", "c++", "html", "css", "javascript", "react", "angular",
    "sql", "mongodb", "postgresql", "aws", "azure", "google cloud",
    "network administration", "cybersecurity", "sdlc", "git", "svn", "windows",
    "linux", "macos", "technical support", "devops", "patient care",
    "electronic health records", "ehr", "pharmacology", "hipaa", "diagnostic testing",
    "patient education", "budgeting & forecasting", "investment analysis",
    "risk management", "financial reporting", "auditing", "market research", "data entry",
    "bookkeeping", "accounts payable", "accounts receivable", "payroll processing",
    "customer relationship management", "statistical analysis", "data visualization",
    "tableau", "power bi", "predictive modeling", "machine learning", "data cleaning",
    "report generation", "business intelligence", "a/b testing", "excel", "active listening",
    "empathy", "troubleshooting", "complaint resolution", "product knowledge",
    "ticketing systems", "phone etiquette", "email communication", "live chat support",
    "service recovery", "digital marketing", "seo", "sem", "smm", "content marketing",
    "email marketing", "campaign management", "brand management", "public relations",
    "advertising", "google ads", "facebook ads", "google analytics", "copywriting",
    "hr", "spss", "sas", "hypothesis testing", "regression analysis", "anova",
    "probability theory", "experimental design", "data interpretation",
    "sampling techniques", "quantitative research", "statistical modeling",
    "project planning", "resource allocation", "team leadership",
    "stakeholder management", "agile methodologies", "scrum", "gantt charts", "jira",
    "asana", "trello", "verbal communication", "written communication",
    "presentation skills", "public speaking", "interpersonal skills",
    "technical writing", "critical thinking", "analytical skills",
    "root cause analysis", "decision making", "creative solutions",
    "strategic thinking", "resourcefulness", "motivation", "mentoring", "delegation",
    "performance management", "coaching", "strategic vision", "time management",
    "prioritization", "multitasking", "attention to detail", "record keeping",
    "meeting coordination", "filing systems", "workflow optimization", "r",
    "photoshop", "adobe photoshop", "final cut pro", "illustrator", "microsoft office",
    "figma", "accounting", "client relations", "data analysis", "customer service",
    "marketing", "statistics", "project management", "financial analysis",
    "communication", "problem-solving", "programming", "python", "java", "c++",
    "nlp", "tensorflow", "pytorch", "analytic skills", "leadership", "teamwork",
    "collaboration", "debugging", "testing", "agile", "databases", "networking",
    "cloud computing", "project coordination", "decision making", "conflict resolution",
    "creative thinking", "research", "report writing", "self-motivated", "self-starter",
    "photoshop", "wordpress"
]

def add_skills_to_sentences(sentences):
    matched= []
    text_lower = " ".join(sentences).lower()
    for skill in skills_domain:
        skill_lower = skill.lower()
        if re.search(rf'\b{re.escape(skill_lower)}\b', text_lower):
            matched.append(skill)
    return matched

df_reader = pd.read_csv("data/dataset_resume/Resume/Resume.csv",chunksize=100)


for i,chunk in enumerate(df_reader):
    print(f"Processing chunk {i+1}")
    
    # Extract relevant sentences from Resume_html
    chunk["Relevant Sentences"] = chunk["Resume_html"].apply(extract_relevant_sentences)   

    # Extract skills from relevant sentences
    chunk["matched_skills"] = chunk["Relevant Sentences"].apply(add_skills_to_sentences)

    # Merge both into raw_data column
    chunk["raw_data"] = chunk.apply(lambda row: (row["Relevant Sentences"], row["matched_skills"]), axis=1)

    # Save to CSV (first chunk with header, rest without)
    chunk[["raw_data"]].to_csv("relevant.csv", mode="a", index=False, header=(i==0))


import ast

df= pd.read_csv("relevant.csv")

def cleaned_text(raw):
    
# raw =df["raw_data"][800]
    parsed = ast.literal_eval(raw)

    cleaned_sentences = [re.sub(r'\\', '', s) for s in parsed[0]]
    cleaned_skills = [re.sub(r'\\', '', s) for s in parsed[1]]

    cleaned_tuple = (cleaned_sentences, cleaned_skills)
    return cleaned_tuple

# print(cleaned_tuple)

df["raw_data"] = df["raw_data"].apply(cleaned_text)
df["raw_data"].to_csv("raw_data.csv")

import ast

df= pd.read_csv("relevant.csv")

def cleaned_text(raw):
    
# raw =df["raw_data"][800]
    parsed = ast.literal_eval(raw)

    cleaned_sentences = [re.sub(r'\\', '', s) for s in parsed[0]]
    cleaned_skills = [re.sub(r'\\', '', s) for s in parsed[1]]

    cleaned_tuple = (cleaned_sentences, cleaned_skills)
    return cleaned_tuple
# print(cleaned_tuple)

df["raw_data"] = df["raw_data"].apply(cleaned_text)
df["raw_data"].to_csv("raw_data.csv")







def cleaned_text(raw):
    try:
        # Only parse if raw is a string
        if isinstance(raw, str):
            parsed = ast.literal_eval(raw)
        else:
            return ("", [])  # fallback for non-string rows

        # Clean slashes
        cleaned_sentences = [re.sub(r'\\', '', s) for s in parsed[0]]
        cleaned_skills = [re.sub(r'\\', '', s) for s in parsed[1]]

        # Join sentences and remove single/double quotes
        joined_sentences = " ".join(cleaned_sentences)
        joined_sentences = re.sub(r"[\"']", '', joined_sentences)

        return (joined_sentences, cleaned_skills)
    
    except (ValueError, SyntaxError, TypeError) as e:
        print(f"Skipping bad row: {raw} — {e}")
        return ("", [])

# Apply to 'raw_data' column
df["raw_data"] = df["raw_data"].apply(cleaned_text)

# Save cleaned output
df["raw_data"].to_csv("raw_data.csv", index=False)

    
               
import pandas as pd
import ast

# Read the cleaned CSV
df = pd.read_csv("raw_data.csv")

# Create list to hold the tuples
raw_data = []

for i, row in df.iterrows():
    try:
        value = row.iloc[0]  # ✅ safe and future-proof
        if isinstance(value, str):
            parsed = ast.literal_eval(value)
            raw_data.append(parsed)
    except Exception as e:
        print(f"Error at row {i}: {e}")

# Preview output
print(raw_data[1:3])  # Show first 2 items


#Train model
def create_training_example( text, skill_list):
    entities = []
    text_lower = text.lower()

    for skill in skill_list:
        skill_lower = skill.lower()
        start = text_lower.find(skill_lower)
        if start != -1:
            end = start + len(skill)
            entities.append((start, end, "SKILL"))

    return (text, {"entities": entities})


Train_data = [
    create_training_example(text, skills)
    for text, skills in raw_data
]

print(Train_data[800])




def filter_valid_entities(nlp, text, entities):
    doc = nlp.make_doc(text)
    valid_entities = []
    occupied = set()

    for start, end, label in sorted(entities, key=lambda x: x[0]):
        span = doc.char_span(start, end, label=label)
        if span and all(i not in occupied for i in range(span.start, span.end)):
            valid_entities.append((start, end, label))
            occupied.update(range(span.start, span.end))
    return valid_entities

# === Clean and prepare training data ===
cleaned_train_data = []
for text, ann in Train_data:
    entities = ann.get("entities", [])
    filtered_ents = filter_valid_entities(nlp, text, entities)
    for _, _, label in filtered_ents:
        ner.add_label(label)
    cleaned_train_data.append((text, {"entities": filtered_ents}))


for i in range(2):
    random.shuffle(cleaned_train_data)
    losses = {}
    for text, annotations in cleaned_train_data:
        example = Example.from_dict(nlp.make_doc(text), annotations)
        nlp.update([example], losses=losses)
    print(f"Epoch {i+1} - Loss: {losses.get('ner', 0):.4f}")

# === Save Model ===
model_path = "skill_ner_model"
nlp.to_disk(model_path)
print(f"✅ Model saved to: {model_path}")



import spacy

# Load the custom trained model from disk
nlp = spacy.load("skill_ner_model")

texts = [
    "I have experience in data analysis, project management, and communication.",
    "My skills include Python, machine learning, and customer service.",
    "Expertise in statistics, marketing strategy, and client relations.",
    "Worked on inventory management using Excel and SQL.",
    "Looking for a position where I can use my skills in accounting and finance."
]

for i, text in enumerate(texts, 1):
    doc = nlp(text)
    print(f"\nTest Case {i}:")
    print(f"Input: {text}")
    print("Extracted Skills:")
    for ent in doc.ents:
        if ent.label_ == "SKILL":
            print(f" - {ent.text}")
