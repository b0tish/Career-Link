import fitz  # PyMuPDF
import spacy
import os
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.parsers import MultiPartParser
from rest_framework import status
from pymongo import MongoClient
from dotenv import load_dotenv
import os

load_dotenv()
MONGODB_URI = os.getenv("MONGODB_URI")

# Load trained spaCy model once
# Get the absolute path to the model directory
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
model_path = os.path.join(BASE_DIR, "ml_model", "skill_ner_model")
nlp = spacy.load(model_path)

client = MongoClient(MONGODB_URI)
db = client["career_link"]                         
job_collection = db["jobs"]                        

def extract_text_from_pdf(pdf_file):
    doc = fitz.open(stream=pdf_file.read(), filetype="pdf")
    text = ""
    for page in doc:
        text += page.get_text()
    return text

def extract_skills(text):
    """
    Extract skills with improved post-processing to handle case sensitivity
    and common variations that the model might miss.
    """
    # First, get skills from the NER model
    doc = nlp(text)
    ner_skills = list({ent.text.lower() for ent in doc.ents if ent.label_ == "SKILL"})
    
    # Post-processing: check for skills that might have been missed due to case sensitivity
    text_lower = text.lower()
    additional_skills = []
    
    # Import skills list (you might need to adjust the path)
    import sys
    import os
    sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'ml_model'))
    from skills_data import skills_domain
    
    for skill in skills_domain:
        skill_lower = skill.lower().strip()
        if not skill_lower:
            continue
            
        # Check if skill is in text but wasn't detected by NER
        if skill_lower in text_lower and skill_lower not in ner_skills:
            # Verify it's a whole word match
            import re
            pattern = r'\b' + re.escape(skill_lower).replace(r'\ ', r'\s+') + r'\b'
            if re.search(pattern, text_lower):
                additional_skills.append(skill_lower)
    
    # Combine and deduplicate
    all_skills = list(set(ner_skills + additional_skills))
    
    return all_skills

def compute_match_score(user_skills, job_skills):  # NEW
    if not user_skills or not job_skills:
        return 0.0
    user_set = set(user_skills)
    job_set = set(skill.lower() for skill in job_skills)
    intersection = user_set & job_set
    union = user_set | job_set
    return len(intersection) / len(union)

class ResumeUploadView(APIView):
    parser_classes = [MultiPartParser]

    def post(self, request, format=None):
        resume_file = request.FILES.get('resume')
        if not resume_file:
            return Response({'error': 'No file uploaded'}, status=status.HTTP_400_BAD_REQUEST)

        try:
            # Extract skills from uploaded PDF
            text = extract_text_from_pdf(resume_file)
            extracted_skills = extract_skills(text)

            # Fetch jobs from MongoDB and compute similarity
            job_matches = []
            for job in job_collection.find():
                job_skills = job.get("required_skills", [])
                score = compute_match_score(extracted_skills, job_skills)

                job_matches.append({
                    "company": job.get("company"),
                    "job_title": job.get("job_title"),
                    "match_score": round(score, 3),
                    "required_skills": job_skills
                })

            # Sort by highest match score and get top 5
            job_matches = sorted(job_matches, key=lambda x: x["match_score"], reverse=True)
            top_jobs = job_matches[:5]

            return Response({
                'extracted_skills': extracted_skills,
                'top_jobs': top_jobs,
                'message': 'Skills extracted and top job matches generated.'
            })

        except Exception as e:
            return Response({'error': str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
