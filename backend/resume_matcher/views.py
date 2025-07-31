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
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

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

# --- NEW: Pre-fit TF-IDF Vectorizer at startup ---
tfidf_vectorizer = TfidfVectorizer(stop_words='english')

try:
    # Fetch all job data from the database once
    all_jobs_cursor = job_collection.find()
    all_jobs = list(all_jobs_cursor)
    
    # Create a corpus of job content (title + skills)
    job_corpus = [
        job.get("job_title", "") + " " + " ".join(job.get("required_skills", []))
        for job in all_jobs
    ]
    
    # If there are jobs, fit the vectorizer to learn the vocabulary and IDF weights
    if job_corpus:
        tfidf_vectorizer.fit(job_corpus)
    print("TF-IDF vectorizer pre-fitted successfully on startup.")

except Exception as e:
    # Handle potential DB connection errors or other issues at startup
    all_jobs = [] # Ensure all_jobs is defined to prevent runtime errors
    print(f"WARNING: Could not pre-fit TF-IDF vectorizer. It will be fitted on-the-fly. Error: {e}")
# --- END NEW ---


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

def compute_skill_match_score(user_skills, job_skills):
    if not user_skills or not job_skills:
        return 0.0
    user_set = set(user_skills)
    job_set = set(skill.lower() for skill in job_skills)
    intersection = user_set & job_set
    # Avoid division by zero if a job has no skills listed
    return len(intersection) / len(job_set) if len(job_set) > 0 else 0.0

def compute_keyword_overlap(resume_text, job_text):
    if not resume_text or not job_text or not hasattr(tfidf_vectorizer, 'vocabulary_') or not tfidf_vectorizer.vocabulary_:
        return 0.0
    
    # Use the globally pre-fitted vectorizer to transform the texts
    try:
        tfidf_matrix = tfidf_vectorizer.transform([resume_text, job_text])
        cosine_sim = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])
        return cosine_sim[0][0]
    except Exception:
        # This can happen if the text contains only words not in the vocabulary
        return 0.0

class ResumeUploadView(APIView):
    parser_classes = [MultiPartParser]

    def post(self, request, format=None):
        resume_file = request.FILES.get('resume')
        if not resume_file:
            return Response({'error': 'No file uploaded'}, status=status.HTTP_400_BAD_REQUEST)

        try:
            # Extract text and skills from uploaded PDF
            resume_text = extract_text_from_pdf(resume_file)
            extracted_skills = extract_skills(resume_text)

            # NEW: Adjusted weights for a more balanced score
            alpha = 0.6  # Weight for skill match
            beta = 0.4   # Weight for keyword overlap

            # Fetch jobs from MongoDB and compute similarity
            job_matches = []
            # Use the 'all_jobs' list fetched at startup to avoid redundant DB calls
            for job in all_jobs:
                job_skills = job.get("required_skills", [])
                job_title = job.get("job_title", "")

                # 1. Calculate SkillMatchScore
                skill_score = compute_skill_match_score(extracted_skills, job_skills)

                # 2. Calculate KeywordOverlap using a more focused text representation
                # NEW: Compare job text against extracted skills instead of the whole resume
                resume_skills_text = " ".join(extracted_skills)
                job_text = job_title + " " + " ".join(job_skills)
                keyword_score = compute_keyword_overlap(resume_skills_text, job_text)

                # 3. Combine scores for final ranking
                final_score = (alpha * skill_score) + (beta * keyword_score)

                job_matches.append({
                    "company": job.get("company"),
                    "job_title": job_title,
                    "match_score": round(final_score, 3),
                    "skill_match": round(skill_score, 3),
                    "keyword_match": round(keyword_score, 3),
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
