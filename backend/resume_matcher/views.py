import fitz  
import spacy
import os
from bson import ObjectId
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.parsers import MultiPartParser
from rest_framework import status, generics
from pymongo import MongoClient
from dotenv import load_dotenv
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

load_dotenv()
MONGODB_URI = os.getenv("MONGODB_URI")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
model_path = os.path.join(BASE_DIR, "ml_model", "skill_ner_model")
nlp = spacy.load(model_path)

client = MongoClient(MONGODB_URI)
db = client["career_link"]                         
job_collection = db["jobs"]                        

tfidf_vectorizer = TfidfVectorizer(stop_words='english')

try:
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
    all_jobs = []
    print(f"WARNING: Could not pre-fit TF-IDF vectorizer. It will be fitted on-the-fly. Error: {e}")


def extract_text_from_pdf(pdf_file):
    doc = fitz.open(stream=pdf_file.read(), filetype="pdf")
    text = ""
    for page in doc:
        text += page.get_text()
    return text

def extract_skills(text):
    doc = nlp(text)
    ner_skills = list({ent.text.lower() for ent in doc.ents if ent.label_ == "SKILL"})
    
    text_lower = text.lower()
    additional_skills = []
    
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

from users.permissions import IsJobProvider
from django.utils.decorators import method_decorator
from django.views.decorators.csrf import csrf_exempt

from .models import SavedJob, CVHistory
from .serializers import SavedJobSerializer, CVHistorySerializer
from rest_framework.permissions import IsAuthenticated

@method_decorator(csrf_exempt, name='dispatch')
class ResumeUploadView(APIView):
    parser_classes = [MultiPartParser]

    def post(self, request, format=None):
        print("ResumeUploadView post method called.")
        resume_file = request.FILES.get('resume')
        if not resume_file:
            return Response({'error': 'No file uploaded'}, status=status.HTTP_400_BAD_REQUEST)

        try:
            # Extract text and skills from uploaded PDF
            resume_text = extract_text_from_pdf(resume_file)
            extracted_skills = extract_skills(resume_text)

            if request.user.is_authenticated:
                print(f"User is authenticated: {request.user.is_authenticated}")
                cv_history_obj = CVHistory.objects.create(
                    user=request.user,
                    file_name=resume_file.name,
                    skills=", ".join(extracted_skills)
                )
                print(f"CVHistory object created: {cv_history_obj.file_name} for {cv_history_obj.user.username}")

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
                # Artificially boost skill_score by 20% and cap at 0.99
                skill_score = min(skill_score + 0.2, 0.99)

                # 2. Calculate KeywordOverlap using a more focused text representation
                # NEW: Compare job text against extracted skills instead of the whole resume
                resume_skills_text = " ".join(extracted_skills)
                job_text = job_title + " " + " ".join(job_skills)
                keyword_score = compute_keyword_overlap(resume_skills_text, job_text)
                # Artificially boost keyword_score by 20% and cap at 0.99
                keyword_score = min(keyword_score + 0.2, 0.99)

                # 3. Combine scores for final ranking
                final_score = (alpha * skill_score) + (beta * keyword_score)

                # The final score is now the match score
                inflated_score = min(final_score, 0.99)

                job_matches.append({
                    "id": str(job.get("_id")),
                    "company": job.get("company"),
                    "job_title": job_title,
                    "location": job.get("location"),
                    "salary": job.get("salary"),
                    "description": job.get("description"),
                    "match": round(inflated_score, 3),
                    "skill_score": skill_score,
                    "keyword_score": keyword_score,
                    "skills": ", ".join(job.get("required_skills", [])),
                    "url": job.get("url")
                })

            # Sort by highest match score and get top 5
            job_matches = sorted(job_matches, key=lambda x: x["match"], reverse=True)
            top_jobs = job_matches[:5]

            return Response({
                'extracted_skills': extracted_skills,
                'top_jobs': top_jobs,
                'message': 'Skills extracted and top job matches generated.'
            })

        except Exception as e:
            return Response({'error': str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

    def delete(self, request, pk, format=None):
        try:
            cv_entry = CVHistory.objects.get(pk=pk, user=request.user)
        except CVHistory.DoesNotExist:
            return Response(status=status.HTTP_404_NOT_FOUND)

        cv_entry.delete()
        return Response(status=status.HTTP_204_NO_CONTENT)

class SavedJobView(generics.DestroyAPIView, APIView): # Inherit from APIView to keep post method
    permission_classes = [IsAuthenticated]
    queryset = SavedJob.objects.all()
    lookup_field = 'pk'

    def get_queryset(self):
        return self.queryset.filter(user=self.request.user)

    def post(self, request, format=None):
        serializer = SavedJobSerializer(data=request.data)
        if serializer.is_valid():
            serializer.save(user=request.user)
            return Response(serializer.data, status=status.HTTP_201_CREATED)
        print(serializer.errors)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

class DashboardView(APIView):
    permission_classes = [IsAuthenticated]

    def get(self, request, format=None):
        saved_jobs = SavedJob.objects.filter(user=request.user)
        cv_history = CVHistory.objects.filter(user=request.user)

        saved_jobs_serializer = SavedJobSerializer(saved_jobs, many=True)
        cv_history_serializer = CVHistorySerializer(cv_history, many=True)

        return Response({
            'saved_jobs': saved_jobs_serializer.data,
            'cv_history': cv_history_serializer.data
        })

class JobProviderDashboardView(APIView):
    permission_classes = [IsJobProvider]

    def get(self, request, format=None):
        return Response({'message': 'Welcome to the Job Provider Dashboard!'})


class JobView(APIView):
    permission_classes = [IsJobProvider]

    def get(self, request, format=None):
        # Get jobs posted by the current job provider
        provider_jobs = job_collection.find({'provider': request.user.username})
        
        # Convert ObjectId to string for JSON serialization
        jobs_list = []
        for job in provider_jobs:
            job['_id'] = str(job['_id'])
            jobs_list.append(job)
            
        return Response(jobs_list)

    def post(self, request, format=None):
        # Add the provider to the job data
        job_data = request.data
        job_data['provider'] = request.user.username
        
        # Extract skills from the job description
        description = job_data.get('description', '')
        extracted_skills = extract_skills(description)
        job_data['required_skills'] = extracted_skills
        
        # Insert the new job into the collection
        result = job_collection.insert_one(job_data)
        
        # Return the created job with its new ID
        created_job = job_collection.find_one({'_id': result.inserted_id})
        created_job['_id'] = str(created_job['_id'])
        
        return Response(created_job, status=status.HTTP_201_CREATED)

    def delete(self, request, job_id, format=None):
        try:
            obj_id = ObjectId(job_id)
        except Exception:
            return Response({'error': 'Invalid ID format'}, status=status.HTTP_400_BAD_REQUEST)

        result = job_collection.delete_one({'_id': obj_id, 'provider': request.user.username})

        if result.deleted_count == 1:
            return Response(status=status.HTTP_204_NO_CONTENT)
        else:
            return Response({'error': 'Job not found or you do not have permission to delete it'}, status=status.HTTP_404_NOT_FOUND)
