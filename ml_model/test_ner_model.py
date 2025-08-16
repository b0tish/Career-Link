import spacy
import os
from spacy.training import Example
from spacy.scorer import Scorer
import re

# Test data constants
GOLD_RAW_DATA = [
    # Data Science & Analytics Examples
    ("Implemented advanced machine learning models using TensorFlow and PyTorch for computer vision applications.", 
     ["machine learning", "tensorflow", "pytorch", "computer vision"]),
    ("Built data pipelines using Apache Spark and Kafka for real-time streaming analytics and big data processing.", 
     ["apache spark", "kafka", "real-time streaming analytics", "big data"]),
    ("Created interactive dashboards using Tableau and Power BI for business intelligence and data visualization.", 
     ["tableau", "power bi", "business intelligence", "data visualization"]),
    ("Performed statistical analysis using R and Python for hypothesis testing and predictive modeling.", 
     ["statistical analysis", "r", "python", "hypothesis testing", "predictive modeling"]),
    
    # Cloud & DevOps Examples
    ("Deployed microservices architecture on AWS using ECS, Lambda functions, and API Gateway with Terraform.", 
     ["microservices architecture", "aws", "ecs", "lambda functions", "api gateway", "terraform"]),
    ("Implemented CI/CD pipelines using GitHub Actions and Docker for automated testing and deployment.", 
     ["ci/cd", "github actions", "docker", "automated testing"]),
    ("Managed Kubernetes clusters on Google Cloud Platform using GKE and Helm charts for container orchestration.", 
     ["kubernetes", "google cloud platform", "gke", "helm", "container orchestration"]),
    ("Set up monitoring stack using Prometheus, Grafana, and Alertmanager for observability and alerting.", 
     ["prometheus", "grafana", "alertmanager", "observability", "alerting"]),
    
    # Web Development Examples
    ("Developed full-stack applications using React, Node.js, and PostgreSQL with RESTful API design.", 
     ["full-stack applications", "react", "node.js", "postgresql", "restful api"]),
    ("Built responsive web applications using Angular, TypeScript, and Bootstrap with progressive web app features.", 
     ["responsive web applications", "angular", "typescript", "bootstrap", "progressive web app"]),
    ("Created dynamic websites using Vue.js, Laravel, and MySQL with authentication and authorization.", 
     ["dynamic websites", "vue.js", "laravel", "mysql", "authentication", "authorization"]),
    ("Implemented GraphQL APIs using Apollo Server and MongoDB with real-time subscriptions.", 
     ["graphql", "apollo server", "mongodb", "real-time subscriptions"]),
    
    # Mobile Development Examples
    ("Built cross-platform mobile applications using Flutter and Dart with Firebase backend services.", 
     ["cross-platform mobile applications", "flutter", "dart", "firebase"]),
    ("Developed native iOS applications using Swift and Xcode with Core Data for local storage.", 
     ["native ios applications", "swift", "xcode", "core data"]),
    ("Created Android applications using Kotlin and Android Studio with Room database integration.", 
     ["android applications", "kotlin", "android studio", "room database"]),
    ("Implemented push notifications and in-app messaging using Firebase Cloud Messaging.", 
     ["push notifications", "in-app messaging", "firebase cloud messaging"]),
    
    # Database & Backend Examples
    ("Designed scalable database architectures using MySQL, Redis, and Elasticsearch for high-performance applications.", 
     ["scalable database architectures", "mysql", "redis", "elasticsearch", "high-performance applications"]),
    ("Implemented data warehousing solutions using Snowflake and dbt for ETL processes and analytics.", 
     ["data warehousing", "snowflake", "dbt", "etl processes", "analytics"]),
    ("Built NoSQL database systems using MongoDB and Cassandra for distributed data storage.", 
     ["nosql database systems", "mongodb", "cassandra", "distributed data storage"]),
    ("Created API gateways using Kong and OAuth2.0 for secure authentication and rate limiting.", 
     ["api gateways", "kong", "oauth2.0", "secure authentication", "rate limiting"]),
    
    # AI & Machine Learning Examples
    ("Trained deep learning models using TensorFlow and Keras for natural language processing tasks.", 
     ["deep learning", "tensorflow", "keras", "natural language processing"]),
    ("Implemented computer vision solutions using OpenCV and PyTorch for image classification and object detection.", 
     ["computer vision", "opencv", "pytorch", "image classification", "object detection"]),
    ("Built recommendation systems using collaborative filtering and matrix factorization algorithms.", 
     ["recommendation systems", "collaborative filtering", "matrix factorization"]),
    ("Developed chatbot applications using Rasa and Dialogflow with natural language understanding.", 
     ["chatbot applications", "rasa", "dialogflow", "natural language understanding"]),
    
    # Security & Networking Examples
    ("Implemented cybersecurity measures using penetration testing and vulnerability assessment tools.", 
     ["cybersecurity", "penetration testing", "vulnerability assessment"]),
    ("Configured network security using firewalls and intrusion detection systems for threat prevention.", 
     ["network security", "firewalls", "intrusion detection systems", "threat prevention"]),
    ("Set up identity and access management using OAuth2.0 and JWT tokens for secure authentication.", 
     ["identity and access management", "oauth2.0", "jwt", "secure authentication"]),
    ("Implemented encryption and data loss prevention strategies for compliance and security.", 
     ["encryption", "data loss prevention", "compliance", "security"]),
    
    # Project Management Examples
    ("Led agile development teams using Scrum methodology and Jira for project tracking and sprint planning.", 
     ["agile development", "scrum", "jira", "project tracking", "sprint planning"]),
    ("Managed cross-functional teams using Kanban boards and Confluence for documentation and collaboration.", 
     ["cross-functional teams", "kanban", "confluence", "documentation", "collaboration"]),
    ("Implemented DevOps practices using Git, Jenkins, and Ansible for continuous integration and deployment.", 
     ["devops", "git", "jenkins", "ansible", "continuous integration", "deployment"]),
    ("Coordinated stakeholder communication using Slack and Microsoft Teams for remote team collaboration.", 
     ["stakeholder communication", "slack", "microsoft teams", "remote team collaboration"]),
    
    # Business Intelligence Examples
    ("Created executive dashboards using Power BI and Tableau for key performance indicators and reporting.", 
     ["executive dashboards", "power bi", "tableau", "key performance indicators", "reporting"]),
    ("Implemented business intelligence solutions using SQL Server and SSRS for data analysis and visualization.", 
     ["business intelligence", "sql server", "ssrs", "data analysis", "visualization"]),
    ("Developed data storytelling techniques using Python and Matplotlib for presentation and communication.", 
     ["data storytelling", "python", "matplotlib", "presentation", "communication"]),
    ("Built predictive analytics models using scikit-learn and pandas for forecasting and trend analysis.", 
     ["predictive analytics", "scikit-learn", "pandas", "forecasting", "trend analysis"]),
    
    # Negative Examples (should not extract skills)
    ("I am passionate about learning new technologies and adapting to changing environments.", []),
    ("She demonstrates excellent leadership qualities and strong communication skills in team settings.", []),
    ("He has a proven track record of delivering high-quality results under pressure.", []),
    ("The company culture emphasizes innovation, collaboration, and continuous improvement.", []),
    ("She excels at problem-solving and enjoys working in fast-paced, dynamic environments.", []),
    ("He is committed to professional development and staying current with industry trends.", []),
    ("The team works effectively together and maintains high standards of quality.", []),
    ("She has a strong work ethic and consistently meets project deadlines.", [])
]

def create_skill_pattern(skill):
    """Create regex pattern for skill matching with case variations."""
    skill_lower = skill.lower().strip()
    pattern = r'\b' + re.escape(skill_lower) + r'\b'
    return pattern.replace(r'\ ', r'\s+')  # Handle spaces in multi-word skills

def get_aligned_entities(nlp_model, text, gold_skills):
    """Generate aligned entities for a given text and list of gold skill strings."""
    doc = nlp_model.make_doc(text)
    entities = []
    text_lower = text.lower()

    for skill in gold_skills:
        pattern = create_skill_pattern(skill)
        
        for match in re.finditer(pattern, text_lower):
            start_char, end_char = match.span()
            span = doc.char_span(start_char, end_char, alignment_mode="expand")
            
            if span is not None:
                entities.append((span.start_char, span.end_char, "SKILL"))

    # Deduplicate and sort entities
    entities = sorted(list(set(entities)), key=lambda x: x[0])
    
    # Merge overlapping entities
    return merge_overlapping_entities(entities)

def merge_overlapping_entities(entities):
    """Merge overlapping or adjacent entities."""
    if not entities:
        return []
    
    final_entities = []
    current_end = -1
    
    for start, end, label in entities:
        if start >= current_end:
            final_entities.append((start, end, label))
            current_end = end
        elif end > current_end:  # If current entity overlaps but extends further
            if final_entities:
                last_added_start, _, last_added_label = final_entities[-1]
                # Update the last added entity to cover the larger span
                final_entities[-1] = (last_added_start, end, last_added_label)
                current_end = end
            else:
                final_entities.append((start, end, label))
                current_end = end
    
    return final_entities

def create_reference_doc(nlp_model, text, gold_skills):
    """Create a reference document with aligned entities."""
    reference_doc = nlp_model.make_doc(text)
    reference_entities = get_aligned_entities(nlp_model, text, gold_skills)
    
    # Clear default ents and set aligned entities
    reference_doc.ents = []
    aligned_spans = [
        reference_doc.char_span(s, e, label=l) 
        for s, e, l in reference_entities 
        if reference_doc.char_span(s, e, label=l) is not None
    ]
    reference_doc.set_ents(aligned_spans)
    
    return reference_doc

def create_training_examples(nlp_model, gold_data):
    """Create training examples from gold data."""
    examples = []
    
    for text, gold_skills in gold_data:
        reference_doc = create_reference_doc(nlp_model, text, gold_skills)
        predicted_doc = nlp_model(text)
        
        example = Example(predicted_doc, reference_doc)
        examples.append(example)
    
    return examples

def evaluate_model_performance(scorer, examples):
    """Evaluate model performance using the scorer."""
    scores = scorer.score(examples)
    
    print("\n--- NER Model Evaluation Metrics (SKILL entity) ---")
    if "ents_per_type" in scores and "SKILL" in scores["ents_per_type"]:
        skill_scores = scores["ents_per_type"]["SKILL"]
        print(f"Precision (P): {skill_scores['p']:.4f}")
        print(f"Recall (R): {skill_scores['r']:.4f}")
        print(f"F1-Score (F): {skill_scores['f']:.4f}")
    else:
        print("No 'SKILL' entity found in evaluation, or metrics not calculated for 'SKILL'.")
        # Print overall entity metrics
        print(f"Overall Entity Precision: {scores.get('ents_p', 0.0):.4f}")
        print(f"Overall Entity Recall: {scores.get('ents_r', 0.0):.4f}")
        print(f"Overall Entity F1-Score: {scores.get('ents_f', 0.0):.4f}")
    
    return scores

def display_model_predictions(nlp_model, gold_data, max_display=30):
    """Display model predictions for visual inspection."""
    print(f"\n--- NER Model Predictions (showing first {max_display} cases) ---")
    
    for i, (text, gold_skills) in enumerate(gold_data[:max_display], 1):
        doc = nlp_model(text)
        print(f"\nTest Case {i}:")
        print(f"Input: {text}")
        
        print("Extracted Skills (NER):")
        found_skills = [ent.text for ent in doc.ents if ent.label_ == "SKILL"]
        
        if found_skills:
            for skill in found_skills:
                print(f" - {skill}")
        else:
            print(" - No skills found by NER model.")
        
        # Compare predictions with gold standard
        pred_skills = set([ent.text.lower().strip() for ent in doc.ents if ent.label_ == "SKILL"])
        gold_skills_set = set([s.lower().strip() for s in gold_skills])
        
        missed = gold_skills_set - pred_skills
        extra = pred_skills - gold_skills_set
        
        if missed:
            print(f"  Missed skills: {missed}")
        if extra:
            print(f"  Extra (false positive) skills: {extra}")

def test_ner_model(model_path="skill_ner_model"):
    """Load and test the trained NER model with evaluation metrics."""
    print("\n=== PHASE 3: MODEL TESTING ===")

    if not os.path.exists(model_path):
        print(f"❌ No trained model found at {model_path}. Please train the model first.")
        return

    try:
        trained_nlp = spacy.load(model_path)
        print(f"Trained model loaded from: {model_path}")
        
        # Verify NER component exists
        if 'ner' not in trained_nlp.pipe_names:
            print("ERROR: 'ner' component not found in the loaded model's pipeline.")
            print("Model pipeline:", trained_nlp.pipe_names)
            return
        
        # Create training examples
        examples = create_training_examples(trained_nlp, GOLD_RAW_DATA)
        
        # Evaluate model performance
        scorer = Scorer()
        scores = evaluate_model_performance(scorer, examples)
        
        # Display predictions for visual inspection
        display_model_predictions(trained_nlp, GOLD_RAW_DATA)
        
    except Exception as e:
        print(f"Error during model testing: {e}")

if __name__ == "__main__":
    test_ner_model()
