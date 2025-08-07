import spacy
import os
from spacy.training import Example
from spacy.scorer import Scorer
import re

# Helper function to create aligned entities for a Doc object
def get_aligned_entities(nlp_model, text, gold_skills):
    """
    Generates aligned (start_char, end_char, label) entities for a given text
    and list of gold skill strings, based on the nlp_model's tokenizer.
    Case-insensitive matching for better alignment.
    """
    doc = nlp_model.make_doc(text)
    entities = []
    text_lower = text.lower()

    for skill in gold_skills:
        skill_lower = skill.lower().strip()
        # Create case-insensitive pattern
        pattern = r'\b' + re.escape(skill_lower) + r'\b'
        pattern = pattern.replace(r'\ ', r'\s+') # Handle spaces in multi-word skills

        for match in re.finditer(pattern, text_lower):
            start_char, end_char = match.span()
            span = doc.char_span(start_char, end_char, alignment_mode="expand")
            
            if span is not None:
                entities.append((span.start_char, span.end_char, "SKILL"))
            # else:
                # print(f"DEBUG: Failed to align gold skill '{skill}' in '{text[start_char:end_char]}' within '{text}'")

    # Deduplicate and sort entities
    entities = sorted(list(set(entities)), key=lambda x: x[0])

    # Basic non-overlapping check (simplified for this context, rely on 'expand' for most)
    final_entities = []
    current_end = -1
    for start, end, label in entities:
        if start >= current_end:
            final_entities.append((start, end, label))
            current_end = end
        elif end > current_end: # If current entity overlaps but extends further
            if final_entities:
                last_added_start, _, last_added_label = final_entities[-1]
                # Update the last added entity to cover the larger span
                final_entities[-1] = (last_added_start, end, last_added_label)
                current_end = end
            else: # Should not happen if start >= current_end handles the first one
                final_entities.append((start, end, label))
                current_end = end
    
    return final_entities


def test_ner_model(model_path="skill_ner_model"):
    """
    Loads the trained NER model and tests it with sample texts,
    including evaluation metrics.
    Also demonstrates the enhanced skill detection function.
    """
    print("\n=== PHASE 3: MODEL TESTING ===")

    if os.path.exists(model_path):
        trained_nlp = spacy.load(model_path)
        print(f"Trained model loaded from: {model_path}")

        # The raw text and the list of actual skills
        gold_raw_data = [
            # Data Science & Analytics Examples
            ("Implemented advanced machine learning models using TensorFlow and PyTorch for computer vision applications.", ["machine learning", "tensorflow", "pytorch", "computer vision"]),
            ("Built data pipelines using Apache Spark and Kafka for real-time streaming analytics and big data processing.", ["apache spark", "kafka", "real-time streaming analytics", "big data"]),
            ("Created interactive dashboards using Tableau and Power BI for business intelligence and data visualization.", ["tableau", "power bi", "business intelligence", "data visualization"]),
            ("Performed statistical analysis using R and Python for hypothesis testing and predictive modeling.", ["statistical analysis", "r", "python", "hypothesis testing", "predictive modeling"]),
            
            # Cloud & DevOps Examples
            ("Deployed microservices architecture on AWS using ECS, Lambda functions, and API Gateway with Terraform.", ["microservices architecture", "aws", "ecs", "lambda functions", "api gateway", "terraform"]),
            ("Implemented CI/CD pipelines using GitHub Actions and Docker for automated testing and deployment.", ["ci/cd", "github actions", "docker", "automated testing"]),
            ("Managed Kubernetes clusters on Google Cloud Platform using GKE and Helm charts for container orchestration.", ["kubernetes", "google cloud platform", "gke", "helm", "container orchestration"]),
            ("Set up monitoring stack using Prometheus, Grafana, and Alertmanager for observability and alerting.", ["prometheus", "grafana", "alertmanager", "observability", "alerting"]),
            
            # Web Development Examples
            ("Developed full-stack applications using React, Node.js, and PostgreSQL with RESTful API design.", ["full-stack applications", "react", "node.js", "postgresql", "restful api"]),
            ("Built responsive web applications using Angular, TypeScript, and Bootstrap with progressive web app features.", ["responsive web applications", "angular", "typescript", "bootstrap", "progressive web app"]),
            ("Created dynamic websites using Vue.js, Laravel, and MySQL with authentication and authorization.", ["dynamic websites", "vue.js", "laravel", "mysql", "authentication", "authorization"]),
            ("Implemented GraphQL APIs using Apollo Server and MongoDB with real-time subscriptions.", ["graphql", "apollo server", "mongodb", "real-time subscriptions"]),
            
            # Mobile Development Examples
            ("Built cross-platform mobile applications using Flutter and Dart with Firebase backend services.", ["cross-platform mobile applications", "flutter", "dart", "firebase"]),
            ("Developed native iOS applications using Swift and Xcode with Core Data for local storage.", ["native ios applications", "swift", "xcode", "core data"]),
            ("Created Android applications using Kotlin and Android Studio with Room database integration.", ["android applications", "kotlin", "android studio", "room database"]),
            ("Implemented push notifications and in-app messaging using Firebase Cloud Messaging.", ["push notifications", "in-app messaging", "firebase cloud messaging"]),
            
            # Database & Backend Examples
            ("Designed scalable database architectures using MySQL, Redis, and Elasticsearch for high-performance applications.", ["scalable database architectures", "mysql", "redis", "elasticsearch", "high-performance applications"]),
            ("Implemented data warehousing solutions using Snowflake and dbt for ETL processes and analytics.", ["data warehousing", "snowflake", "dbt", "etl processes", "analytics"]),
            ("Built NoSQL database systems using MongoDB and Cassandra for distributed data storage.", ["nosql database systems", "mongodb", "cassandra", "distributed data storage"]),
            ("Created API gateways using Kong and OAuth2.0 for secure authentication and rate limiting.", ["api gateways", "kong", "oauth2.0", "secure authentication", "rate limiting"]),
            
            # AI & Machine Learning Examples
            ("Trained deep learning models using TensorFlow and Keras for natural language processing tasks.", ["deep learning", "tensorflow", "keras", "natural language processing"]),
            ("Implemented computer vision solutions using OpenCV and PyTorch for image classification and object detection.", ["computer vision", "opencv", "pytorch", "image classification", "object detection"]),
            ("Built recommendation systems using collaborative filtering and matrix factorization algorithms.", ["recommendation systems", "collaborative filtering", "matrix factorization"]),
            ("Developed chatbot applications using Rasa and Dialogflow with natural language understanding.", ["chatbot applications", "rasa", "dialogflow", "natural language understanding"]),
            
            # Security & Networking Examples
            ("Implemented cybersecurity measures using penetration testing and vulnerability assessment tools.", ["cybersecurity", "penetration testing", "vulnerability assessment"]),
            ("Configured network security using firewalls and intrusion detection systems for threat prevention.", ["network security", "firewalls", "intrusion detection systems", "threat prevention"]),
            ("Set up identity and access management using OAuth2.0 and JWT tokens for secure authentication.", ["identity and access management", "oauth2.0", "jwt", "secure authentication"]),
            ("Implemented encryption and data loss prevention strategies for compliance and security.", ["encryption", "data loss prevention", "compliance", "security"]),
            
            # Project Management Examples
            ("Led agile development teams using Scrum methodology and Jira for project tracking and sprint planning.", ["agile development", "scrum", "jira", "project tracking", "sprint planning"]),
            ("Managed cross-functional teams using Kanban boards and Confluence for documentation and collaboration.", ["cross-functional teams", "kanban", "confluence", "documentation", "collaboration"]),
            ("Implemented DevOps practices using Git, Jenkins, and Ansible for continuous integration and deployment.", ["devops", "git", "jenkins", "ansible", "continuous integration", "deployment"]),
            ("Coordinated stakeholder communication using Slack and Microsoft Teams for remote team collaboration.", ["stakeholder communication", "slack", "microsoft teams", "remote team collaboration"]),
            
            # Business Intelligence Examples
            ("Created executive dashboards using Power BI and Tableau for key performance indicators and reporting.", ["executive dashboards", "power bi", "tableau", "key performance indicators", "reporting"]),
            ("Implemented business intelligence solutions using SQL Server and SSRS for data analysis and visualization.", ["business intelligence", "sql server", "ssrs", "data analysis", "visualization"]),
            ("Developed data storytelling techniques using Python and Matplotlib for presentation and communication.", ["data storytelling", "python", "matplotlib", "presentation", "communication"]),
            ("Built predictive analytics models using scikit-learn and pandas for forecasting and trend analysis.", ["predictive analytics", "scikit-learn", "pandas", "forecasting", "trend analysis"]),
            
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

        # Prepare examples for the scorer using explicit reference and predicted docs
        examples = []
        for text, gold_skills in gold_raw_data:
            reference_doc = trained_nlp.make_doc(text)
            reference_entities = get_aligned_entities(trained_nlp, text, gold_skills)
            reference_doc.ents = trained_nlp.make_doc(text).ents # Clear default ents if any
            reference_doc.set_ents([reference_doc.char_span(s, e, label=l) for s, e, l in reference_entities if reference_doc.char_span(s, e, label=l) is not None])


            # Run the trained model on the text to get predictions
            predicted_doc = trained_nlp(text)
            
            # Create the Example object with both reference and predicted docs
            example = Example(predicted_doc, reference_doc)
            examples.append(example)

        # Confirm that the NER component exists in the loaded model's pipeline
        if 'ner' not in trained_nlp.pipe_names:
            print("ERROR: 'ner' component not found in the loaded model's pipeline.")
            print("Model pipeline:", trained_nlp.pipe_names)
            return # Exit if NER isn't present

        scorer = Scorer() # Initialize without the nlp object for this method of scoring
        scores = scorer.score(examples)

        print("\n--- NER Model Evaluation Metrics (SKILL entity) ---")
        if "ents_per_type" in scores and "SKILL" in scores["ents_per_type"]:
            skill_scores = scores["ents_per_type"]["SKILL"]
            print(f"Precision (P): {skill_scores['p']:.4f}")
            print(f"Recall (R): {skill_scores['r']:.4f}")
            print(f"F1-Score (F): {skill_scores['f']:.4f}")
        else:
            print("No 'SKILL' entity found in evaluation, or metrics not calculated for 'SKILL'.")
            # If still zero, print overall entity metrics to see if other labels are being predicted
            print(f"Overall Entity Precision: {scores.get('ents_p', 0.0):.4f}")
            print(f"Overall Entity Recall: {scores.get('ents_r', 0.0):.4f}")
            print(f"Overall Entity F1-Score: {scores.get('ents_f', 0.0):.4f}")

        # The debugging section should now give you even more confidence
        # print("\n--- DEBUG: Inspecting Model Predictions vs. Gold Standard (Detailed) ---")
        # for i, example in enumerate(examples[:5]): # Check first 5 examples
        #     print(f"\nText: {example.reference.text}")
        #
        #     print("  Gold Entities (from reference_doc):")
        #     if example.reference.ents:
        #         for ent in example.reference.ents:
        #             print(f"    - '{ent.text}' ({ent.label_}) Start: {ent.start_char}, End: {ent.end_char}")
        #     else:
        #         print("    - No gold entities for this text.")
        #
        #     print("  Predicted Entities (from predicted_doc):")
        #     if example.predicted.ents: # Use example.predicted.ents now
        #         for ent in example.predicted.ents:
        #             print(f"    - '{ent.text}' ({ent.label_}) Start: {ent.start_char}, End: {ent.end_char}")
        #     else:
        #         print("    - No predicted entities for this text.")
        # print("----------------------------------------------------------")


        print("\n--- NER Model Predictions (for visual inspection) ---")
        # Keep some visual inspection for individual sentences
        for i, (text, gold_skills) in enumerate(gold_raw_data, 1): # Display first 5 for brevity
            doc = trained_nlp(text)
            print(f"\nTest Case {i}:")
            print(f"Input: {text}")
            print("Extracted Skills (NER):")
            found_skills = [ent.text for ent in doc.ents if ent.label_ == "SKILL"]
            if found_skills:
                for skill in found_skills:
                    print(f" - {skill}")
            else:
                print(" - No skills found by NER model.")
            pred_skills = set([ent.text.lower().strip() for ent in doc.ents if ent.label_ == "SKILL"])
            gold_skills_set = set([s.lower().strip() for s in gold_skills])
            missed = gold_skills_set - pred_skills
            extra = pred_skills - gold_skills_set
            if missed:
                print(f"Test Case {i}: Missed skills: {missed}")
            if extra:
                print(f"Test Case {i}: Extra (false positive) skills: {extra}")

        # Test the enhanced skill detection function separately
        # print("\n--- ENHANCED SKILL DETECTION (using generate_data.py function) ---")
        # sample_text_for_enhanced_detection = "My professional profile includes expertise in Advanced Python, Java Enterprise Edition, and intricate Data Analysis techniques. I've also managed cross-functional teams in Project Management and implemented Machine Learning solutions for clients. Furthermore, I am proficient in SQL Server, AWS Cloud services, and have strong Communication skills. My experience extends to Digital Marketing, Content Creation, and utilizing Adobe Photoshop for graphic design. I've also worked with Primavera Complete for project scheduling."
        #
        # detected_skills = add_skills_to_sentences([sample_text_for_enhanced_detection])
        # print(f"Input: {sample_text_for_enhanced_detection[:200]}...")
        # print(f"Skills detected by enhanced function: {detected_skills}")
        # print(f"Number of skills found: {len(detected_skills)}")

    else:
        print(f"❌ No trained model found at {model_path}. Please train the model first.")

if __name__ == "__main__":
    test_ner_model()
