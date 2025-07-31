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
    """
    doc = nlp_model.make_doc(text)
    entities = []
    text_lower = text.lower()

    for skill in gold_skills:
        skill_lower = skill.lower().strip()
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
            ("Developed a full-stack solution using React for the frontend and Node.js for the backend. Deployed with Docker and Kubernetes.", ["react", "node.js", "docker", "kubernetes"]),
            ("My role involved data analysis with Python, using libraries like Pandas and NumPy. I also have experience with SQL.", ["data analysis", "python", "pandas", "numpy", "sql"]),
            ("I have no relevant skills.", []),
            ("The team used Agile methodologies, specifically Scrum, to manage the project. We tracked our work in JIRA.", ["agile", "scrum", "jira"]),
            ("I am proficient in Microsoft Office, including Excel and Word.", ["microsoft office", "excel", "word"]),
            ("The project required knowledge of cloud computing, specifically AWS and Azure.", ["cloud computing", "aws", "azure"]),
            ("I have experience with machine learning, using TensorFlow and PyTorch.", ["machine learning", "tensorflow", "pytorch"]),
            ("Built RESTful APIs using Django and Flask, and managed deployments with Docker and Kubernetes.", ["django", "flask", "docker", "kubernetes", "restful apis"]),
            ("Designed and implemented user interfaces with Figma and Sketch, focusing on user-centered design principles.", ["figma", "sketch", "user-centered design"]),
            ("Expert in digital marketing strategies, including SEO, SEM, and content creation for social media platforms.", ["digital marketing", "seo", "sem", "content creation", "social media"]),
            ("Administered patient care, managed electronic health records (EHR), and provided critical care support.", ["patient care", "electronic health records", "ehr", "critical care"]),
            ("Developed comprehensive lesson plans for high school students, utilizing Google Classroom and differentiated instruction.", ["lesson planning", "google classroom", "differentiated instruction"]),
            ("Successfully led cross-functional teams in software development life cycles and quality assurance.", ["team leadership", "software development life cycle", "quality assurance"]),
            ("Conducted in-depth market research and competitive analysis to identify new business opportunities.", ["market research", "competitive analysis"]),
            ("Optimized cloud infrastructure on AWS and handled database administration for PostgreSQL.", ["cloud infrastructure", "aws", "database administration", "postgresql"]),
            ("Skilled in financial reporting, budgeting, and risk management for enterprise-level operations.", ["financial reporting", "budgeting", "risk management"]),
            ("Implemented cybersecurity protocols and conducted vulnerability assessments to protect sensitive data.", ["cybersecurity", "vulnerability assessments"]),
            ("Created engaging visual designs using Adobe Creative Suite, including Photoshop and Illustrator.", ["adobe creative suite", "photoshop", "illustrator", "visual design"]),
            ("Facilitated training sessions and workshops, demonstrating strong presentation and public speaking skills.", ["training", "workshops", "presentation skills", "public speaking"]),
            ("Analyzed large datasets using R and Pandas to derive actionable insights for strategic decision-making.", ["r", "pandas", "data analysis", "strategic decision-making"]),
            ("Managed supplier relationships and optimized supply chain logistics for cost reduction.", ["supplier relationship management", "supply chain logistics", "cost reduction"]),
            ("Expertise in network administration, including configuring routers, switches, and firewalls.", ["network administration", "routers", "switches", "firewalls"]),
            ("Applied statistical analysis to interpret experimental data and formulate conclusions.", ["statistical analysis", "experimental data interpretation"]),
            ("Provided technical support to end-users, troubleshooting hardware and software issues.", ["technical support", "troubleshooting", "hardware", "software"]),
            ("Skilled in negotiation and client relationship management, fostering long-term partnerships.", ["negotiation", "client relationship management"]),
            ("Developed mobile applications for iOS using Swift and Xcode, ensuring responsive user experience.", ["mobile application development", "ios", "swift", "xcode"]),
            ("Proficient in Java, Python, and Data Structures. Experience in cloud platforms like AWS.", ["java", "python", "data structures", "aws"]),
            ("Worked on Machine Learning algorithms and Deep Learning frameworks using TensorFlow and PyTorch.", ["machine learning", "deep learning", "tensorflow", "pytorch"]),
            ("Experienced in frontend development with HTML5, CSS3, and React. Basic knowledge of Git.", ["html5", "css3", "react", "git"]),
            ("Led DevOps projects using Jenkins and Docker for automated CI/CD pipelines.", ["devops", "jenkins", "docker", "ci/cd"]),
            ("Managed databases using MySQL and PostgreSQL. Familiar with ORM tools like Hibernate.", ["mysql", "postgresql", "hibernate"]),
            ("Built mobile apps using Swift and React Native. Hands-on with Xcode.", ["swift", "react native", "xcode"]),
            ("Handled backend systems using Django and Flask. Implemented RESTful APIs.", ["django", "flask", "restful apis"]),
            ("Collaborated in Agile environments. Strong Communication and Time Management skills.", ["agile", "communication", "time management"]),
            ("Experience in cybersecurity with tools like Wireshark, Metasploit, and Kali Linux.", ["wireshark", "metasploit", "kali linux", "cybersecurity"]),
            ("Worked with SAP ERP modules and integrated financial reports using Excel and Tableau.", ["sap", "excel", "tableau"]),
            ("Expertise in Photoshop, Illustrator, and Adobe XD for UI/UX design.", ["photoshop", "illustrator", "adobe xd", "ui/ux design"]),
            ("Involved in academic research using MATLAB and LaTeX. Familiar with academic writing.", ["matlab", "latex", "academic writing"]),
            ("Conducted market research and email marketing using HubSpot and Google Analytics.", ["hubspot", "google analytics", "email marketing", "market research"]),
            ("Led cross-functional teams using Jira for project tracking and Scrum methodologies.", ["jira", "scrum", "project tracking"]),
            ("Created 3D models using AutoCAD and SolidWorks. Simulated stress tests.", ["autocad", "solidworks"]),
            ("No technical skills mentioned in this sentence.", []),
            ("The weather is sunny today.", []),
            ("I enjoy hiking and reading books.", []),
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
        for i, (text, _) in enumerate(gold_raw_data, 1): # Display first 5 for brevity
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
