import spacy
import os
from spacy.training import Example
from spacy.scorer import Scorer
import re
from generate_csv_data import add_skills_to_sentences

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
            ("I have experience in data analysis, project management, and communication.", ["data analysis", "project management", "communication"]),
            ("My skills include Python, machine learning, and customer service.", ["Python", "machine learning", "customer service"]),
            ("Expertise in statistics, marketing strategy, and client relations.", ["statistics", "marketing strategy", "client relations"]),
            ("Worked on inventory management using Excel and SQL.", ["inventory management", "Excel", "SQL"]),
            ("Looking for a position where I can use my skills in accounting and finance.", ["accounting", "finance"]),
            ("Managed a team in software development and agile methodologies.", ["software development", "agile methodologies"]),
            ("Proficient in C++, Java, and object-oriented programming.", ["C++", "Java", "object-oriented programming"]),
            ("Utilized AWS for cloud deployment and network administration.", ["AWS", "cloud deployment", "network administration"]),
            ("Developed strategies for digital marketing and content creation.", ["digital marketing", "content creation"]),
            ("Summary Seeking a Planning Engineer position to utilize my skills and abilities in an industry that offers security and professional growth while being resourceful innovative and flexible. Highlights Packages : AutoCAD 2D & 3D, Primavera Complete (Web,Client, Progress Reporter, Team Member) M.S-Office, M.S-Dos, Digital Designing & Video Editing (Adobe-Photoshop, Page Maker, Illustrator, Corel-Draw, Adobe-Preimere, Ulead Video Studio, Macromedia Flash, Projects, Computer Fundamentals and Information Technology).", ["Planning Engineer", "AutoCAD 2D & 3D", "Primavera Complete", "M.S-Office", "M.S-Dos", "Digital Designing", "Video Editing", "Adobe-Photoshop", "Page Maker", "Illustrator", "Corel-Draw", "Adobe-Preimere", "Ulead Video Studio", "Macromedia Flash", "Computer Fundamentals", "Information Technology"]),
            ("Skilled in JavaScript, HTML5, CSS3, and responsive web design.", ["JavaScript", "HTML5", "CSS3", "responsive web design"]),
            ("Strong background in cloud computing, specifically Azure and Google Cloud Platform.", ["cloud computing", "Azure", "Google Cloud Platform"]),
            ("Implemented ERP systems and CRM software for various clients.", ["ERP systems", "CRM software"]),
            ("Successfully led teams in software testing and quality assurance.", ["software testing", "quality assurance"]),
            ("Experience with database management using PostgreSQL and MongoDB.", ["database management", "PostgreSQL", "MongoDB"]),
            ("Developed robust APIs using Node.js and Express.js.", ["APIs", "Node.js", "Express.js"]),
            ("Adept at financial modeling, budgeting, and forecasting.", ["financial modeling", "budgeting", "forecasting"]),
            ("Managed large-scale data migration projects using ETL tools.", ["data migration", "ETL tools"]),
            ("Proficient in cybersecurity protocols and network security.", ["cybersecurity protocols", "network security"]),
            ("Expert in technical writing, documentation, and user manuals.", ["technical writing", "documentation", "user manuals"]),
            ("My responsibilities included Scrum Master duties and Kanban implementation.", ["Scrum Master", "Kanban implementation"]),
            ("Hands-on experience with Docker, Kubernetes, and continuous integration.", ["Docker", "Kubernetes", "continuous integration"]),
            ("Performed risk assessment and compliance auditing.", ["risk assessment", "compliance auditing"]),
            ("Extensive knowledge of supply chain management and logistics.", ["supply chain management", "logistics"]),
            ("Conducted market research and competitive analysis.", ["market research", "competitive analysis"]),
            ("Utilized Photoshop and Illustrator for graphic design projects.", ["Photoshop", "Illustrator", "graphic design"]),
            ("Familiar with various operating systems, including Linux and Windows Server.", ["operating systems", "Linux", "Windows Server"]),
            ("Designed and deployed machine learning models in production.", ["machine learning models"]),
            ("Excellent problem-solving skills and critical thinking.", ["problem-solving skills", "critical thinking"]),
            ("Experience in sales, negotiation, and contract management.", ["sales", "negotiation", "contract management"]),
            ("Strong command over various data structures and algorithms.", ["data structures", "algorithms"]),
            ("Administered SQL Server databases and optimized queries.", ["SQL Server", "optimized queries"]),
            ("Implemented secure coding practices and conducted code reviews.", ["secure coding practices", "code reviews"]),
            ("Expert in business intelligence and data visualization tools like Tableau.", ["business intelligence", "data visualization tools", "Tableau"]),
            ("Contributed to open-source projects using Git and GitHub.", ["open-source projects", "Git", "GitHub"]),
            ("Facilitated workshops on agile transformation and product ownership.", ["agile transformation", "product ownership"]),
            ("Designed user interfaces (UI) and user experiences (UX) using Figma.", ["user interfaces", "UI", "user experiences", "UX", "Figma"]),
            ("Authored research papers on natural language processing.", ["research papers", "natural language processing"]),
            ("Proficient in using JIRA for project tracking.", ["JIRA", "project tracking"]),
            ("Successfully managed client relationships and stakeholder communication.", ["client relationships", "stakeholder communication"]),
            ("Adept at statistical analysis and experimental design.", ["statistical analysis", "experimental design"]),
            ("Experience with virtualization technologies like VMware and VirtualBox.", ["virtualization technologies", "VMware", "VirtualBox"]),
            ("Developed mobile applications for Android and iOS platforms.", ["mobile applications", "Android", "iOS platforms"]),
            ("Provided technical support and troubleshooting for enterprise software.", ["technical support", "troubleshooting"]),
            ("Strong analytical skills with a focus on data-driven decision making.", ["analytical skills", "data-driven decision making"]),
            ("Managed social media campaigns and content marketing strategies.", ["social media campaigns", "content marketing strategies"]),
            ("Familiar with GDPR compliance and data privacy regulations.", ["GDPR compliance", "data privacy regulations"]),
            ("Conducted performance tuning for web applications.", ["performance tuning"]),
            ("Implemented CI/CD pipelines using Jenkins and GitLab CI.", ["CI/CD pipelines", "Jenkins", "GitLab CI"]),
            ("My core competencies include strategic planning and organizational development.", ["strategic planning", "organizational development"])
        ]

        # Prepare examples for the scorer using explicit reference and predicted docs
        examples = []
        for text, gold_skills in gold_raw_data:
            # Create the reference Doc (gold standard)
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
        for i, (text, _) in enumerate(gold_raw_data[:10], 1): # Display first 5 for brevity
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
        print("\n--- ENHANCED SKILL DETECTION (using generate_data.py function) ---")
        sample_text_for_enhanced_detection = "My professional profile includes expertise in Advanced Python, Java Enterprise Edition, and intricate Data Analysis techniques. I've also managed cross-functional teams in Project Management and implemented Machine Learning solutions for clients. Furthermore, I am proficient in SQL Server, AWS Cloud services, and have strong Communication skills. My experience extends to Digital Marketing, Content Creation, and utilizing Adobe Photoshop for graphic design. I've also worked with Primavera Complete for project scheduling."
        
        detected_skills = add_skills_to_sentences([sample_text_for_enhanced_detection])
        print(f"Input: {sample_text_for_enhanced_detection[:200]}...")
        print(f"Skills detected by enhanced function: {detected_skills}")
        print(f"Number of skills found: {len(detected_skills)}")

    else:
        print(f"❌ No trained model found at {model_path}. Please train the model first.")

if __name__ == "__main__":
    test_ner_model()
