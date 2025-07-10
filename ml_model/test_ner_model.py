import spacy
import os

def test_ner_model(model_path="skill_ner_model"):
    print("\n=== PHASE 3: MODEL TESTING ===")
    if os.path.exists(model_path):
        trained_nlp = spacy.load(model_path)
        print(f"Trained model loaded from: {model_path}")
        test_texts = [
            "I have experience in data analysis, project management, and communication.",
            "My skills include Python, machine learning, and customer service.",
            "Expertise in statistics, marketing strategy, and client relations.",
            "Worked on inventory management using Excel and SQL.",
            "Looking for a position where I can use my skills in accounting and finance.",
            "Managed a team in software development and agile methodologies.",
            "Proficient in C++, Java, and object-oriented programming.",
            "Utilized AWS for cloud deployment and network administration.",
            "Developed strategies for digital marketing and content creation.",
            "Summary Seeking a Planning Engineer position to utilize my skills and abilities in an industry that offers security and professional growth while being resourceful innovative and flexible. Highlights Packages : AutoCAD 2D & 3D, Primavera Complete (Web,Client, Progress Reporter, Team Member) M.S-Office, M.S-Dos, Digital Designing & Video Editing (Adobe-Photoshop, Page Maker, Illustrator, Corel-Draw, Adobe-Preimere, Ulead Video Studio, Macromedia Flash, Projects, Computer Fundamentals and Information Technology)."
            "Skilled in JavaScript, HTML5, CSS3, and responsive web design.",
            "Strong background in cloud computing, specifically Azure and Google Cloud Platform.",
            "Implemented ERP systems and CRM software for various clients.",
            "Successfully led teams in software testing and quality assurance.",
            "Experience with database management using PostgreSQL and MongoDB.",
            "Developed robust APIs using Node.js and Express.js.",
            "Adept at financial modeling, budgeting, and forecasting.",
            "Managed large-scale data migration projects using ETL tools.",
            "Proficient in cybersecurity protocols and network security.",
            "Expert in technical writing, documentation, and user manuals.",
            "My responsibilities included Scrum Master duties and Kanban implementation.",
            "Hands-on experience with Docker, Kubernetes, and continuous integration.",
            "Performed risk assessment and compliance auditing.",
            "Extensive knowledge of supply chain management and logistics.",
            "Conducted market research and competitive analysis.",
            "Utilized Photoshop and Illustrator for graphic design projects.",
            "Familiar with various operating systems, including Linux and Windows Server.",
            "Designed and deployed machine learning models in production.",
            "Excellent problem-solving skills and critical thinking.",
            "Experience in sales, negotiation, and contract management.",
            "Strong command over various data structures and algorithms.",
            "Administered SQL Server databases and optimized queries.",
            "Implemented secure coding practices and conducted code reviews.",
            "Expert in business intelligence and data visualization tools like Tableau.",
            "Contributed to open-source projects using Git and GitHub.",
            "Facilitated workshops on agile transformation and product ownership.",
            "Designed user interfaces (UI) and user experiences (UX) using Figma.",
            "Authored research papers on natural language processing.",
            "Proficient in using JIRA for project tracking.",
            "Successfully managed client relationships and stakeholder communication.",
            "Adept at statistical analysis and experimental design.",
            "Experience with virtualization technologies like VMware and VirtualBox.",
            "Developed mobile applications for Android and iOS platforms.",
            "Provided technical support and troubleshooting for enterprise software.",
            "Strong analytical skills with a focus on data-driven decision making.",
            "Managed social media campaigns and content marketing strategies.",
            "Familiar with GDPR compliance and data privacy regulations.",
            "Conducted performance tuning for web applications.",
            "Implemented CI/CD pipelines using Jenkins and GitLab CI.",
            "My core competencies include strategic planning and organizational development."
        ]
        print("\n--- NER Model Predictions ---")
        for i, text in enumerate(test_texts, 1):
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
if __name__ == "__main__":
    test_ner_model()
