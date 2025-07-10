import re
import ast
import spacy
from spacy.matcher import PhraseMatcher
from skills_data import skills_domain

def clean_html(text):
    return re.sub(r"<[^>]+>|\s+", " ", str(text)).strip()

def extract_relevant_sentences(text):
    if not text:
        return []
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(clean_html(text))
    keywords = {
        "experience", "project", "management", "responsibility", "work history", "work experience",
        "job description", "role", "tasks", "positions", "certification", "abilities",
        "technical skills", "summary", "profile", "accomplishments"
    }
    return [sent.text.strip() for sent in doc.sents if any(k in sent.text.lower() for k in keywords)]

def add_skills_to_sentences(sentences):
    if not sentences:
        return []
    temp_nlp = spacy.blank("en")
    matcher = PhraseMatcher(temp_nlp.vocab, attr="LOWER")
    skill_map = {}

    for skill in skills_domain:
        variants = [skill.lower()]
        if "." in skill:
            variants.append(skill.replace(".", "").lower())
        if any(c in skill for c in "/&+#"):
            cleaned = re.sub(r'[/&+#]', ' ', skill).lower()
            cleaned = re.sub(r'\s+', ' ', cleaned)
            variants.append(cleaned)
        for v in set(variants):
            matcher.add("SKILL", [temp_nlp.make_doc(v)])
            skill_map[v] = skill

    text = " ".join(sentences).lower()
    text = re.sub(r'[^\w\s.#+&/-]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    doc = temp_nlp.make_doc(text)

    matches = matcher(doc)
    return list({skill_map[doc[start:end].text.strip()] for _, start, end in matches})

def cleaned_text(raw):
    try:
        parsed = ast.literal_eval(raw)
        if isinstance(parsed, tuple) and len(parsed) == 2:
            s, skills = parsed
            s = " ".join([re.sub(r'\s+', ' ', x.strip()) for x in s]) if isinstance(s, list) else str(s)
            skills = [re.sub(r'\\', '', str(skill)) for skill in skills] if isinstance(skills, list) else []
            return (s, skills)
    except:
        pass
    return ("", [])

def create_training_example(text, skill_list):
    doc = spacy.blank("en").make_doc(text)
    text_lower = text.lower()
    entities = []
    for skill in skill_list:
        start = 0
        while True:
            pos = text_lower.find(skill.lower(), start)
            if pos == -1:
                break
            char_span = doc.char_span(pos, pos + len(skill), alignment_mode="expand")
            if char_span:
                entities.append((char_span.start_char, char_span.end_char, "SKILL"))
            start = pos + len(skill)
    return (text, {"entities": entities})

def filter_valid_entities(nlp_model, text, entities):
    """Validate and fix entity alignments"""
    doc = nlp_model.make_doc(text)
    valid_entities = []
    
    for start, end, label in entities:
        # Try different alignment modes
        span = doc.char_span(start, end, label=label, alignment_mode="contract")
        if span is not None:
            valid_entities.append((span.start_char, span.end_char, label))
            continue
            
        span = doc.char_span(start, end, label=label, alignment_mode="expand")
        if span is not None:
            valid_entities.append((span.start_char, span.end_char, label))
            continue
            
        # If still no alignment, try strict mode
        span = doc.char_span(start, end, label=label, alignment_mode="strict")
        if span is not None:
            valid_entities.append((span.start_char, span.end_char, label))
    
    return valid_entities

