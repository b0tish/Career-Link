import pandas as pd
import spacy
from spacy.training import Example
from spacy.training import offsets_to_biluo_tags
import re
import ast
import random
from sklearn.model_selection import train_test_split
from spacy.matcher import PhraseMatcher
from skills_data import skills_domain # Assuming skills_data.py exists and contains skills_domain
import os
import PyPDF2

# Initialize spacy model
nlp = spacy.load("en_core_web_sm")
model_path = "skill_ner_model"

# Define keywords for sentence extraction
keywords = [
    "experience", "project", "management", "responsibility",
    "work history", "work experience", "job description",
    "role", "tasks", "positions", "certification", "abilities",
    "technical skills", "summary", "profile", "accomplishments"
]

# Function to clean HTML content
def clean_html(text):
    return re.sub(r"<[^>]+>|\s+", " ", str(text)).strip()

# Function to extract text from PDF
def extract_text_from_pdf(pdf_path):
    text = ""
    try:
        with open(pdf_path, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            for page_num in range(len(reader.pages)):
                text += reader.pages[page_num].extract_text()
    except Exception as e:
        print(f"Error reading PDF {pdf_path}: {e}")
    return text

# Function to extract relevant sentences based on keywords
def extract_relevant_sentences(text):
    if pd.isna(text) or not text:
        return []
    text = clean_html(text)
    doc = nlp(text)
    keyword_set = set(keywords)
    return [
        sent.text.strip()
        for sent in doc.sents
        if any(keyword in sent.text.lower() for keyword in keyword_set)
    ]

def add_skills_to_sentences(sentences):
    if not sentences:
        return []
    
    temp_nlp = spacy.blank("en")
    matcher = PhraseMatcher(temp_nlp.vocab, attr="LOWER")
    
    skill_patterns = []
    skill_map = {}    

    for skill in skills_domain:
        normalized_skill = skill.lower().strip()
        
        pattern_doc = temp_nlp.make_doc(normalized_skill)
        skill_patterns.append(pattern_doc)
        skill_map[normalized_skill] = skill
        
        if "." in skill:
            variant = skill.replace(".", "").lower().strip()
            if variant != normalized_skill:
                variant_doc = temp_nlp.make_doc(variant)
                skill_patterns.append(variant_doc)
                skill_map[variant] = skill
        
        if any(char in skill for char in ["/", "&", "+", "#"]):
            clean_skill = re.sub(r'[/&+#]', ' ', skill).lower().strip()
            clean_skill = re.sub(r'\s+', ' ', clean_skill)
            if clean_skill != normalized_skill and clean_skill:
                clean_doc = temp_nlp.make_doc(clean_skill)
                skill_patterns.append(clean_doc)
                skill_map[clean_skill] = skill
    
    matcher.add("SKILL", skill_patterns)
    
    full_text = " ".join(sentences)
    
    processed_text = full_text.lower()
    processed_text = re.sub(r'[^\w\s.#+&/-]', ' ', processed_text)
    processed_text = re.sub(r'\s+', ' ', processed_text).strip()
    
    doc = temp_nlp.make_doc(processed_text)
    
    matches = matcher(doc)
    matched_skills = set()
    
    for match_id, start, end in matches:
        span = doc[start:end]
        matched_text = span.text.lower().strip()
        
        if matched_text in skill_map:
            matched_skills.add(skill_map[matched_text])
        else:
            for skill in skills_domain:
                if skill.lower() == matched_text:
                    matched_skills.add(skill)
                    break
    
    additional_matches = find_skills_with_regex(full_text, skills_domain)
    matched_skills.update(additional_matches)
    
    return list(matched_skills)

def find_skills_with_regex(text, skills_list):
    found_skills = set()
    text_lower = text.lower()
    
    for skill in skills_list:
        skill_lower = skill.lower()
        
        escaped_skill = re.escape(skill_lower)
        
        pattern = escaped_skill.replace(r'\ ', r'\s+')
        
        pattern = r'\b' + pattern + r'\b'
        
        try:
            if re.search(pattern, text_lower):
                found_skills.add(skill)
        except re.error:
            if skill_lower in text_lower:
                words = re.findall(r'\b\w+\b', text_lower)
                if skill_lower in words or any(skill_lower in word for word in words):
                    found_skills.add(skill)
    
    return found_skills

def cleaned_text(raw):
    try:
        parsed = ast.literal_eval(raw)
        if isinstance(parsed, tuple) and len(parsed) == 2:
            sentences = parsed[0]
            skills = parsed[1]
            
            if isinstance(sentences, list):
                cleaned_sentences = [re.sub(r'\s+', ' ', s.strip()) for s in sentences]
                full_text = " ".join(cleaned_sentences)
            else:
                full_text = re.sub(r'\s+', ' ', str(sentences).strip())
            
            if isinstance(skills, list):
                cleaned_skills = [re.sub(r'\\', '', str(s)) for s in skills]
            else:
                cleaned_skills = []
            
            return (full_text, cleaned_skills)
        else:
            return ("", [])
    except (ValueError, SyntaxError, TypeError) as e:
        print(f"Skipping bad row: {raw} — {e}")
        return ("", [])

# Uncomment and run these sections if you need to generate/process data
# if os.path.exists("./data/Resume/Resume/Resume.csv"):
#      df_to_process = pd.read_csv("./data/Resume/Resume/Resume.csv").iloc[501:801]
#      print(f"Processing {len(df_to_process)} CSV data entries.")
#      df_to_process["Relevant Sentences"] = df_to_process["Resume_html"].apply(extract_relevant_sentences)
#      df_to_process["matched_skills"] = df_to_process["Relevant Sentences"].apply(add_skills_to_sentences)
#      df_to_process["raw_data"] = list(zip(df_to_process["Relevant Sentences"], df_to_process["matched_skills"]))
#
#      df_to_process[["raw_data"]].to_csv("relevant.csv", mode="w", index=False, header=True)
#      print("CSV data processed and saved to relevant.csv")
# else:
#      print("Resume.csv not found, skipping CSV processing...")
#
# # Process PDF files and append to relevant.csv
# pdf_root_dir = "data/Resume/data/data/"
# first_pdf_write = False if os.path.exists("relevant.csv") else True
#
# if os.path.exists(pdf_root_dir):
#      print(f"Processing PDF files from {pdf_root_dir}...")
#
#      for root, dirs, files in os.walk(pdf_root_dir):
#          pdf_files_in_folder = sorted([f for f in files if f.endswith(".pdf")])
#
#          for file in pdf_files_in_folder:
#              pdf_path = os.path.join(root, file)
#              print(f"Processing PDF: {pdf_path}")
#              pdf_text = extract_text_from_pdf(pdf_path)
#
#              if pdf_text:
#                  relevant_sentences = extract_relevant_sentences(pdf_text)
#                  if relevant_sentences:
#                      matched_skills = add_skills_to_sentences(relevant_sentences)
#
#                      pdf_df = pd.DataFrame([{
#                          "raw_data": (relevant_sentences, matched_skills)
#                      }])
#
#                      pdf_df[["raw_data"]].to_csv(
#                          "relevant.csv",
#                          mode="a",
#                          index=False,
#                          header=first_pdf_write
#                      )
#                      first_pdf_write = False
#      print("PDF processing completed.")
# else:
#      print("PDF directory not found, skipping PDF processing...")

# if os.path.exists("relevant.csv"):
#      df = pd.read_csv("relevant.csv")
#      print(f"Cleaning {len(df)} rows of data...")
#      df["raw_data"] = df["raw_data"].apply(cleaned_text)
#      df.to_csv("raw_data.csv", index=False)
#      print("Raw data cleaned and saved to raw_data.csv")


def get_optimizer_with_custom_lr(nlp_model, base_lr=0.001, new_model_lr=0.01):
    """
    Get optimizer with custom learning rate for incremental training
    
    Args:
        nlp_model: spaCy model
        base_lr: Learning rate for existing model (lower for incremental training)
        new_model_lr: Learning rate for new model (higher for initial training)
    """
    if os.path.exists(model_path):
        # Use lower learning rate for existing model to prevent catastrophic forgetting
        learning_rate = base_lr
        print(f"Using LOWER learning rate: {learning_rate} (incremental training)")
    else:
        # Use higher learning rate for new model
        learning_rate = new_model_lr
        print(f"Using HIGHER learning rate: {learning_rate} (new model)")
    
    # Initialize the optimizer. For incremental training, it's important to
    # use nlp_model.resume_training() to continue training with the existing optimizer state.
    optimizer = nlp_model.resume_training()
    
    # Set custom learning rate on the optimizer directly if it has a learn_rate attribute.
    # This often depends on the specific optimizer implementation within Thinc (spaCy's ML library).
    if hasattr(optimizer, 'learn_rate'):
        optimizer.learn_rate = learning_rate
    # If using Thinc's Adam, you might need to adjust parameters within its config or properties.
    elif hasattr(optimizer, 'beta1') and hasattr(optimizer, 'beta2'): # Check for common optimizer attributes
        print("Note: Optimizer might not expose 'learn_rate' directly, relying on internal schedules.")
        # For more control, one would typically pass a config object to nlp.begin_training()
        # or recreate the optimizer with the desired learning rate from scratch if doing a full restart.
    
    return optimizer, learning_rate


def setup_custom_training_config(learning_rate=0.001):
    """
    Setup custom training configuration with specific learning rate
    This function primarily defines the config, but it's used when initializing a *new* trainer,
    not typically when resuming training from an existing model's optimizer.
    """
    from spacy.training import Config
    
    config_str = f"""
    [training.optimizer]
    @optimizers = "Adam.v1"
    learn_rate = {learning_rate}
    beta1 = 0.9
    beta2 = 0.999
    
    [training.optimizer.learn_rate]
    @schedules = "warmup_linear.v1"
    warmup_steps = 250
    total_steps = 20000
    initial_rate = {learning_rate}
    """
    
    config = Config().from_str(config_str)
    return config


def create_model_backup(model_path, backup_suffix="backup"):
    """Create a backup of the existing model"""
    if os.path.exists(model_path):
        import shutil
        import datetime
        
        # Create timestamped backup
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = f"{model_path}_{backup_suffix}_{timestamp}"
        
        try:
            shutil.copytree(model_path, backup_path)
            print(f"✅ Backup created at: {backup_path}")
            return backup_path
        except Exception as e:
            print(f"❌ Failed to create backup: {e}")
            return None
    else:
        print("No existing model found to backup.")
        return None

def restore_from_backup(backup_path, model_path):
    """Restore model from backup if something goes wrong"""
    if os.path.exists(backup_path):
        import shutil
        if os.path.exists(model_path):
            shutil.rmtree(model_path)
        shutil.copytree(backup_path, model_path)
        print(f"✅ Model restored from backup: {backup_path}")
    else:
        print("❌ Backup path not found!")


print("\n=== PHASE 2: NER MODEL TRAINING ===")

random_seed = 42
random.seed(random_seed)


backup_path = create_model_backup(model_path)

if os.path.exists(model_path):
    print("Loading existing model for incremental training...")
    training_nlp = spacy.load(model_path)
    is_existing_model = True
else:
    print("Creating new model...")
    training_nlp = spacy.blank("en")
    is_existing_model = False

if "ner" not in training_nlp.pipe_names:
    ner = training_nlp.add_pipe("ner", last=True)
else:
    ner = training_nlp.get_pipe("ner")

if "SKILL" not in ner.labels:
    ner.add_label("SKILL")


def create_training_example(text, skill_list):
    """Create training example with proper token alignment"""
    doc = training_nlp.make_doc(text)
    entities = []

    # Clean and normalize the text
    text_lower = text.lower()

    for skill in skill_list:
        skill_lower = skill.lower().strip()

        # Find all occurrences of the skill in the text
        start = 0
        while True:
            pos = text_lower.find(skill_lower, start)
            if pos == -1:
                break

            # Check word boundaries
            if pos > 0 and text_lower[pos-1].isalnum():
                start = pos + 1
                continue
            if pos + len(skill_lower) < len(text_lower) and text_lower[pos + len(skill_lower)].isalnum():
                start = pos + 1
                continue

            # Try to align with token boundaries
            char_span = doc.char_span(pos, pos + len(skill_lower), alignment_mode="expand")
            if char_span is not None:
                entities.append((char_span.start_char, char_span.end_char, "SKILL"))

            start = pos + len(skill_lower)

    # Remove duplicates and sort
    entities = sorted(list(set(entities)), key=lambda x: x[0])

    # Remove overlapping entities
    filtered_entities = []
    last_end = 0

    for start, end, label in entities:
        if start >= last_end:
            filtered_entities.append((start, end, label))
            last_end = end

    return (text, {"entities": filtered_entities})

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


def check_alignment(nlp_model, text, entities):
    """Check if entities are properly aligned using spaCy's built-in function"""
    doc = nlp_model.make_doc(text)
    try:
        tags = offsets_to_biluo_tags(doc, entities)
        # Count misaligned entities (marked with '-')
        misaligned_count = tags.count('-')
        total_entities = len([tag for tag in tags if tag != 'O'])

        if misaligned_count > 0:
            print(f"Warning: {misaligned_count} out of {total_entities} entities misaligned in text: {text[:50]}...")
            return False
        return True
    except Exception as e:
        print(f"Error checking alignment: {e}")
        return False


df = pd.read_csv("raw_data.csv")

processed_raw_data = []
for _, row in df.iterrows():
    try:
        data_tuple = ast.literal_eval(row.iloc[0])
        if isinstance(data_tuple, tuple) and len(data_tuple) == 2 and \
           isinstance(data_tuple[0], str) and isinstance(data_tuple[1], list):
            if data_tuple[0].strip() and data_tuple[1]:
                processed_raw_data.append(data_tuple)
        else:
            print(f"Skipping malformed row: {row.iloc[0]}")
    except (ValueError, SyntaxError, TypeError) as e:
        print(f"Skipping bad row due to literal_eval error: {row.iloc[0]} — {e}")

print(f"Total valid data entries: {len(processed_raw_data)}")

Train_data = []
for text, skills in processed_raw_data:
    example = create_training_example(text, skills)
    Train_data.append(example)

cleaned_train_data = []
for text, ann in Train_data:
    entities = ann.get("entities", [])
    filtered_ents = filter_valid_entities(training_nlp, text, entities)
    if filtered_ents:    
        cleaned_train_data.append((text, {"entities": filtered_ents}))

print(f"Total training examples with valid entities: {len(cleaned_train_data)}")

if len(cleaned_train_data) < 10:
    print("WARNING: Very few training examples. Consider adding more data.")
    exit()

train_examples, val_examples = train_test_split(
    cleaned_train_data, test_size=0.2, random_state=random_seed
)
print(f"Training examples: {len(train_examples)}")
print(f"Validation examples: {len(val_examples)}")

def get_examples():
    examples = []
    for text, annotations in cleaned_train_data:
        doc = training_nlp.make_doc(text)
        example = Example.from_dict(doc, annotations)
        examples.append(example)
    return examples


# LEARNING RATE CONFIGURATION
# The get_optimizer_with_custom_lr function handles the logic for choosing the LR.
# Call it once to get the optimizer and the determined learning rate.
optimizer, LEARNING_RATE = get_optimizer_with_custom_lr(training_nlp, base_lr=0.0009, new_model_lr=0.001)

# Initialize or resume training
if not is_existing_model:
    print("Initializing new model with data...")
    training_nlp.initialize(lambda: get_examples()) # Pass a callable function
    # The optimizer returned by get_optimizer_with_custom_lr already calls resume_training() internally.
    # No need to call resume_training() again here.
    # Ensure the optimizer variable is correctly assigned.
else:
    print("Resuming training from existing model...")
    # The optimizer is already obtained from get_optimizer_with_custom_lr, which calls resume_training().

try:
    if hasattr(optimizer, 'learn_rate'):
        # This line is redundant if get_optimizer_with_custom_lr already sets it, but good for verification.
        optimizer.learn_rate = LEARNING_RATE
        print(f"✅ Final optimizer learning rate explicitly set to: {LEARNING_RATE}")
    else:
        print("⚠️ Optimizer might not expose 'learn_rate' directly or it's handled internally by spaCy's training loop.")
except Exception as e:
    print(f"⚠️ Error setting learning rate after optimizer retrieval: {e}")

batch_size = 8
n_epochs = 30
patience = 15
best_f1 = 0.0
epochs_without_improvement = 0


if is_existing_model:
    # More conservative training for existing model
    n_epochs = min(n_epochs, 20)  # Fewer epochs
    patience = max(patience, 5)   # More patience
    print(f"🛡️ Using conservative training: {n_epochs} max epochs, patience={patience}")

print(f"Starting training for up to {n_epochs} epochs...")
for epoch in range(n_epochs):
    random.shuffle(train_examples)
    losses = {}
    print(f"\nStarting epoch {epoch+1}/{n_epochs}...")

    # Wrap training examples in spaCy's Example objects
    spacy_train_examples = []
    for text, annotations in train_examples:
        doc = training_nlp.make_doc(text)
        example = Example.from_dict(doc, annotations)
        spacy_train_examples.append(example)

    # Use spacy.util.minibatch for batching
    random.shuffle(spacy_train_examples)
    batches = spacy.util.minibatch(spacy_train_examples, size=batch_size)
    
    for batch in batches:
        # Update the model with the current batch
        # The 'sgd' argument expects the optimizer object
        dropout_rate = 0.05 if is_existing_model else 0.1
        training_nlp.update(batch, drop=dropout_rate, losses=losses, sgd=optimizer)

    val_examples_spacy = []
    for text, annotations in val_examples:
        doc = training_nlp.make_doc(text)
        example = Example.from_dict(doc, annotations)
        val_examples_spacy.append(example)

    scores = training_nlp.evaluate(val_examples_spacy)

    current_f1 = scores.get("ents_f", 0.0)
    current_precision = scores.get("ents_p", 0.0)
    current_recall = scores.get("ents_r", 0.0)

    print(f"Epoch {epoch+1} - Training Loss: {losses.get('ner', 0):.4f}")
    print(f"Validation Metrics: Precision={current_precision:.4f}, Recall={current_recall:.4f}, F1-score={current_f1:.4f}")

    if current_f1 > best_f1:
        best_f1 = current_f1
        epochs_without_improvement = 0
        training_nlp.to_disk(model_path)
        print(f"Model improved and saved to: {model_path} with F1: {best_f1:.4f}")
    else:
        epochs_without_improvement += 1
        print(f"No improvement for {epochs_without_improvement} epochs (Best F1: {best_f1:.4f})")
        if epochs_without_improvement >= patience:
            print(f"Early stopping triggered after {epoch+1} epochs due to no improvement.")
            break

print(f"\n🎉 Training completed successfully!")
print(f"Best F1 Score: {best_f1:.4f}")
print(f"Final model saved to: {model_path}")
    
# Clean up backup if training was successful
if backup_path and os.path.exists(backup_path):
    import shutil
    shutil.rmtree(backup_path)
    print("✅ Backup cleaned up after successful training")

    
    
if os.path.exists(model_path):
    trained_nlp = spacy.load(model_path)
    print(f"Trained model loaded from: {model_path}")

    # Test the model
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
    ]

    for i, text in enumerate(test_texts, 1):
        doc = trained_nlp(text)
        print(f"\nTest Case {i}:")
        print(f"Input: {text}")
        print("Extracted Skills:")
        found_skills = [ent.text for ent in doc.ents if ent.label_ == "SKILL"]
        if found_skills:
            for skill in found_skills:
                print(f" - {skill}")
        else:
            print(" - No skills found.")

    # Test the enhanced skill detection function separately
    print("\n=== ENHANCED SKILL DETECTION TEST ===")
    sample_text = "Summary Seeking a Planning Engineer position to utilize my skills and abilities in an industry that offers security and professional growth while being resourceful innovative and flexible. Highlights Packages : AutoCAD 2D & 3D, Primavera Complete (Web,Client, Progress Reporter, Team Member) M.S-Office, M.S-Dos, Digital Designing & Video Editing (Adobe-Photoshop, Page Maker, Illustrator, Corel-Draw, Adobe-Preimere, Ulead Video Studio, Macromedia Flash, Projects, Computer Fundamentals and Information Technology)."

    detected_skills = add_skills_to_sentences([sample_text])
    print(f"Skills detected by enhanced function: {detected_skills}")
    print(f"Number of skills found: {len(detected_skills)}")

else:
    print("❌ No trained model found. Training may have failed.")
