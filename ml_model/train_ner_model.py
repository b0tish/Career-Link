import spacy
import re
from spacy.training import Example
from spacy.training import offsets_to_biluo_tags
import pandas as pd
import ast
import random
import os
import shutil
import datetime
from sklearn.model_selection import train_test_split
from skills_data import skills_domain

# Configuration constants
MODEL_PATH = "skill_ner_model"
DEFAULT_LEARNING_RATES = {"existing": 0.0005, "new": 0.0008}
TRAINING_PARAMS = {
    "existing": {"epochs": 25, "patience": 8},
    "new": {"epochs": 30, "patience": 10}
}

def backup_model(model_path, backup_suffix="backup"):
    """Create a timestamped backup of the existing model."""
    if not os.path.exists(model_path):
        print("No existing model found to backup.")
        return None
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = f"{model_path}_{backup_suffix}_{timestamp}"
    
    try:
        shutil.copytree(model_path, backup_path)
        print(f"Backup created at: {backup_path}")
        return backup_path
    except Exception as e:
        print(f"Failed to create backup: {e}")
        return None

def restore_model(backup_path, model_path):
    """Restore model from backup."""
    if not os.path.exists(backup_path):
        print("Backup path not found!")
        return
    
    if os.path.exists(model_path):
        shutil.rmtree(model_path)
    shutil.copytree(backup_path, model_path)
    print(f"Model restored from backup: {backup_path}")

def setup_model_and_optimizer(model_path):
    """Load or create model and setup optimizer."""
    if os.path.exists(model_path):
        print(f"Loading existing model from {model_path}...")
        nlp = spacy.load(model_path)
        is_existing = True
    else:
        print("Creating new model from 'en_core_web_md'...")
        try:
            nlp = spacy.load("en_core_web_md")
        except OSError:
            print("Downloading 'en_core_web_md' model...")
            spacy.cli.download("en_core_web_md")
            nlp = spacy.load("en_core_web_md")
        is_existing = False
    
    # Setup NER component
    if "ner" not in nlp.pipe_names:
        ner = nlp.add_pipe("ner", last=True)
    else:
        ner = nlp.get_pipe("ner")
    
    if "SKILL" not in ner.labels:
        ner.add_label("SKILL")
    
    # Setup optimizer
    optimizer = nlp.resume_training()
    learning_rate = DEFAULT_LEARNING_RATES["existing" if is_existing else "new"]
    
    # Try to set learning rate
    if hasattr(optimizer, 'learn_rate'):
        optimizer.learn_rate = learning_rate
    elif hasattr(optimizer, 'set_param'):
        try:
            optimizer.set_param('learn_rate', learning_rate)
        except Exception:
            pass
    
    print(f"Using {'LOWER' if is_existing else 'HIGHER'} learning rate: {learning_rate}")
    
    return nlp, optimizer, is_existing

def create_skill_pattern(skill):
    """Create regex pattern for skill matching."""
    skill_lower = skill.lower().strip()
    if not skill_lower:
        return None
    
    skill_words = skill_lower.split()
    
    if len(skill_words) == 1:
        # Single word - match case variations and word boundaries
        return rf'\b{re.escape(skill_lower)}\b'
    else:
        # Multi-word - allow flexible spacing and punctuation between words
        pattern_parts = []
        for i, word in enumerate(skill_words):
            if i == 0:
                # First word: allow word boundary and case variations
                pattern_parts.append(rf'\b{re.escape(word)}')
            else:
                # Subsequent words: allow flexible spacing/punctuation
                pattern_parts.append(rf'[\s\W]*{re.escape(word)}')
        
        # End with word boundary
        return ''.join(pattern_parts) + r'\b'

def find_skills_in_text(nlp_model, text, skill_list):
    """Find skills in text and return entities."""
    doc = nlp_model.make_doc(text)
    entities = []
    text_lower = text.lower()

    for skill in skill_list:
        skill_lower = skill.lower().strip()
        if not skill_lower:
            continue
            
        # Simple string search for now (more reliable than regex)
        if skill_lower in text_lower:
            start = text_lower.find(skill_lower)
            end = start + len(skill_lower)
            
            # Create span in original text
            span = doc.char_span(start, end, label="SKILL")
            if span is not None:
                entities.append((span.start_char, span.end_char, "SKILL"))
    
    # Merge overlapping entities
    return merge_overlapping_entities(entities)

def merge_overlapping_entities(entities):
    """Merge overlapping or adjacent entities."""
    if not entities:
        return []
    
    entities = sorted(list(set(entities)), key=lambda x: x[0])
    merged = [entities[0]]
    
    for ent in entities[1:]:
        last = merged[-1]
        if ent[0] <= last[1] + 1:  # overlap or adjacent
            merged[-1] = (last[0], max(last[1], ent[1]), last[2])
        else:
            merged.append(ent)
    
    return merged

def create_training_example(nlp_model, text, skill_list):
    """Create a spaCy training example."""
    entities = find_skills_in_text(nlp_model, text, skill_list)
    return Example.from_dict(nlp_model.make_doc(text), {"entities": entities})

def load_training_data(data_path):
    """Load and process training data from CSV."""
    if not os.path.exists(data_path):
        print(f"Error: Training data file not found at {data_path}")
        return []
    
    df = pd.read_csv(data_path, on_bad_lines='skip')
    processed_data = []
    skills_domain_set = {skill.lower() for skill in skills_domain}
    
    for _, row in df.iterrows():
        try:
            data_tuple = ast.literal_eval(row.iloc[0])
            if not (isinstance(data_tuple, tuple) and len(data_tuple) == 2 and 
                   isinstance(data_tuple[0], str) and isinstance(data_tuple[1], list)):
                continue
            
            text, skills = data_tuple
            if not text.strip():
                continue

            # Filter skills to only include those from our curated list
            cleaned_skills = [
                skill for skill in skills 
                if any(skill.lower().strip() == domain_skill.lower() for domain_skill in skills_domain_set)
            ]
            
            if cleaned_skills:  # Only add examples with valid skills
                processed_data.append((text, cleaned_skills))
            
        except (ValueError, SyntaxError, TypeError) as e:
            print(f"Skipping bad row: {e}")
    
    return processed_data

def validate_entities(nlp_model, text, entities):
    """Validate entity alignment using spaCy."""
    if not entities:
        return True
    
    # Simple validation: check if entities are within text bounds
    text_length = len(text)
    for start, end, label in entities:
        if start < 0 or end > text_length or start >= end:
            return False
    
    return True

def clean_training_data(nlp_model, data_list):
    """Clean and validate training data."""
    cleaned_data = []
    
    for text, skills in data_list:
        # Create example to get entities
        example = create_training_example(nlp_model, text, skills)
        entities = [(ent.start_char, ent.end_char, ent.label_) for ent in example.reference.ents]
        
        if entities and validate_entities(nlp_model, text, entities):
            cleaned_data.append((text, {"entities": entities}))
        elif not entities:
            # Add examples with no entities as negative examples
            cleaned_data.append((text, {"entities": []}))
    
    return cleaned_data

def train_epochs(nlp_model, train_data, val_data, optimizer, n_epochs, batch_size, patience, model_path):
    """Train the model for multiple epochs with early stopping."""
    best_f1 = 0.0
    epochs_without_improvement = 0
    
    print(f"Starting training for up to {n_epochs} epochs...")
    
    for epoch in range(n_epochs):
        print(f"\nStarting epoch {epoch+1}/{n_epochs}...")
        
        # Shuffle and train
        random.shuffle(train_data)
        losses = {}
        
        # Convert to spaCy examples and train in batches
        spacy_examples = []
        for text, data in train_data:
            example = Example.from_dict(nlp_model.make_doc(text), data)
            spacy_examples.append(example)
        
        random.shuffle(spacy_examples)
        
        # Train in batches
        batches = spacy.util.minibatch(spacy_examples, size=batch_size)
        for batch in batches:
            nlp_model.update(batch, drop=0.1, losses=losses, sgd=optimizer)

        # Evaluate on validation set
        val_examples = []
        for text, data in val_data:
            example = Example.from_dict(nlp_model.make_doc(text), data)
            val_examples.append(example)
        
        scores = nlp_model.evaluate(val_examples)

        current_f1 = scores.get("ents_f", 0.0) or 0.0
        current_precision = scores.get("ents_p", 0.0) or 0.0
        current_recall = scores.get("ents_r", 0.0) or 0.0

        print(f"Epoch {epoch+1} - Training Loss (NER): {losses.get('ner', 0):.4f}")
        print(f"Validation Metrics: Precision={current_precision:.4f}, Recall={current_recall:.4f}, F1-score={current_f1:.4f}")

        # Early stopping logic
        if current_f1 > best_f1:
            best_f1 = current_f1
            epochs_without_improvement = 0
            nlp_model.to_disk(model_path)
            print(f"Model improved and saved to: {model_path} with F1: {best_f1:.4f}")
        else:
            epochs_without_improvement += 1
            print(f"No improvement for {epochs_without_improvement} epochs (Best F1: {best_f1:.4f})")
            if epochs_without_improvement >= patience:
                print(f"Early stopping triggered after {epoch+1} epochs due to no improvement.")
                break
    
    return best_f1

def train_ner_model(data_path="raw_data.csv", model_output_path="skill_ner_model", random_seed=42):
    """Main function to train the spaCy NER model."""
    print("\n=== PHASE 2: NER MODEL TRAINING ===")

    random.seed(random_seed)
    backup_path = backup_model(model_output_path)

    # Setup model and optimizer
    nlp_model, optimizer, is_existing = setup_model_and_optimizer(model_output_path)

    # Load and process training data
    raw_data = load_training_data(data_path)
    
    if not raw_data:
        print("No valid training data found. Please run generate_data.py first.")
        if backup_path:
            restore_model(backup_path, model_output_path)
        return

    print(f"Total valid data entries loaded: {len(raw_data)}")

    # Clean training data
    cleaned_data = clean_training_data(nlp_model, raw_data)
    print(f"Total training examples with valid entities: {len(cleaned_data)}")

    if len(cleaned_data) < 10:
        print("WARNING: Very few training examples. Consider adding more data to raw_data.csv.")
        if backup_path:
            restore_model(backup_path, model_output_path)
        return

    # Split data for training
    train_data, val_data = train_test_split(cleaned_data, test_size=0.2, random_state=random_seed)
    print(f"Training examples: {len(train_data)}")
    print(f"Validation examples: {len(val_data)}")

    # Initialize new model if needed
    if not is_existing:
        print("Initializing new model...")
        nlp_model.begin_training([create_training_example(nlp_model, text, data) for text, data in cleaned_data])

    # Training parameters
    params = TRAINING_PARAMS["existing" if is_existing else "new"]
    batch_size = 16

    if is_existing:
        print(f"🛡️ Using conservative training: {params['epochs']} max epochs, patience={params['patience']}")

    # Train the model
    best_f1 = train_epochs(
        nlp_model, train_data, val_data, optimizer,
        params['epochs'], batch_size, params['patience'], model_output_path
    )

    print(f"\n🎉 Training completed successfully!")
    print(f"Best F1 Score: {best_f1:.4f}")
    print(f"Final model saved to: {model_output_path}")
        
    # Cleanup backup
    if backup_path and os.path.exists(backup_path):
        try:
            shutil.rmtree(backup_path)
            print("✅ Backup cleaned up after successful training")
        except Exception as e:
            print(f"⚠️ Error cleaning up backup: {e}")

if __name__ == "__main__":
    train_ner_model(data_path="./raw_data.csv")
