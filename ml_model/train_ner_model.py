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
from sklearn.model_selection import KFold

model_path = "skill_ner_model"

def create_model_backup(model_path, backup_suffix="backup"):
    if os.path.exists(model_path):
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = f"{model_path}_{backup_suffix}_{timestamp}"
        
        try:
            shutil.copytree(model_path, backup_path)
            print(f"Backup created at: {backup_path}")
            return backup_path
        except Exception as e:
            print(f"Failed to create backup: {e}")
            return None
    else:
        print("No existing model found to backup.")
        return None

def restore_from_backup(backup_path, model_path):
    if os.path.exists(backup_path):
        import shutil
        if os.path.exists(model_path):
            shutil.rmtree(model_path)
        shutil.copytree(backup_path, model_path)
        print(f"Model restored from backup: {backup_path}")
    else:
        print("Backup path not found!")

def get_optimizer_with_custom_lr(nlp_model, base_lr=0.0008, new_model_lr=0.001):
    if os.path.exists(model_path):
        learning_rate = base_lr
        print(f"Using LOWER learning rate: {learning_rate} (incremental training)")
    else:
        learning_rate = new_model_lr
        print(f"Using HIGHER learning rate: {learning_rate} (new model)")
    
    optimizer = nlp_model.resume_training()
    if hasattr(optimizer, 'learn_rate'):
        optimizer.learn_rate = learning_rate
    elif hasattr(optimizer, 'set_param'): 
        try:
            optimizer.set_param('learn_rate', learning_rate)
        except Exception:
            print("Note: Optimizer might not expose 'learn_rate' directly or setting failed.")
    else:
        print("Note: Optimizer might not expose 'learn_rate' directly, relying on internal schedules.")
            
    return optimizer, learning_rate

def create_training_example(nlp_model, text, skill_list):
    doc = nlp_model.make_doc(text)
    entities = []

    text_lower = text.lower()

    for skill in skill_list:
        skill_lower = skill.lower().strip()

        start_idx = 0
        while True:
            pos = text_lower.find(skill_lower, start_idx)
            if pos == -1:
                break

            pattern = r'\b' + re.escape(skill_lower) + r'\b'
            pattern = pattern.replace(r'\ ', r'\s+') 
            
            match = re.match(pattern, text_lower[pos:])
            
            if match:
                char_span = doc.char_span(pos, pos + len(skill_lower), alignment_mode="expand")
                if char_span is not None:
                    entities.append((char_span.start_char, char_span.end_char, "SKILL"))
            start_idx = pos + 1 # Move past the current match to find subsequent ones

    entities = sorted(list(set(entities)), key=lambda x: x[0])

    filtered_entities = []
    current_end = -1
    for start, end, label in entities:
        if start >= current_end: # No overlap, or starts after previous one ends
            filtered_entities.append((start, end, label))
            current_end = end
        elif end > current_end: # Overlaps, but current entity extends further
            # Replace the last added entity if the current one is strictly better (longer)
            last_start, last_end, last_label = filtered_entities[-1]
            if (end - start) > (last_end - last_start): # If current is longer
                filtered_entities[-1] = (last_start, end, label) # Extend the existing entry
            current_end = max(current_end, end) # Update current_end regardless

    return (text, {"entities": filtered_entities})

def filter_valid_entities(nlp_model, text, entities):
    """
    Validates and fixes entity alignments using spaCy's char_span with different alignment modes.
    This helps ensure that the generated entities correspond to actual tokens in the spaCy Doc.
    Prioritizes 'strict' then 'contract' then 'expand'.
    """
    doc = nlp_model.make_doc(text)
    valid_entities = []

    for start, end, label in entities:
        span = None
        # Try strict alignment first
        span = doc.char_span(start, end, label=label, alignment_mode="strict")
        if span:
            valid_entities.append((span.start_char, span.end_char, label))
            continue
        
        # If strict fails, try contract
        span = doc.char_span(start, end, label=label, alignment_mode="contract")
        if span:
            valid_entities.append((span.start_char, span.end_char, label))
            continue

        # If contract fails, try expand
        span = doc.char_span(start, end, label=label, alignment_mode="expand")
        if span:
            valid_entities.append((span.start_char, span.end_char, label))
            continue
            
        # print(f"Warning: Could not align entity '{text[start:end]}' ({start}-{end}) in text: '{text[:50]}...'")
            
    return valid_entities


def check_alignment(nlp_model, text, entities):
    """
    Checks if entities are properly aligned using spaCy's built-in offsets_to_biluo_tags.
    This helps identify problematic examples that might cause training errors.
    """
    doc = nlp_model.make_doc(text)
    try:
        # This function will raise an error if alignments are severely off
        tags = offsets_to_biluo_tags(doc, entities)
        # Count misaligned entities (marked with '-')
        misaligned_count = tags.count('-')
        # Count actual entities to know the proportion of misalignment
        total_entities = len([tag for tag in tags if tag != 'O'])

        if misaligned_count > 0:
            # print(f"Warning: {misaligned_count} out of {total_entities} entities misaligned in text: {text[:75]}...")
            return False
        return True
    except Exception as e:
        print(f"Error checking alignment for text '{text[:75]}...': {e}")
        return False

def train_ner_model(data_path="raw_data.csv", model_output_path="skill_ner_model", random_seed=42):
    """
    Main function to train the spaCy NER model.
    """
    print("\n=== PHASE 2: NER MODEL TRAINING ===")

    random.seed(random_seed)

    backup_path = create_model_backup(model_output_path)

    # Determine if loading an existing model or creating a new one
    if os.path.exists(model_output_path):
        print(f"Loading existing model for incremental training from {model_output_path}...")
        training_nlp = spacy.load(model_output_path)
        is_existing_model = True
    else:
        try:
            training_nlp = spacy.load("en_core_web_md") 
            print("Created new model based on 'en_core_web_md'.")
        except OSError:
            print("Downloading 'en_core_web_md' model... This may take a moment.")
            spacy.cli.download("en_core_web_md")
            training_nlp = spacy.load("en_core_web_md")
            print("Successfully downloaded and loaded 'en_core_web_md'.")
        is_existing_model = False

    # Add NER pipe if it doesn't exist, or get it if it does
    if "ner" not in training_nlp.pipe_names:
        ner = training_nlp.add_pipe("ner", last=True)
    else:
        ner = training_nlp.get_pipe("ner")

    # Add "SKILL" label to the NER pipe if not already present
    if "SKILL" not in ner.labels:
        ner.add_label("SKILL")

    # Load and process training data
    if not os.path.exists(data_path):
        print(f"Error: Training data file not found at {data_path}. Please run generate_data.py first.")
        if backup_path:
            restore_from_backup(backup_path, model_output_path)
        return
        
    df = pd.read_csv(data_path)

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

    print(f"Total valid data entries loaded: {len(processed_raw_data)}")

    # Create spaCy training examples
    Train_data = []
    for text, skills in processed_raw_data:
        example = create_training_example(training_nlp, text, skills)
        Train_data.append(example)

    cleaned_train_data = []
    for text, ann in Train_data:
        entities = ann.get("entities", [])
        filtered_ents = filter_valid_entities(training_nlp, text, entities)
        if filtered_ents:
            if check_alignment(training_nlp, text, filtered_ents):
                cleaned_train_data.append((text, {"entities": filtered_ents}))
            else:
                print(f"Skipping example due to remaining misalignment after filtering: {text[:75]}...")
        else:
            print(f"Skipping example with no valid entities: {text[:75]}...")
            pass # It's okay to skip examples with no entities if that's the desired outcome

    print(f"Total training examples with valid and aligned entities: {len(cleaned_train_data)}")

    if len(cleaned_train_data) < 10:
        print("WARNING: Very few training examples. Consider adding more data to raw_data.csv.")
        if backup_path:
            restore_from_backup(backup_path, model_output_path)
        return

    # Split data into training and validation sets
    train_examples, val_examples = train_test_split(
        cleaned_train_data, test_size=0.2, random_state=random_seed
    )
    print(f"Training examples: {len(train_examples)}")
    print(f"Validation examples: {len(val_examples)}")

    # Function to yield spaCy Example objects for initialization/training
    def get_spacy_examples(data_list):
        spacy_examples = []
        for text, annotations in data_list:
            doc = training_nlp.make_doc(text)
            example = Example.from_dict(doc, annotations)
            spacy_examples.append(example)
        return spacy_examples

    # Get optimizer with custom learning rate based on whether it's a new or existing model
    # Slightly refined learning rates for better convergence
    optimizer, LEARNING_RATE = get_optimizer_with_custom_lr(training_nlp, base_lr=0.0005, new_model_lr=0.0008)

    # Initialize the model if it's new
    if not is_existing_model:
        print("Initializing new model with data and optimizer...")
        training_nlp.begin_training(get_spacy_examples(cleaned_train_data)) # More appropriate for fine-tuning
    else:
        print("Continuing training with existing model and its optimizer state...")

    # Training parameters
    batch_size = 16 # Increased batch size
    n_epochs = 30 # Increased max epochs
    patience = 10 # Number of epochs to wait for improvement before early stopping
    best_f1 = 0.0
    epochs_without_improvement = 0

    # Adjust parameters for incremental training (more conservative)
    if is_existing_model:
        n_epochs = min(n_epochs, 25) # Max epochs for fine-tuning
        patience = max(patience, 8)  # More patience for subtle improvements
        print(f"🛡️ Using conservative training: {n_epochs} max epochs, patience={patience}")

    print(f"Starting training for up to {n_epochs} epochs...")
    for epoch in range(n_epochs):
        random.shuffle(train_examples) # Shuffle training data each epoch
        losses = {}
        print(f"\nStarting epoch {epoch+1}/{n_epochs}...")

        # Convert training examples to spaCy Example objects for the batching
        spacy_train_examples = get_spacy_examples(train_examples)
        random.shuffle(spacy_train_examples) # Shuffle again before batching

        # Process in minibatches
        batches = spacy.util.minibatch(spacy_train_examples, size=batch_size)
        
        for batch in batches:
            # Determine dropout rate
            dropout_rate = 0.1 # Consistent dropout for fine-tuning
            
            # Update the model with the current batch
            training_nlp.update(batch, drop=dropout_rate, losses=losses, sgd=optimizer)

        # Evaluate on validation set
        val_examples_spacy = get_spacy_examples(val_examples)
        scores = training_nlp.evaluate(val_examples_spacy)

        current_f1 = scores.get("ents_f", 0.0)
        current_precision = scores.get("ents_p", 0.0)
        current_recall = scores.get("ents_r", 0.0)

        print(f"Epoch {epoch+1} - Training Loss (NER): {losses.get('ner', 0):.4f}")
        print(f"Validation Metrics: Precision={current_precision:.4f}, Recall={current_recall:.4f}, F1-score={current_f1:.4f}")

        # Early stopping logic
        if current_f1 > best_f1:
            best_f1 = current_f1
            epochs_without_improvement = 0
            training_nlp.to_disk(model_output_path) # Save the best model
            print(f"Model improved and saved to: {model_output_path} with F1: {best_f1:.4f}")
        else:
            epochs_without_improvement += 1
            print(f"No improvement for {epochs_without_improvement} epochs (Best F1: {best_f1:.4f})")
            if epochs_without_improvement >= patience:
                print(f"Early stopping triggered after {epoch+1} epochs due to no improvement.")
                break

    print(f"\n🎉 Training completed successfully!")
    print(f"Best F1 Score: {best_f1:.4f}")
    print(f"Final model saved to: {model_output_path}")
        
    if backup_path and os.path.exists(backup_path):
        try:
            shutil.rmtree(backup_path)
            print("✅ Backup cleaned up after successful training")
        except Exception as e:
            print(f"⚠️ Error cleaning up backup: {e}")

if __name__ == "__main__":
    train_ner_model(data_path="raw_data.csv")
