import spacy
from spacy.training import Example
from spacy.training import offsets_to_biluo_tags
import pandas as pd
import ast
import random
import os
import shutil
import datetime
from sklearn.model_selection import train_test_split

# Define model path
model_path = "skill_ner_model"

def create_model_backup(model_path, backup_suffix="backup"):
    """Create a backup of the existing model."""
    if os.path.exists(model_path):
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
    """Restore model from backup if something goes wrong."""
    if os.path.exists(backup_path):
        import shutil
        if os.path.exists(model_path):
            shutil.rmtree(model_path)
        shutil.copytree(backup_path, model_path)
        print(f"✅ Model restored from backup: {backup_path}")
    else:
        print("❌ Backup path not found!")

def get_optimizer_with_custom_lr(nlp_model, base_lr=0.0001, new_model_lr=0.001):
    """
    Get optimizer with custom learning rate for incremental training.
    
    Args:
        nlp_model: spaCy model (either new blank model or loaded existing model).
        base_lr: Learning rate for existing model (lower for incremental training).
        new_model_lr: Learning rate for new model (higher for initial training).
    """
    if os.path.exists(model_path):
        learning_rate = base_lr
        print(f"Using LOWER learning rate: {learning_rate} (incremental training)")
    else:
        learning_rate = new_model_lr
        print(f"Using HIGHER learning rate: {learning_rate} (new model)")
    
    # Initialize or resume the optimizer based on the model state
    optimizer = nlp_model.resume_training()
    
    # Attempt to set the learning rate if the optimizer exposes such an attribute
    # Note: spaCy's internal optimizer handling can be complex. For precise control,
    # it's often better to configure the optimizer when starting a new pipeline.
    # However, for fine-tuning, `resume_training` is the correct approach.
    if hasattr(optimizer, 'learn_rate'):
        optimizer.learn_rate = learning_rate
    elif hasattr(optimizer, 'set_param'): # Some optimizers might have a set_param method
        try:
            optimizer.set_param('learn_rate', learning_rate)
        except Exception:
            print("Note: Optimizer might not expose 'learn_rate' directly or setting failed.")
    else:
        print("Note: Optimizer might not expose 'learn_rate' directly, relying on internal schedules.")
            
    return optimizer, learning_rate

def create_training_example(nlp_model, text, skill_list):
    """
    Create training example with proper token alignment.
    This function processes text and a list of skills to generate
    spaCy-compatible entity annotations (start_char, end_char, label).
    It includes logic to handle word boundaries and prevent overlapping entities.
    """
    doc = nlp_model.make_doc(text)
    entities = []

    text_lower = text.lower()

    for skill in skill_list:
        skill_lower = skill.lower().strip()

        # Find all occurrences of the skill in the text
        start = 0
        while True:
            pos = text_lower.find(skill_lower, start)
            if pos == -1:
                break

            # Basic word boundary check to avoid partial matches
            # e.g., "skills" containing "skill"
            if pos > 0 and text_lower[pos-1].isalnum():
                start = pos + 1
                continue
            if pos + len(skill_lower) < len(text_lower) and text_lower[pos + len(skill_lower)].isalnum():
                start = pos + 1
                continue

            # Align with token boundaries using char_span
            # "expand" mode is generally robust for matching spans
            char_span = doc.char_span(pos, pos + len(skill_lower), alignment_mode="expand")
            if char_span is not None:
                entities.append((char_span.start_char, char_span.end_char, "SKILL"))

            start = pos + len(skill_lower)

    # Remove duplicates and sort by start character
    entities = sorted(list(set(entities)), key=lambda x: x[0])

    # Remove overlapping entities to ensure validity for spaCy
    filtered_entities = []
    last_end = 0
    for start, end, label in entities:
        if start >= last_end: # If the current entity does not overlap with the previous one
            filtered_entities.append((start, end, label))
            last_end = end
        elif end > last_end: # If it overlaps but extends further, take the longer one (simple heuristic)
            filtered_entities[-1] = (filtered_entities[-1][0], end, label)
            last_end = end

    return (text, {"entities": filtered_entities})

def filter_valid_entities(nlp_model, text, entities):
    """
    Validates and fixes entity alignments using spaCy's char_span with different alignment modes.
    This helps ensure that the generated entities correspond to actual tokens in the spaCy Doc.
    """
    doc = nlp_model.make_doc(text)
    valid_entities = []

    for start, end, label in entities:
        # Try different alignment modes for robustness
        span = doc.char_span(start, end, label=label, alignment_mode="contract")
        if span is not None:
            valid_entities.append((span.start_char, span.end_char, label))
            continue

        span = doc.char_span(start, end, label=label, alignment_mode="expand")
        if span is not None:
            valid_entities.append((span.start_char, span.end_char, label))
            continue

        span = doc.char_span(start, end, label=label, alignment_mode="strict")
        if span is not None:
            valid_entities.append((span.start_char, span.end_char, label))
            
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
            print(f"Warning: {misaligned_count} out of {total_entities} entities misaligned in text: {text[:75]}...")
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
        print("Creating new blank spaCy model...")
        training_nlp = spacy.blank("en")
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
            # Ensure the row content is correctly interpreted as a tuple
            data_tuple = ast.literal_eval(row.iloc[0])
            if isinstance(data_tuple, tuple) and len(data_tuple) == 2 and \
               isinstance(data_tuple[0], str) and isinstance(data_tuple[1], list):
                if data_tuple[0].strip() and data_tuple[1]: # Only include if text and skills are present
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

    # Filter for examples with properly aligned entities
    cleaned_train_data = []
    for text, ann in Train_data:
        entities = ann.get("entities", [])
        filtered_ents = filter_valid_entities(training_nlp, text, entities)
        # Only add examples if they have at least one valid entity
        if filtered_ents:
            # Re-check alignment after filtering to be extra sure
            if check_alignment(training_nlp, text, filtered_ents):
                cleaned_train_data.append((text, {"entities": filtered_ents}))
            else:
                print(f"Skipping example due to remaining misalignment after filtering: {text[:75]}...")
        else:
            # print(f"Skipping example with no valid entities: {text[:75]}...")
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
    optimizer, LEARNING_RATE = get_optimizer_with_custom_lr(training_nlp, base_lr=0.0009, new_model_lr=0.001)

    # Initialize the model if it's new
    if not is_existing_model:
        print("Initializing new model with data and optimizer...")
        training_nlp.initialize(lambda: get_spacy_examples(cleaned_train_data), sgd=optimizer)
        # Note: initialize sets up the optimizer. If using `resume_training`, it's already linked.
    else:
        print("Continuing training with existing model and its optimizer state...")
        # The optimizer is already obtained from get_optimizer_with_custom_lr, which calls resume_training().

    # Training parameters
    batch_size = 8
    n_epochs = 30
    patience = 15 # Number of epochs to wait for improvement before early stopping
    best_f1 = 0.0
    epochs_without_improvement = 0

    # Adjust parameters for incremental training (more conservative)
    if is_existing_model:
        n_epochs = min(n_epochs, 20) # Max epochs for fine-tuning
        patience = max(patience, 7)  # More patience for subtle improvements
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
            dropout_rate = 0.05 if is_existing_model else 0.1 # Lower dropout for incremental training
            
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
        
    # Clean up backup if training was successful
    if backup_path and os.path.exists(backup_path):
        try:
            shutil.rmtree(backup_path)
            print("✅ Backup cleaned up after successful training")
        except Exception as e:
            print(f"⚠️ Error cleaning up backup: {e}")

if __name__ == "__main__":
    # Ensure `raw_data.csv` exists before training
    # You might want to call `generate_data.py` first if raw_data.csv doesn't exist
    # from generate_data import generate_and_process_data
    # data_file = generate_and_process_data() # This will generate raw_data.csv
    # if data_file:
    #     train_ner_model(data_path=data_file)
    # else:
    #     print("Training aborted: No data file generated.")
    
    # For now, assuming raw_data.csv is already generated
    train_ner_model(data_path="raw_data.csv")
