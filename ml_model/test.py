import random
import pandas as pd

# Templates for generating synthetic training data
templates = {
    "experience": [
        "I have {years} years of experience in {skill}.",
        "Experienced in {skill} with {years} years of hands-on work.",
        "Skilled in {skill} with extensive experience.",
        "Proficient in {skill} having worked with it for {years} years."
    ],
    "projects": [
        "Developed projects using {skill} and {other_skill}.",
        "Built applications with {skill}, {other_skill}, and {third_skill}.",
        "Created solutions using {skill} for data processing.",
        "Implemented systems using {skill} and modern frameworks."
    ],
    "skills_list": [
        "My technical skills include {skill}, {other_skill}, and {third_skill}.",
        "Proficient in {skill}, {other_skill}, and various other technologies.",
        "Technical expertise: {skill}, {other_skill}, {third_skill}.",
        "Skills: {skill} • {other_skill} • {third_skill}"
    ],
    "job_description": [
        "Responsible for {skill} development and {other_skill} implementation.",
        "Worked with {skill} to build scalable applications.",
        "Used {skill} and {other_skill} to solve complex problems.",
        "Managed projects involving {skill}, {other_skill}, and {third_skill}."
    ]
}

# Priority skills that need more training data
priority_skills = [
    "python", "java", "c++", "javascript", "sql", "aws", "excel", "react",
    "machine learning", "docker", "kubernetes", "git", "linux", "angular",
    "node.js", "mongodb", "postgresql", "tensorflow", "pandas", "numpy"
]

def generate_synthetic_data(skills_list, num_examples=100):
    """
    Generate synthetic training examples for underrepresented skills
    """
    synthetic_data = []
    years = ["2", "3", "5", "7", "10", "over 5"]
    
    for _ in range(num_examples):
        # Choose random template category and template
        category = random.choice(list(templates.keys()))
        template = random.choice(templates[category])
        
        # Choose skills
        primary_skill = random.choice(skills_list)
        other_skills = random.sample([s for s in skills_list if s != primary_skill], 
                                   min(2, len(skills_list) - 1))
        
        # Fill template
        if "{years}" in template:
            text = template.format(
                skill=primary_skill,
                other_skill=other_skills[0] if other_skills else "programming",
                third_skill=other_skills[1] if len(other_skills) > 1 else "development",
                years=random.choice(years)
            )
        else:
            text = template.format(
                skill=primary_skill,
                other_skill=other_skills[0] if other_skills else "programming",
                third_skill=other_skills[1] if len(other_skills) > 1 else "development"
            )
        
        # Create skills list (skills that appear in the text)
        text_lower = text.lower()
        found_skills = []
        for skill in [primary_skill] + other_skills:
            if skill.lower() in text_lower:
                found_skills.append(skill)
        
        if found_skills:  # Only add if skills were found
            synthetic_data.append((text, found_skills))
    
    return synthetic_data

def augment_existing_data(existing_data_file="raw_data.csv", output_file="augmented_data.csv"):
    """
    Augment existing training data with synthetic examples
    """
    # Load existing data
    df = pd.read_csv(existing_data_file)
    
    # Generate synthetic data
    synthetic_data = generate_synthetic_data(priority_skills, num_examples=200)
    
    # Convert to same format as existing data
    synthetic_df = pd.DataFrame([
        {"raw_data": str((text, skills))} for text, skills in synthetic_data
    ])
    
    # Combine with existing data
    combined_df = pd.concat([df, synthetic_df], ignore_index=True)
    
    # Save augmented data
    combined_df.to_csv(output_file, index=False)
    
    print(f"Original data: {len(df)} examples")
    print(f"Synthetic data: {len(synthetic_df)} examples")
    print(f"Combined data: {len(combined_df)} examples")
    print(f"Augmented data saved to: {output_file}")
    
    return output_file

# Example usage:
if __name__ == "__main__":
    # Generate some sample synthetic data
    sample_data = generate_synthetic_data(priority_skills[:10], num_examples=10)
    
    print("Sample synthetic training data:")
    for i, (text, skills) in enumerate(sample_data[:5], 1):
        print(f"{i}. Text: {text}")
        print(f"   Skills: {skills}")
        print()
    
    # Uncomment to augment your actual data:
    # augment_existing_data()
