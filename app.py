from flask import Flask, request, render_template, redirect, url_for, send_from_directory
from transformers import pipeline as transformers_pipeline
import pickle
from docx import Document
from charset_normalizer import detect
import os
import re
import spacy

app = Flask(__name__)

# Directory to save uploaded resumes
UPLOAD_FOLDER = 'uploaded_resumes'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Initialize SpaCy NER model
nlp = spacy.load("en_core_web_sm")

# Load the QA pipeline
qa_pipeline = transformers_pipeline("question-answering", model="distilbert-base-uncased-distilled-squad")

# Load pre-trained classification pipeline and label encoder
try:
    with open("pipeline.pkl", "rb") as model_pipeline_file:
        model_pipeline = pickle.load(model_pipeline_file)

    with open("le.pkl", "rb") as le_file:
        label_encoder = pickle.load(le_file)
except Exception as e:
    print(f"Error loading models: {e}")

# Helper function for QA extraction
def extract_with_qa(question, context):
    try:
        answer = qa_pipeline(question=question, context=context)
        return answer["answer"] if answer["score"] > 0.5 else "Not Found"
    except Exception as e:
        return "Not Found"

# Parsing Functions
def parse_resume(text):
    return {
        "Name": extract_name(text),
        "Email": extract_email(text),
        "Phone": extract_phone(text),
        "Skills": extract_skills(text),
        "Education": extract_education(text),
        "Experience": extract_experience(text),
        "Certifications": extract_certifications(text),
    }

# 1. Extract Name
def extract_name(text):
    """
    Extracts the name of the candidate from the resume text.
    Combines QA, SpaCy NER, Regex, and context-based filtering for better accuracy.
    """
    # Step 1: Use QA for extracting name
    qa_extracted_name = extract_with_qa("What is the candidate's name?", text)
    if qa_extracted_name != "Not Found" and validate_name(qa_extracted_name):
        return qa_extracted_name

    # Step 2: Use SpaCy NER for extraction
    doc = nlp(text)
    for ent in doc.ents:
        if ent.label_ == "PERSON" and validate_name(ent.text):
            return ent.text.strip()

    # Step 3: Use context-based prioritization (focus on top 200 characters of text)
    lines = text.splitlines()[:10]  # Get the first 10 lines of the resume
    for line in lines:
        potential_name = extract_name_regex(line)
        if potential_name:
            return potential_name

    # Step 4: Regex as a fallback
    fallback_name = extract_name_regex(text)
    if fallback_name:
        return fallback_name

    # Default return if no valid name is found
    return "Not Found"


def extract_name_regex(text):
    """
    Regex-based extraction of name-like patterns.
    """
    name_pattern = r"\b[A-Z][a-z]+(?:\s[A-Z][a-z]+(?:\s[A-Z][a-z]+)?)?\b"  # Match up to three proper words
    matches = re.findall(name_pattern, text)
    for match in matches:
        if validate_name(match):
            return match
    return None


def validate_name(name):
    """
    Validates a name based on common garbage words and formatting.
    """
    garbage_words = {
        "Developer", "Engineer", "Manager", "Experience", "Education", "Certifications",
        "Skills", "Technologies", "Objective", "Contact", "Summary", "Project", "Team",
        "Maintenance", "Support", "Responsibilities", "Achievements", "Proficiency", "Expertise"
    }
    words = name.split()
    return (
        1 <= len(words) <= 3  # Allow one to three words
        and not any(word in garbage_words for word in words)  # Avoid garbage words
        and all(word[0].isupper() and word[1:].islower() for word in words)  # Proper capitalization
        and all(word.isalpha() for word in words)  # Only alphabetic characters
    )

# 2. Extract Email
def extract_email(text):
    email_pattern = r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}"
    match = re.search(email_pattern, text)
    return match.group() if match else "Not Found"

# 3. Extract Phone
def extract_phone(text):
    phone_pattern = r"\+?\d{1,4}[-.\s]?\(?\d{1,3}\)?[-.\s]?\d{1,4}[-.\s]?\d{1,4}[-.\s]?\d{1,9}"
    matches = re.findall(phone_pattern, text)
    for match in matches:
        if len(match) >= 10:
            return match
    return "Not Found"

# 4. Extract Skills
def extract_skills(text):
    skillsets = {
        "PeopleSoft": ["PeopleSoft", "HCM", "Financials", "Payroll", "PeopleCode", "SQR", "Integration Broker"],
        "SQL Developer": ["SQL", "T-SQL", "SSIS", "ETL", "Database Administration", "Query Optimization"],
        "Workday": ["Workday", "Core HCM", "Payroll", "Compensation", "Workday Studio", "EIB"],
        "React JS Developer": ["React", "JavaScript", "HTML", "CSS", "Redux", "Node.js", "TypeScript"]
    }
    all_skills = {skill.lower() for category in skillsets.values() for skill in category}
    matched_skills = [skill for skill in all_skills if skill in text.lower()]
    return ", ".join(matched_skills) if matched_skills else "Not Found"

# 5. Extract Education
def extract_education(text):
    education_keywords = ["PhD", "Master", "Bachelor", "MBA", "B.Tech", "M.Tech", "BSc", "B.Sc", "BCA", "MCA", "BCS", "BE", "ME", "M.E."]
    pattern = r"\b(" + "|".join(education_keywords) + r")\b"
    matches = re.findall(pattern, text, re.IGNORECASE)
    validated_matches = sorted(set(matches), key=lambda x: education_keywords.index(x) if x in education_keywords else 999)
    return ", ".join(validated_matches) if validated_matches else "Not Found"

# 6. Extract Experience
def extract_experience(text):
    experience_pattern = r"(\d+)\s*(?:years?|months?)"
    matches = re.findall(experience_pattern, text, re.IGNORECASE)
    if matches:
        return f"{max(map(int, matches))} years"
    return "Not Found"

# 7. Extract Certifications
def extract_certifications(text):
    """
    Extracts certifications from the resume text using keywords and context-based regex patterns.
    """
    # Define a list of common certification keywords
    certification_keywords = [
        "certified", "certification", "certificate", "exam", "accreditation", "license",
        "Microsoft Certified", "AWS Certified", "PMP", "CFA", "CPA", "Google Cloud Certified",
        "Six Sigma", "Scrum Master", "ITIL", "Cisco Certified", "CompTIA", "Oracle Certified",
        "TOGAF", "ISTQB", "CEH", "CISSP"
    ]
    
    # Regex to match certification phrases containing the keywords
    pattern = r"\b(?:{})\b.*?(?=\n|\.|\;|\!|$)".format("|".join(re.escape(keyword) for keyword in certification_keywords))
    matches = re.findall(pattern, text, re.IGNORECASE)
    
    # Clean and deduplicate the matched certifications
    cleaned_matches = [match.strip() for match in matches]
    unique_certifications = sorted(set(cleaned_matches), key=lambda x: text.lower().find(x.lower()))
    
    # Return as a comma-separated string
    return ", ".join(unique_certifications) if unique_certifications else "Not Found"


# Classification
def classify_resume(parsed_text):
    try:
        category_index = model_pipeline.predict([parsed_text])[0]
        return label_encoder.inverse_transform([category_index])[0]
    except Exception as e:
        return f"Classification Error: {e}"

# Ranking Logic
def calculate_ranking(parsed_details):
    """
    Calculates the ranking score based on Skills, Experience, and Education.
    Handles multiple degrees by selecting the highest-ranked one.
    """
    # Count skills
    skills_count = len(parsed_details["Skills"].split(", ")) if parsed_details["Skills"] != "Not Found" else 0

    # Extract experience in years
    experience_years = 0
    if parsed_details["Experience"] != "Not Found":
        experience_match = re.search(r"(\d+)", parsed_details["Experience"])
        if experience_match:
            experience_years = int(experience_match.group(1))

    # Map education to scores
    education_score_mapping = {
        "PhD": 3,
        "Master": 2,
        "M.Tech": 2,
        "ME": 2,
        "MBA": 2,
        "MCA": 2,
        "MSc": 2,
        "Bachelor": 1,
        "B.Tech": 1,
        "BE": 1,
        "B.Sc": 1,
        "BSc": 1,
        "BCA": 1,
        "BCS": 1,
    }
    
    # Extract highest-ranked education
    education_score = 0
    if parsed_details["Education"] != "Not Found":
        degrees = parsed_details["Education"].split(", ")
        education_score = max(
            [education_score_mapping.get(degree, 0) for degree in degrees]
        )

    # Calculate total score
    total_score = (0.4 * skills_count) + (0.4 * experience_years) + (0.2 * education_score)
    return round(total_score, 2)


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/uploaded_resumes/<filename>')
def uploaded_file(filename):
    """
    Serves the uploaded resumes in their original format.
    """
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename, as_attachment=True)


@app.route('/submit', methods=['POST'])
def submit():
    parsed_resumes = []

    # Check for uploaded files
    if 'resume_file' in request.files:
        files = request.files.getlist('resume_file')
        if files and files[0].filename != '':
            # Process each file
            for file in files:
                try:
                    # Save the original file in the upload directory
                    file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
                    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
                    file.save(file_path)

                    # Read and process the file
                    file_text = ""
                    if file.filename.endswith('.docx'):
                        # Handle .docx files
                        doc = Document(file_path)
                        file_text = '\n'.join([p.text for p in doc.paragraphs])
                    elif file.filename.endswith('.pdf'):
                        # Handle .pdf files
                        from PyPDF2 import PdfReader
                        pdf_reader = PdfReader(file_path)
                        file_text = ''.join([page.extract_text() for page in pdf_reader.pages if page.extract_text()])
                    elif file.filename.endswith('.txt'):
                        # Handle .txt files
                        with open(file_path, 'r', encoding='utf-8') as f:
                            file_text = f.read()
                    else:
                        # Unsupported file format
                        return render_template('index.html', error=f"Unsupported file format: {file.filename}. Please upload .docx, .pdf, or .txt files.")

                    # Parse the resume text
                    parsed_details = parse_resume(file_text)
                    parsed_details["Category"] = classify_resume(file_text)
                    parsed_details["Ranking"] = calculate_ranking(parsed_details)
                    parsed_details["Link"] = url_for('uploaded_file', filename=file.filename)  # Original file link
                    parsed_resumes.append(parsed_details)

                except Exception as e:
                    return render_template('index.html', error=f"Error processing file {file.filename}: {e}")

    # Handle pasted text (optional, no file saved for this case)
    text_input = request.form.get('resume_text', '').strip()
    if text_input:
        try:
            # Save pasted text in a temporary file
            file_name = f"resume_pasted_{len(parsed_resumes) + 1}.txt"
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], file_name)
            os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(text_input)

            # Parse the pasted text
            parsed_details = parse_resume(text_input)
            parsed_details["Category"] = classify_resume(text_input)
            parsed_details["Ranking"] = calculate_ranking(parsed_details)
            parsed_details["Link"] = url_for('uploaded_file', filename=file_name)
            parsed_resumes.append(parsed_details)
        except Exception as e:
            return render_template('index.html', error=f"Error processing pasted text: {e}")

    # If no files or text are provided
    if not parsed_resumes:
        return render_template('index.html', error="Please upload files or paste resume text.")

    # Sort parsed resumes by ranking in descending order
    parsed_resumes = sorted(parsed_resumes, key=lambda x: x["Ranking"], reverse=True)

    return render_template('index.html', parsed_resumes=parsed_resumes)


@app.route('/clear', methods=['POST'])
def clear():
    return redirect(url_for('index'))

if __name__ == '__main__':
    app.run(debug=True)
