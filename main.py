import streamlit as st
import nltk
from sentence_transformers import SentenceTransformer, util
import pdfplumber
from rake_nltk import Rake

# Download necessary NLTK resources
nltk.download('stopwords')
nltk.download('punkt')

# Load Sentence-BERT model
sbert_model = SentenceTransformer('all-MiniLM-L6-v2')

# Function to extract text from PDF using pdfplumber
def extract_text_from_pdf(pdf_file):
    try:
        text = ""
        with pdfplumber.open(pdf_file) as pdf:
            for page in pdf.pages:
                text += page.extract_text()
        return text.strip()
    except Exception as e:
        st.error(f"Error reading PDF: {e}")
        return ""

# Extract a specific section from the text
def extract_section(text, section):
    start_idx = text.lower().find(section)
    if start_idx == -1:
        return None  # Section not found

    # Find the next heading or end of the text
    next_heading_idx = text.find("\n", start_idx + len(section))
    section_text = text[start_idx:next_heading_idx].strip() if next_heading_idx != -1 else text[start_idx:].strip()
    return section_text

# Refined function to extract the conclusion
def extract_conclusion(text):
    conclusion_start = text.lower().find('conclusion')
    if conclusion_start == -1:
        return None
    conclusion_text = text[conclusion_start:]

    # Look for the next major heading (e.g., Introduction, Methodology) and trim text after that.
    conclusion_end = conclusion_text.lower().find("introduction")  # or any other major section name
    if conclusion_end == -1:
        conclusion_end = conclusion_text.lower().find("methods")  # Adding another heading for safety
    if conclusion_end != -1:
        conclusion_text = conclusion_text[:conclusion_end]

    return conclusion_text.strip()

# Validate word count for a specific section
def validate_section_word_count(section_text, max_word_count):
    if section_text is None:
        return False, 0, "The conclusion section is missing."
    word_count = len(section_text.split())
    if word_count > max_word_count:
        return False, word_count, f"The conclusion exceeds the maximum word count of {max_word_count}. Current count: {word_count}."
    return True, word_count, "The conclusion meets the word count requirement."

# Compute similarity score
def compute_similarity_score(text):
    # Use Sentence-BERT embeddings for similarity computation
    reference_text = "This is a sample reference text for measuring similarity."
    embedding1 = sbert_model.encode(text, convert_to_tensor=True)
    embedding2 = sbert_model.encode(reference_text, convert_to_tensor=True)
    similarity = util.pytorch_cos_sim(embedding1, embedding2)
    return float(similarity)

# Predict paper acceptance dynamically with feedback
def predict_paper_acceptance(new_paper_text):
    feedback = []

    # Extract and validate conclusion
    conclusion_text = extract_conclusion(new_paper_text)
    conclusion_valid, conclusion_count, conclusion_feedback = validate_section_word_count(conclusion_text, 300)
    feedback.append(conclusion_feedback)

    if not conclusion_valid:
        return "Paper Rejected", feedback, None, None, None, None, conclusion_count

    # Compute scores dynamically
    plagiarism_score = compute_similarity_score(new_paper_text)
    similarity_score = compute_similarity_score(conclusion_text)
    innovation_score = 1 - plagiarism_score  # Example computation
    novelty_score = 0.5 + similarity_score / 2  # Example logic for novelty

    # Generate specific feedback for scores
    if plagiarism_score > 0.6:
        feedback.append(f"High plagiarism score ({plagiarism_score:.2f}). Consider rephrasing and adding unique content.")
    else:
        feedback.append("Plagiarism score is acceptable.")

    if novelty_score < 0.6:
        feedback.append(f"Low novelty score ({novelty_score:.2f}). Enhance originality or innovative aspects of the paper.")
    else:
        feedback.append("Novelty score is good.")

    if innovation_score < 0.5:
        feedback.append(f"Low innovation score ({innovation_score:.2f}). Emphasize unique ideas and contributions.")
    else:
        feedback.append("Innovation score indicates a good level of originality.")

    # Add overall feedback
    feedback.append("Ensure proper citation and a clear structure for better understanding.")

    return "Paper Accepted", feedback, plagiarism_score, similarity_score, innovation_score, novelty_score, conclusion_count

# Streamlit application
st.title("Research Paper Evaluation Tool")
st.write("Upload a PDF to analyze and evaluate its content dynamically.")

uploaded_file = st.file_uploader("Upload PDF File", type="pdf")

if uploaded_file is not None:
    st.info("Processing your file...")

    # Extract text from the uploaded PDF
    paper_text = extract_text_from_pdf(uploaded_file)

    if paper_text:
        # Predict acceptance and generate feedback
        result, feedback, plagiarism, similarity, innovation, novelty, conclusion_count = predict_paper_acceptance(paper_text)

        # Display results
        st.subheader("Evaluation Result")
        st.write(f"**Result:** {result}")

        st.subheader("Feedback")
        for reason in feedback:
            st.write(f"- {reason}")

        st.subheader("Scores")
        st.write(f"**Plagiarism Score:** {plagiarism:.2f}")
        st.write(f"**Similarity Score:** {similarity:.2f}")
        st.write(f"**Innovation Score:** {innovation:.2f}")
        st.write(f"**Novelty Score:** {novelty:.2f}")
    else:
        st.error("Could not extract text from the uploaded PDF. Please try another file.")
