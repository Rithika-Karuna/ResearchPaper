import streamlit as st
from sentence_transformers import SentenceTransformer, util
import pdfplumber

# Load Sentence-BERT model
sbert_model = SentenceTransformer('all-MiniLM-L6-v2')

# Extract text from PDF using pdfplumber
def extract_text_from_pdf(pdf_path):
    try:
        text = ""
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                text += page.extract_text()
        return text.strip()
    except Exception as e:
        st.error(f"Error reading {pdf_path}: {e}")
        return ""

# Extract title from the first page
def extract_title_from_pdf(pdf_path):
    try:
        with pdfplumber.open(pdf_path) as pdf:
            first_page = pdf.pages[0]
            lines = first_page.extract_text().split("\n")
            title = lines[0] + " " + lines[1] if len(lines) > 1 else lines[0]
            return title.strip() if title else "Title could not be identified."
    except Exception as e:
        st.error(f"Error reading PDF title: {e}")
        return "Error occurred during title extraction."

# Extract the conclusion section
def extract_conclusion(text):
    conclusion_start = text.lower().find('conclusion')
    if conclusion_start == -1:
        return None
    conclusion_text = text[conclusion_start:]
    headings = ['introduction', 'methods', 'references', 'abstract']
    end_index = len(conclusion_text)
    for heading in headings:
        heading_start = conclusion_text.lower().find(heading)
        if heading_start != -1:
            end_index = min(end_index, heading_start)
    return conclusion_text[:end_index].strip()

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
    reference_text = "This is a sample reference text for measuring similarity."
    embedding1 = sbert_model.encode(text, convert_to_tensor=True)
    embedding2 = sbert_model.encode(reference_text, convert_to_tensor=True)
    similarity = util.pytorch_cos_sim(embedding1, embedding2)
    return float(similarity)

# Predict paper acceptance dynamically with feedback
def predict_paper_acceptance(new_paper_text, title):
    feedback = [f"Topic of the research paper: {title}"]

    conclusion_text = extract_conclusion(new_paper_text)
    conclusion_valid, conclusion_count, conclusion_feedback = validate_section_word_count(conclusion_text, 300)
    feedback.append(conclusion_feedback)

    # Calculate similarity score once
    score = compute_similarity_score(new_paper_text)
    plagiarism_score = score
    similarity_score = score
    innovation_score = 1 - plagiarism_score
    novelty_score = 0.5 + similarity_score / 2

    if not conclusion_valid:
        feedback.append("Paper Rejected due to missing or invalid conclusion.")
        feedback.append(f"High plagiarism score ({plagiarism_score:.2f}). Consider rephrasing and adding unique content.")
        feedback.append(f"Low novelty score ({novelty_score:.2f}). Enhance originality.")
        feedback.append(f"Low innovation score ({innovation_score:.2f}). Emphasize unique ideas.")
        feedback.append("Ensure proper citation and a clear structure.")
        return "Paper Rejected", feedback, plagiarism_score, similarity_score, innovation_score, novelty_score, conclusion_count

    feedback.append("Paper Accepted")
    return "Paper Accepted", feedback, plagiarism_score, similarity_score, innovation_score, novelty_score, conclusion_count

# Streamlit app
st.title("Research Paper Review System")

uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")

if uploaded_file is not None:
    with st.spinner("Processing PDF..."):
        title = extract_title_from_pdf(uploaded_file)
        new_paper_text = extract_text_from_pdf(uploaded_file)

    if new_paper_text:
        st.subheader("Paper Title")
        st.write(title)

        result, feedback, plagiarism, similarity, innovation, novelty, conclusion_count = predict_paper_acceptance(new_paper_text, title)

        st.subheader("Result")
        st.write(result)

        st.subheader("Feedback")
        for comment in feedback:
            st.write(f"- {comment}")

        st.subheader("Scores")
        st.write(f"Plagiarism Score: {plagiarism:.2f}")
        st.write(f"Similarity Score: {similarity:.2f}")
        st.write(f"Innovation Score: {innovation:.2f}")
        st.write(f"Novelty Score: {novelty:.2f}")
    else:
        st.error("Could not extract text from the uploaded PDF.")
