import streamlit as st
from langchain_deeplake.vectorstores import DeeplakeVectorStore
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
import docx
import numpy as np
import openai
import os
from PyPDF2 import PdfReader
from sklearn.metrics.pairwise import cosine_similarity
import tempfile
import spacy

MAX_RESUMES_SUMMARIZED = 10

# Load API keys
openai.api_key = st.secrets["OPENAI_API_KEY"]
os.environ["ACTIVELOOP_TOKEN"] = st.secrets["ACTIVELOOP_TOKEN"]

# File Read Functions
def parse_file(file):
    if file.name.endswith(".pdf"):
        reader = PdfReader(file)
        return "\n".join([page.extract_text() or "" for page in reader.pages])
    elif file.name.endswith(".docx"):
        doc = docx.Document(file)
        return "\n".join([para.text for para in doc.paragraphs])
    elif file.name.endswith(".txt"):
        return file.read().decode("utf-8")
    else:
        return "Unsupported file type"

# Get OpenAI embedding
@st.cache_resource(show_spinner=False)
def get_embedding():
    return OpenAIEmbeddings(model="text-embedding-3-small")

# def extract_candidate_name(resume_text):
#     doc = nlp(text)
#     for ent in doc.ents:
#         if ent.label_ == "PERSON":
#             return ent.text
#     return "Unknown"

def extract_candidate_name(resume_text):
    """
    Attempts to extract the candidate's name from the resume.
    Assumes the name is within the first few lines.
    """
    lines = resume_text.strip().splitlines()
    for line in lines[:10]:  # Check the top 10 lines
        line = line.strip()
        if line and len(line.split()) <= 5 and any(char.isalpha() for char in line):
            return line
    return "Unknown"

# Generate GPT summary
@st.cache_data(show_spinner=False)
def generate_summary_with_gpt(job_text, resume_text):
    prompt = f"""
You are a helpful recruiter assistant. Given the job description and the candidate's resume, write a concise 2–3 sentence summary explaining how well the candidate fits the job. Highlight relevant experience and standout skills.

Job Description:
{job_text}

Resume:
{resume_text}

Summary:
"""
    response = openai.chat.completions.create(
        model="gpt-5-nano",
        messages=[{"role": "user", "content": prompt}],
        verbosity="high"
    )
    return response.choices[0].message.content.strip()

# --- App Layout ---
st.title("Job Description and Resume Matcher")
st.markdown("## New and Improved with ChatGPT-5!")
st.markdown("""
            <ul>
              <li>(1) Provide a job description.</li>
              <li>(2) Provide candidate resumes.</li>
              <li>We'll match and rank candidates by relevance, and provide summaries of why each candidate is a great fit for the role!</li>
            </ul>
            """, unsafe_allow_html=True)


# --- Upload Job Description ---
st.markdown("#### 👷 (1) Job Description")
job_text = st.text_area("Paste Job Description, then type CTRL+ENTER", height=200)
job_file = st.file_uploader("Or upload job description file", type=["txt", "pdf", "docx"], key="job")
if job_file:
    job_text = parse_file(job_file)

# --- INPUT RESUMES ---
st.markdown("#### 👥 (2) Candidate Resumes")
resume_files = st.file_uploader("Upload Candidate Resumes", type=["txt", "pdf", "docx"], accept_multiple_files=True)

if job_text and resume_files:
    with st.spinner("🔄 Processing resumes and matching..."):

        embedding = get_embedding()

        # Load spaCy English model once
        nlp = spacy.load("en_core_web_sm")

        # Read resume texts
        resumes = []
        for file in resume_files:
            text = parse_file(file)
            candidate_name = extract_candidate_name(text)
            resumes.append(Document(page_content=text, metadata={
                "id": file.name,
                "candidate_name": candidate_name
            }))
        
        # Create or load Deeplake vectorstore
        vs = DeeplakeVectorStore.from_documents(
            documents=resumes,
            embedding=embedding,
            dataset_path="hub://kentasuzue/resume-matcher",  # replace with your path
            overwrite=True  # overwrite dataset each run; remove if you want to append
        )

        # Search top matches for job description
        results = vs.similarity_search_with_score(job_text, k=min(MAX_RESUMES_SUMMARIZED, len(resume_files)))

        candidates = []
        for doc, score in results:
            summary = generate_summary_with_gpt(job_text, doc.page_content)
            candidates.append({
                "name": doc.metadata.get("candidate_name", doc.metadata.get("id", "Unknown")),
                "similarity": score,
                "summary": summary
            })

    # --- DISPLAY RESULTS ---
    st.header("🏆 Top Matching Candidates")
    for i, candidate in enumerate(candidates):
        st.subheader(f"#{i+1}: {candidate['name']}")
        st.write(f"**Similarity Score:** `{candidate['similarity']:.4f}`")
        st.write(f"**Summary:** {candidate['summary']}")
        st.markdown("---")
