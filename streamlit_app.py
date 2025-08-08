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
    return OpenAIEmbeddings(model="text-embedding-3-large")

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
        messages=[{"role": "user", "content": prompt}]
        # max_completion_tokens=300
    )
    return response.choices[0].message.content.strip()

# --- App Layout ---
st.title("Job Description and Resume Matcher (GPT-5-nano)")
st.markdown("Provide a job description and candidate resumes. We'll match and rank candidates by relevance.")

# --- Upload Job Description ---
st.markdown("### 📄 (1) Job Description")
job_text = st.text_area("Paste Job Description, then type CTRL+ENTER", height=200)
job_file = st.file_uploader("Or upload job description file", type=["txt", "pdf", "docx"], key="job")
if job_file:
    job_text = parse_file(job_file)

# --- INPUT RESUMES ---
st.markdown("### 👥 (2) Candidate Resumes")
resume_files = st.file_uploader("Upload Candidate Resumes", type=["txt", "pdf", "docx"], accept_multiple_files=True)

if job_text and resume_files:
    with st.spinner("🔄 Processing resumes and matching..."):

        embedding = get_embedding()

        # Read resume texts
        resumes = []
        for file in resume_files:
            text = parse_file(file)
            resumes.append(Document(page_content=text, metadata={"id": file.name}))

        # Create or load Deeplake vectorstore
        vs = DeeplakeVectorStore.from_documents(
            documents=resumes,
            embedding=embedding,
            dataset_path="hub://kentasuzue/resume-matcher",  # replace with your path
            overwrite=True  # overwrite dataset each run; remove if you want to append
        )

        # Search top matches for job description
        results = vs.similarity_search_with_score(job_text, k=min(10, len(resume_files)))

        candidates = []
        for doc, score in results:
            summary = generate_summary_with_gpt(job_text, doc.page_content)
            candidates.append({
                "name": doc.metadata.get("id", "Unknown"),
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
