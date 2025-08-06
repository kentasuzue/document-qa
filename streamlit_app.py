st.write("Deeplake version:", deeplake.__version__)
from deeplake import VectorStore
import docx
import numpy as np
import openai
import os
from PyPDF2 import PdfReader
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st
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
@st.cache_data(show_spinner=False)
def get_embedding(text):
    response = openai.embeddings.create(
        model="text-embedding-3-large",
        input=text.strip().replace("\n", " ")
    )
    return response.data[0].embedding


# Generate GPT summary
@st.cache_data(show_spinner=False)
def generate_summary_with_gpt(job_text, resume_text):
    prompt = f"""
You are a helpful recruiter assistant. Given the job description and the candidate's resume, write a concise 2–3 sentence summary explaining how well the candidate fits the job. Highlight relevant experience and standout skills.

Job Description:
{job_text[:1500]}

Resume:
{resume_text[:3000]}

Summary:
"""
    response = openai.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        max_tokens=200
    )
    return response.choices[0].message.content.strip()

# --- App Layout ---
st.title("Job Description and Resume Matcher (GPT-4o-mini + ActiveLoop)")
st.markdown("Upload a job description and resumes. We'll match and rank candidates by relevance.")

# Let the user upload a file via `st.file_uploader`.
resume_files = st.file_uploader("Upload Candidate Resumes", type=["txt", "pdf", "docx"], accept_multiple_files=True, key="resumes")

# --- Upload Job Description ---
job_text = st.text_area("Paste Job Description", height=200)
job_file = st.file_uploader("Or upload job description file", type=["txt", "pdf", "docx"], key="job")
if job_file:
    job_text = parse_file(job_file)

if resume_files and job_text:

    # Process the uploaded file and question.
    document = uploaded_file.read().decode()
    messages = [
        {
            "role": "user",
            "content": f"Here's a document: {document} \n\n---\n\n {question}",
        }
    ]

    # Generate an answer using the OpenAI API.
    stream = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=messages,
        stream=True,
    )

    # Stream the response to the app using `st.write_stream`.
    st.write_stream(stream)
# --- INPUT JOB ---
job_text = st.text_area("Paste Job Description", height=200)
job_file = st.file_uploader("Or upload job description file", type=["txt", "pdf", "docx"])
if job_file:
    job_text = parse_file(job_file)

# --- INPUT RESUMES ---
resume_files = st.file_uploader("Upload Candidate Resumes", type=["txt", "pdf", "docx"], accept_multiple_files=True)

if job_text and resume_files:
    with st.spinner("🔄 Processing resumes and matching..."):

        # Create or connect to Deep Lake vector store
        vs = VectorStore(
            path="hub://your-username/resume-matcher",  # Replace with your hub path
            embedding_function=get_embedding
        )

        # Add resumes to vector store
        for file in resume_files:
            resume_text = parse_file(file)
            vs.add(texts=[resume_text], ids=[file.name])

        # Embed job description
        job_embedding = get_embedding(job_text)

        # Search similar resumes
        search_results = vs.search(query_vector=job_embedding, k=min(10, len(resume_files)))

        # Collect top results with summaries
        candidates = []
        for match in search_results["matches"]:
            resume_text = match["text"]
            score = match["score"]
            name = match["id"]
            summary = generate_summary_with_gpt(job_text, resume_text)

            candidates.append({
                "name": name,
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
