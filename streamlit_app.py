from deeplake import VectorStore
import docx
import numpy as np
from openai import OpenAI
import os
from PyPDF2 import PdfReader
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st

# Load API keys
openai.api_key = st.secrets["OPENAI_API_KEY"]
os.environ["ACTIVELOOP_TOKEN"] = st.secrets["ACTIVELOOP_TOKEN"]

# File Read Functions
def read_pdf(file):
    reader = PdfReader(file)
    return "\n".join([page.extract_text() or "" for page in reader.pages])

def read_docx(file):
    doc = docx.Document(file)
    return "\n".join([para.text for para in doc.paragraphs])

def read_text_file(file):
    return file.read().decode("utf-8")

def parse_file(file):
    if file.name.endswith(".pdf"):
        return read_pdf(file)
    elif file.name.endswith(".doc") or file.name.endswith(".docx"):
        return read_docx(file)
    elif file.name.endswith(".txt"):
        return read_text_file(file)
    else:
        return "Unsupported file format"

# Get OpenAI embedding
@st.cache_data(show_spinner=False)
def get_embedding(text):
    response = openai.embeddings.create(
        model="text-embedding-3-small",
        input=text.strip().replace("\n", " ")
    )
    return response.data[0].embedding

# --- App Layout ---
st.title("Job Description and Resume Matcher")
st.write(
    "Upload a document below and ask a question about it – GPT will answer! "
    "To use this app, you need to provide an OpenAI API key, which you can get [here](https://platform.openai.com/account/api-keys). "
)

# Generate GPT summary
@st.cache_data(show_spinner=False)
def generate_match_summary(job_text, resume_text):
    prompt = f"""
You are a helpful recruiter assistant. Summarize in 2-3 sentences how well this resume matches the job.

Job:
{job_text}

Resume:
{resume_text[:3000]}
"""
    response = openai.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=150
    )
    return response.choices[0].message.content.strip()


# Create an OpenAI client.
client = OpenAI(api_key=openai_api_key)

# Let the user upload a file via `st.file_uploader`.
uploaded_file = st.file_uploader(
    "Upload a document (.txt or .md)", type=("txt", "md")
)

# Ask the user for a question via `st.text_area`.
question = st.text_area(
    "Now ask a question about the document!",
    placeholder="Can you give me a short summary?",
    disabled=not uploaded_file,
)

if uploaded_file and question:

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
