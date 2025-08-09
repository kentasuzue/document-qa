1) Approach:
The architecture is based on Streamlit using Python, OpenAI's ChatGPT, and ActiveLoop.
At https://share.streamlit.io/ this app's settings have the secrets "OPENAI_API_KEY" and "ACTIVELOOP_TOKEN".  The job description and each resume are converted to the embeddings in "text-embedding-3-large".  **The DeeplakeVectorStore method similarity_search_with_score specifies that distance_metric is 'cos' to override the default of 'L2'.  Specifying 'cos' as distance_metric make the app comply with the assignment's instructions to "Compute cosine similarity between the job and each resume".**
ChatGPT-5 is used, because it's cool!  

3) Assumptions:
Job description and resumes are assumed to be in English.
All files are assumed to be in pdf, docx, or txt (UTF-8) format.
The resumes are assumed to begin with the candidate's name, which is extracted from the beginning of each resume.   

4) Anything else:
OpenAI data controls enable sharing of the contents of the job description and resumes with OpenAI.
Confidential information such as social security numbers should not be entered.  
