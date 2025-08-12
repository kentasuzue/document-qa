1) Approach:
The architecture is based on Streamlit using Python, OpenAI's ChatGPT, and LangChain with ActiveLoop.
At https://share.streamlit.io/ this app's settings have the secrets "OPENAI_API_KEY" and "ACTIVELOOP_TOKEN".  The user is prompted with easy-to-understand numbered instructions and sections to provide (1) job description and (2) candidate resumes.  The job description and each resume are converted to the embeddings from OpenAI's "text-embedding-3-large", and compared with the DeeplakeVectorStore method similarity_search_with_score.  **The DeeplakeVectorStore method similarity_search_with_score specifies that distance_metric is 'cos' to override the default of 'L2'.  Specifying 'cos' as distance_metric makes the app comply with instructions to "Compute cosine similarity between the job and each resume".**  The ChatGPT-5 model "gpt-5-nano" is used to summarize each resume with verbosity of "high" because it's cool!  Each candidate's name is extracted from the beginning of each resume, so that in the final report the user is presented with a friendly list, such that each list item begins with an easy-to-read candidate's name instead of a hard-to-read resume file name. 

2) Assumptions:
Job description and resumes are assumed to be in English.
All files are assumed to be in pdf, docx, or txt (UTF-8) format.
The resumes are assumed to begin with the candidate's name, which is extracted from the beginning of each resume.
No two candidates have the same name.  If two candidates do have the same name, then the final summary may recommend two candidates with the same name, necessitating that the name collision be resolved by comparing the original resumes with the ChatGPT-5 generated summaries.

3) Anything else:
OpenAI data controls are set to enable sharing of the contents of the job description and resumes with OpenAI.
Confidential information such as social security numbers should not be entered.
MAX_RESUMES_SUMMARIZED at the beginning of streamlit_app.py can be changed from 10 to change how many candidate summaries are shown.
On occasion Streamlit fails to load; then the user should refresh the webpage.
