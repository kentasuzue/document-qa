1) Approach:
The architecture is based on Streamlit using Python, OpenAI's ChatGPT, and ActiveLoop.
At https://share.streamlit.io/ this app's settings have the secrets "OPENAI_API_KEY" and "ACTIVELOOP_TOKEN".
ChatGPT-5 is used, because it's cool!

2) Assumptions:
Job description and resumes are assumed to be in English.
The resumes are assumed to begin with the candidate's name, which is extracted from the beginning of each resume.   

3) Anything else:
OpenAI data controls enable sharing of the contents of the job description and resumes with OpenAI.
Confidential information such as social security numbers should not be entered.  
