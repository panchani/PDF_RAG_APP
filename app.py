import streamlit as st
import os
from dotenv import load_dotenv
from groq import Groq
from pypdf import PdfReader

load_dotenv()

api_key = os.getenv("GROK_API_KEY")

if not api_key:
    st.error("GROK_API_KEY not found")
    st.stop()

client = Groq(api_key=api_key)

st.title("PDF Aware Groq Chatbot")

#PDF Upload
uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

def extract_pdf_text(file):
    reader = PdfReader(file)
    text = ""

    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"

    return text

# detect new pdf upload
if uploaded_file is not None:

    if "current_pdf" not in st.session_state or \
       st.session_state.current_pdf != uploaded_file.name:

        st.session_state.current_pdf = uploaded_file.name
        st.session_state.pdf_context = extract_pdf_text(uploaded_file)

        # reset chat history
        st.session_state.messages = []

        st.success("New PDF loaded. Chat reset.")

# show current pdf
if "current_pdf" in st.session_state:
    st.info(f"Current PDF: {st.session_state.current_pdf}")

# store chat history
if "messages" not in st.session_state:
    st.session_state.messages = [{
            "role": "system",
            "content": """
You are a restricted assistant.

You are ONLY allowed to answer questions related to:
- Information Technology (IT)
- Corporate sector
- Programming
- Software development
- Technical topics
- Greetings
- The uploaded PDF

Rules:
- Use the provided context when answering questions about the PDF.
- If the answer is not in the context say:
"I couldn't find that information in the document."
- If the user is asking general question related to the allowed topic then you can ignore the context"
If the user asks anything outside the allowed topics respond with:
"Sorry, I'm not allowed to answer that."
"""
        }]

# display old messages
for m in st.session_state.messages:
    if m["role"]!="system":
        with st.chat_message(m["role"]):
            st.markdown(m["content"])

# user input
prompt = st.chat_input("Ask something")

if prompt:
    # st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)
   
    # build user prompt with context
    context = st.session_state.get("pdf_context", "")


    user_prompt = f"""
    Use the context below to answer the question.

    context:
    {context}

    Question:
    {prompt}
    """
    mes = st.session_state.messages + [{"role": "user", "content": user_prompt}]
    st.write(st.session_state)
    try:
        res = client.chat.completions.create(
            #model="openai/gpt-oss-20b",
            #model="groq/compound",
            model="llama-3.1-8b-instant",
            messages=mes
        )
        reply = res.choices[0].message.content
    except Exception as e:
        reply = f"Error: {e}"
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.session_state.messages.append({"role": "assistant", "content": reply})

    with st.chat_message("assistant"):
        st.markdown(reply) 