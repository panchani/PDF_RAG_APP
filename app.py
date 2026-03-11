import streamlit as st
import os
from dotenv import load_dotenv
from groq import Groq
import tempfile
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv()
vector_db=None
api_key = os.getenv("GROK_API_KEY")

if not api_key:
    st.error("GROK_API_KEY not found")
    st.stop()

client = Groq(api_key=api_key)

st.title("PDF Aware Groq Chatbot")

#PDF Upload
uploaded_file = st.file_uploader("Upload a PDF", type="pdf")
if uploaded_file is not None:

    # create temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.read())
        temp_path = tmp_file.name

    # now pass the path to PyPDFLoader
    loader = PyPDFLoader(temp_path)
    docs = loader.load()
       
    # Split into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=4000,
        chunk_overlap=50
    )

    chunks = text_splitter.split_documents(docs)

    # create embeddings
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vector_db = FAISS.from_documents(chunks, embeddings)

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

if prompt and vector_db:

    # st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)
   
    # build user prompt with context
    context = vector_db.similarity_search(
    prompt,
    k=3
    )


    user_prompt = f"""
    Use the context below to answer the question.

    context:
    {context}

    Question:
    {prompt}
    """
    mes = st.session_state.messages + [{"role": "user", "content": user_prompt}]
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
elif prompt:
    with st.chat_message("user"):
        st.markdown(prompt)
   
    user_prompt = f"""
    answer the question considering system restrictions.
    Question:
    {prompt}
    """
    mes = st.session_state.messages + [{"role": "user", "content": user_prompt}]
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