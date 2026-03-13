import os
from dotenv import load_dotenv
from groq import Groq
import tempfile
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
import yaml
load_dotenv()


with open("config.yaml", "r") as file:
    config = yaml.safe_load(file)


embedding_model = HuggingFaceEmbeddings(
    model_name=config["embedding"]["model_name"],
    cache_folder=config["embedding"]["cache_folder"]
)

system_prompt=[{
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
def create_chunks(uploaded_file):
    # create temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.read())
        temp_path = tmp_file.name

    # now pass the path to PyPDFLoader
    loader = PyPDFLoader(temp_path)
    docs = loader.load()
       
    # Split into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=config["text_splitter"]["chunk_size"],
        chunk_overlap=config["text_splitter"]["chunk_overlap"]
    )

    chunks = text_splitter.split_documents(docs)

    return chunks

def get_embedding_model():
    return embedding_model

def create_vector_db(chunks, embeddings):
    vector_db = FAISS.from_documents(chunks, embeddings)
    return vector_db

def return_vector_db(uploaded_file):
    chunks=create_chunks(uploaded_file)
    embeddings=get_embedding_model()
    vector_db=create_vector_db(chunks, embeddings)
    return vector_db

def retrieve_context(vector_db,prompt):
    context=vector_db.similarity_search(
    prompt,
    k=config["retriever"]["k"]    )
    return context

def get_user_prompt_without_rag(prompt):
    user_prompt = f"""
    answer the question considering system restrictions.
    Question:
    {prompt}
    """
    return user_prompt

def get_user_prompt_rag(context, prompt):
    user_prompt = f"""
    Use the context below to answer the question.

    context:
    {context}

    Question:
    {prompt}
    """
    return user_prompt

api_key = os.getenv("GROQ_API_KEY")

client = Groq(api_key=api_key)

def llm_inference(mes):
    res = client.chat.completions.create(
            #model="openai/gpt-oss-20b",
            #model="groq/compound",
            model="llama-3.1-8b-instant",
            messages=mes
        )
    reply = res.choices[0].message.content
    return reply