import tempfile
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from .config import config, get_embedding_model

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

def create_vector_db(chunks, embeddings):
    vector_db = FAISS.from_documents(chunks, embeddings)
    return vector_db

def return_vector_db(uploaded_file):
    chunks = create_chunks(uploaded_file)
    embeddings = get_embedding_model()
    vector_db = create_vector_db(chunks, embeddings)
    return vector_db

def retrieve_context(vector_db, prompt):
    context = vector_db.similarity_search(
        prompt,
        k=config["retriever"]["k"]
    )
    return context