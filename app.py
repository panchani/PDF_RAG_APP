import streamlit as st
from dotenv import load_dotenv 
from PyPDF2 import PdfReader
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings 
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEndpoint
from langchain_community.llms.huggingface_hub import HuggingFaceHub
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from operator import itemgetter
from transformers import pipeline
from langchain_huggingface import HuggingFacePipeline


def get_pdf_text(pdf_docs):
    text=""
    for pdf in pdf_docs:
        pdf_reader=PdfReader(pdf)
        for page in pdf_reader.pages:
            text+=page.extract_text()
    return text

def get_text_chunks(raw_text):
    text_splitter = CharacterTextSplitter(
    separator="\n",
    chunk_size=1000,
    chunk_overlap=200,
    length_function=len
    )
    chunks=text_splitter.split_text(raw_text)
    return chunks 

def get_vectorstore(text_chunks):
    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2"
    )
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore
    
def get_conversation_chain(vectorstore):
    pipe = pipeline(
    "text-generation",
    model="google/flan-t5-base",
    device=-1,
    max_new_tokens=512
    )

    llm = HuggingFacePipeline(pipeline=pipe)
    retriever = vectorstore.as_retriever()
    
    # QA Prompt
    template = """Answer the question based only on the following context:
{context}

Question: {question}"""
    prompt = ChatPromptTemplate.from_template(template)
    
    # RAG Chain
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)
    
    rag_chain = (
        {"context": retriever | RunnableLambda(format_docs), "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    
    return rag_chain

def handle_userinput(user_question):
    if st.session_state.conversation is not None:
        response = st.session_state.conversation.invoke(user_question)
        st.write(response) 


def main():
    load_dotenv()
    st.set_page_config(page_title="PDF RAG App", page_icon=":books:")

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    st.header("Chat with your PDF documents :books:")
    user_question = st.text_input("Upload your PDF files and start asking questions about their content!")
    if user_question:
        handle_userinput(user_question)
    with st.sidebar:
        st.subheader("Your Documents")
        pdf_docs = st.file_uploader("Upload PDF files", type=["pdf"], accept_multiple_files=True)
        if st.button("Process"):
            with st.spinner("Processing your documents..."):
                raw_text = get_pdf_text(pdf_docs)

                text_chunks = get_text_chunks(raw_text)

                vectorstore = get_vectorstore(text_chunks)

                st.session_state.conversation = get_conversation_chain(vectorstore)
    
if __name__ == "__main__":
    main()