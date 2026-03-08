import streamlit as st

def main():
    st.set_page_config(page_title="PDF RAG App", page_icon=":books:")

    st.header("Chat with your PDF documents :books:")
    st.text_input("Upload your PDF files and start asking questions about their content!")

    with st.sidebar:
        st.subheader("Your Documents")
        st.file_uploader("Upload PDF files", type=["pdf"], accept_multiple_files=True)
        st.button("Process Documents")
    
if __name__ == "__main__":
    main()