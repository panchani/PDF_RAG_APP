import streamlit as st
import os
import tempfile
import functionality as ft
vector_db=None


st.title("PDF Aware Groq Chatbot")

#PDF Upload
uploaded_file = st.file_uploader("Upload a PDF", type="pdf")
if uploaded_file is not None:

    vector_db=ft.return_vector_db(uploaded_file)

   
# store chat history
if "messages" not in st.session_state:
    st.session_state.messages = ft.system_prompt

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
    context = ft.retrieve_context(vector_db,prompt)

    st.markdown(context)
    user_prompt = ft.get_user_prompt_rag(context, prompt)
    mes = st.session_state.messages + [{"role": "user", "content": user_prompt}]
    try:
        reply=ft.llm_inference(mes)
    except Exception as e:
        reply = f"Error: {e}"
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.session_state.messages.append({"role": "assistant", "content": reply})

    with st.chat_message("assistant"):
        st.markdown(reply) 
elif prompt:
    with st.chat_message("user"):
        st.markdown(prompt)
   
    user_prompt = ft.get_user_prompt_without_rag(prompt)
    mes = st.session_state.messages + [{"role": "user", "content": user_prompt}]
    try:
        reply=ft.llm_inference(mes)



    except Exception as e:
        reply = f"Error: {e}"
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.session_state.messages.append({"role": "assistant", "content": reply})

    with st.chat_message("assistant"):
        st.markdown(reply) 