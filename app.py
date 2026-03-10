import streamlit as st
import os
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

api_key = os.getenv("GROK_API_KEY")
if not api_key:
    st.error("GROK_API_KEY not found")
    st.stop()

client = Groq(api_key=api_key)

st.title("Groq Chatbot")

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
- Professional workplace topics
- Greetings (hello, hi, good morning etc.)

If the user asks anything outside these topics (politics, history, entertainment, personal advice, etc.), you MUST respond with exactly:

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
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

    try:
        res = client.chat.completions.create(
            #model="openai/gpt-oss-20b",
            #model="groq/compound",
            model="llama-3.1-8b-instant",
            messages=st.session_state.messages
        )
        reply = res.choices[0].message.content
    except Exception as e:
        reply = f"Error: {e}"

    st.session_state.messages.append({"role": "assistant", "content": reply})

    with st.chat_message("assistant"):
        st.markdown(reply) 