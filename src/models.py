import os
from groq import Groq

def llm_inference(mes):
    api_key = os.getenv("GROQ_API_KEY")
    client = Groq(api_key=api_key)
    res = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=mes
    )
    reply = res.choices[0].message.content
    return reply