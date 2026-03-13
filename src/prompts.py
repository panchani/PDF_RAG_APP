system_prompt = [{
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