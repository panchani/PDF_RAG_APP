from flask import Flask, request, jsonify
import functinality as ft
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Global state (similar to st.session_state)
vector_db = None
messages = ft.system_prompt.copy()

# -------------------------
# Ping Endpoint
# -------------------------
@app.route("/ping", methods=["GET"])
def ping():
    return jsonify({
        "status": "ok",
        "message": "API is running"
    })

# -------------------------
# PDF Upload Endpoint
# -------------------------
@app.route("/upload_pdf", methods=["POST"])
def upload_pdf():
    global vector_db

    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]

    vector_db = ft.return_vector_db(file)

    return jsonify({"message": "PDF processed successfully"})


# -------------------------
# Chat Endpoint
# -------------------------
@app.route("/chat", methods=["POST"])
def chat():

    global messages, vector_db

    data = request.json
    prompt = data.get("prompt")

    if not prompt:
        return jsonify({"error": "No prompt provided"}), 400


    # RAG case
    if vector_db:

        context = ft.retrieve_context(vector_db, prompt)

        user_prompt = ft.get_user_prompt_rag(context, prompt)

        mes = messages + [{"role": "user", "content": user_prompt}]

        try:
            reply = ft.llm_inference(mes)
        except Exception as e:
            reply = f"Error: {e}"

    # Non RAG case
    else:

        user_prompt = ft.get_user_prompt_without_rag(prompt)

        mes = messages + [{"role": "user", "content": user_prompt}]

        try:
            reply = ft.llm_inference(mes)
        except Exception as e:
            reply = f"Error: {e}"


    # Save history
    messages.append({"role": "user", "content": prompt})
    messages.append({"role": "assistant", "content": reply})

    return jsonify({
        "reply": reply,
        "history": messages
    })


# -------------------------
# Run server
# -------------------------
if __name__ == "__main__":
    app.run(debug=True)