# PDF RAG Assistant

A **Retrieval Augmented Generation (RAG)** application that allows users to upload a PDF and ask questions about its contents.
The system retrieves relevant chunks from the document using **FAISS vector search** and generates answers using an **LLM (via Groq API)**.

The assistant is restricted to answering questions related to:

* Information Technology (IT)
* Programming
* Software development
* Corporate / technical topics
* Content from the uploaded PDF
* Greetings

If a question is outside these topics, the assistant refuses to answer.

---

# Features

* Upload PDF and build a vector database
* Semantic search using **sentence-transformer embeddings**
* Context retrieval using **FAISS**
* LLM inference using **Groq**
* Configurable system via `config.yaml`
* Modular prompt management
* REST API using **Flask**
* Dockerized for easy deployment

---

# Project Structure

```
pdf_rag_app/
│
├── app.py
├── functionality.py
├── prompts.py
├── config.yaml
├── Dockerfile
├── requirements.txt
├── models/
└── README.md
```

---

# Configuration

Application parameters are defined in `config.yaml`.

Example:

```yaml
embedding:
  model_name: "sentence-transformers/all-MiniLM-L6-v2"
  cache_folder: "./models"

text_splitter:
  chunk_size: 4000
  chunk_overlap: 50

retriever:
  k: 3
```

You can modify these values without changing the code.

---

# Environment Variables

Create a `.env` file in the project root.

```
GROQ_API_KEY=your_groq_api_key
```

---

# Running Locally

## 1. Clone the repository

```
git clone <repo_url>
cd pdf_rag_app
```

---

## 2. Create virtual environment

Linux / Mac

```
python3 -m venv .venv
source .venv/bin/activate
```

Windows

```
python -m venv .venv
.venv\Scripts\activate
```

---

## 3. Install dependencies

```
pip install -r requirements.txt
```

---

## 4. Run the application

```
python app.py
```

The server will start at:

```
http://localhost:5000
```

---

# API Endpoints

## Health Check

```
GET /ping
```

Response:

```json
{
  "status": "running"
}
```

---

## Upload PDF

```
POST /upload
```

Uploads a PDF and builds the vector database.

---

## Ask Question

```
POST /chat
```

Example request:

```json
{
  "message": "What architecture does the transformer use?"
}
```

---

# Running with Docker

## 1. Build the Docker image

```
docker build -t rag-flask-app .
```

---

## 2. Run the container

```
docker run -p 5000:5000 --env-file .env rag-flask-app
```

The application will be available at:

```
http://localhost:5000
```

---

# Stopping the Container

Find the container:

```
docker ps
```

Stop it:

```
docker stop <container_id>
```

---

# Technologies Used

* Python
* Flask
* LangChain
* FAISS
* HuggingFace Embeddings
* Groq LLM API
* Docker

---

# Future Improvements

* Streaming responses
* Persistent vector database
* Multi-document support
* Authentication
* Better context formatting for RAG
* UI for document interaction

---

# License

MIT License
