# RAG Document Chatbot (FAISS + OpenAI)

A minimal, job-ready Retrieval-Augmented Generation (RAG) project:
- Ingest `.txt` / `.pdf` documents
- Chunk text
- Create embeddings (OpenAI)
- Store vectors in FAISS
- Retrieve top-k chunks
- Answer with OpenAI Responses API using retrieved context

## Setup

### 1) Create environment + install
```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Mac/Linux:
# source .venv/bin/activate

pip install -r requirements.txt
```

### 2) Add your API key
Copy `.env.example` → `.env` and fill:
```bash
OPENAI_API_KEY=...
```

### 3) Put documents into `/data`
- Add .txt or .pdf files
- A sample file is already included.

## Run

### A) Ingest documents
```bash
python src/ingest.py
```

### B) Ask questions
```bash
python src/query.py "How many PTO days do employees get?"
python src/query.py "What is the remote work policy?"
```

## Notes
- If the answer is not in your docs, the assistant will say:
  "Not found in the provided documents."
- Tune chunk_size / overlap in `src/utils.py`.
