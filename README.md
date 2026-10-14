# 📚 Retrieval-Augmented Generation (RAG) Pipeline

A document-based Question Answering system that combines semantic search with Large Language Models (LLMs) to generate context-aware responses from uploaded documents.

This project implements a Retrieval-Augmented Generation (RAG) pipeline using Python, vector embeddings, ChromaDB, and Groq-hosted LLMs.

## 🚀 Project Overview

Large Language Models can generate useful answers, but they may not know the information contained in a user's private documents. RAG addresses this limitation by retrieving relevant document content and providing it to an LLM as context before generating an answer.

This project processes PDF and text documents, converts their content into embeddings, stores the embeddings in a vector database, retrieves relevant information for a query, and generates a response using an LLM.

## ✨ Features

- **Document Loading:** Processes PDF and text documents.
- **Text Chunking:** Splits document content into smaller chunks for efficient retrieval.
- **Text Embeddings:** Converts text into numerical vector representations.
- **Vector Storage:** Uses ChromaDB to store and retrieve document embeddings.
- **Semantic Retrieval:** Searches for document chunks relevant to a user's query.
- **LLM Integration:** Uses Groq-hosted language models to generate responses.
- **Context-Aware Generation:** Supplies retrieved document content to the LLM to answer questions.
- **Fallback Responses:** Supports answering from the LLM's general knowledge when relevant document context is unavailable, subject to the configured retrieval logic.

## 🛠️ Technology Stack

| Technology | Purpose |
|---|---|
| Python | Core programming language |
| Jupyter Notebook | Development and experimentation |
| ChromaDB | Vector database |
| Text Embeddings | Semantic representation of documents |
| Groq API | Access to hosted language models |
| LangChain Groq | LLM integration |
| PDF and text processing libraries | Document ingestion |

## 🔄 How It Works

1. **Document ingestion:** Load source documents such as PDFs and text files.
2. **Text preprocessing:** Extract and split the document text into manageable chunks.
3. **Embedding generation:** Convert the chunks into vector embeddings.
4. **Vector storage:** Store the embeddings and associated text in ChromaDB.
5. **Query processing:** Convert the user's question into an embedding.
6. **Similarity search:** Retrieve relevant document chunks from the vector database.
7. **Response generation:** Provide the retrieved context to the LLM to generate an answer.
8. **Fallback handling:** If no sufficiently relevant context is found, optionally generate a response using the LLM's general knowledge.

## 📂 Project Structure

```text
RAG/
├── RAG-PIPELINNE.ipynb
├── .gitignore
├── data/
│   ├── python.txt
│   ├── pdfs/
│   │   ├── Research 1.pdf
│   │   └── Research 2.pdf
│   └── vector_store/       # Generated locally; not committed
└── README.md
```

The vector-store directory is generated locally and excluded from version control. Your actual folder contents may differ depending on your configuration.

## ⚙️ Installation and Setup

### 1. Clone the repository

```bash
git clone https://github.com/Susz-mishra/RAG-PROJECT.git
cd RAG-PROJECT
```

### 2. Create and activate a Conda environment (optional)

```bash
conda create -n rag_env python=3.11
conda activate rag_env
```

If you already have a compatible Python environment, you can use it instead.

### 3. Install the required dependencies

Install the packages used by your notebook. For example:

```bash
pip install jupyter chromadb langchain-core langchain-groq pypdf pymupdf
```

Install any additional embedding or document-processing dependencies required by your implementation.

### 4. Configure the Groq API key

Obtain an API key from [Groq Console](https://console.groq.com/keys).

On Windows Anaconda Prompt, set the key for the current terminal session:

```bat
set GROQ_API_KEY=your_actual_api_key
```

Replace the placeholder with your own key. Never commit your actual API key to GitHub.

The notebook should read the key from the environment:

```python
import os

api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    raise ValueError("GROQ_API_KEY is not configured")
```

If you launch Jupyter from this terminal, the notebook can inherit the environment variable.

### 5. Run the notebook

```bash
jupyter notebook
```

Open `RAG-PIPELINNE.ipynb` and execute the cells in order, ensuring that the required documents are available in the configured data directory.

## 💬 Example Usage

A user can ask a question about the documents:

```python
query = "What is encoder-decoder?"

answer = generate_output(
    query,
    retriever,
    llm
)

print(answer)
```

The response depends on the retrieved document context, the configured similarity threshold, and the behavior of the selected LLM.

## 🔐 Security and Best Practices

- Keep API keys in environment variables rather than hardcoding them.
- Exclude `.env` files from Git using `.gitignore`.
- Do not commit private documents or sensitive data.
- Exclude generated vector-store files unless there is a specific reason to share them.
- Never publish API keys in notebook outputs, screenshots, or repository history.

## 🎯 Learning Outcomes

Through this project, I explored:

- Retrieval-Augmented Generation architecture.
- Document processing and text chunking.
- Embedding generation and semantic search.
- Vector database operations using ChromaDB.
- Integration of hosted LLM APIs.
- Context construction and retrieval-based response generation.
- Environment-variable management for API credentials.

## 🔮 Future Improvements

- Add a web interface using Streamlit or FastAPI.
- Display retrieved source passages alongside generated answers.
- Improve retrieval quality using hybrid search and reranking.
- Add automated evaluation for retrieval relevance and answer quality.
- Support document uploads and multiple document collections.

## 👨‍💻 Author

** Anuj Kumar Mishra**

Computer Science Undergraduate | AI/ML & Generative AI

- GitHub: [Susz-mishra](https://github.com/Susz-mishra)
- Project Repository: [RAG-PROJECT](https://github.com/Susz-mishra/RAG-PROJECT)

---

*Built as a hands-on project to explore document retrieval, vector databases, and LLM-powered question answering.*
