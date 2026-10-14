
import os
import uuid
from pathlib import Path

import chromadb
import streamlit as st
from dotenv import load_dotenv
from groq import Groq
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer


# --------------------------------------------------
# 1. APP CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="RAG Document Assistant",
    page_icon="📚",
    layout="wide"
)

st.title("📚 RAG Document Assistant")
st.caption("Ask questions about your documents using RAG and Groq LLMs.")

BASE_DIR = Path(__file__).resolve().parent
VECTOR_STORE_PATH = str(BASE_DIR / "data" / "vector_store")
COLLECTION_NAME = "pdf_documents"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

TOP_K = 3
MAX_DISTANCE = 0.9

load_dotenv(BASE_DIR / ".env", override=True)


# --------------------------------------------------
# 2. LOAD API KEY
# --------------------------------------------------

def get_api_key():
    try:
        return st.secrets["GROQ_API_KEY"]
    except Exception:
        return os.getenv("GROQ_API_KEY")


# --------------------------------------------------
# 3. LOAD EMBEDDING MODEL AND CHROMA DATABASE
# --------------------------------------------------

@st.cache_resource
def load_components():
    api_key = get_api_key()

    if not api_key:
        raise ValueError(
            "GROQ_API_KEY is missing. Add it to Streamlit Cloud "
            "Settings > Secrets, or to your local .env file."
        )

    embedding_model = SentenceTransformer(EMBEDDING_MODEL)

    os.makedirs(VECTOR_STORE_PATH, exist_ok=True)

    client = chromadb.PersistentClient(
        path=VECTOR_STORE_PATH
    )

    # Show available collections in the app.
    collection_names = [
        item.name for item in client.list_collections()
    ]

    # Create the collection if it does not exist.
    # A newly created collection will be empty until PDFs are added.
    if COLLECTION_NAME in collection_names:
        collection = client.get_collection(
            name=COLLECTION_NAME
        )
    else:
        collection = client.create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"}
        )

    groq_client = Groq(api_key=api_key)

    return (
        embedding_model,
        client,
        collection,
        groq_client
    )


# --------------------------------------------------
# 4. READ AND SPLIT PDF TEXT
# --------------------------------------------------

def extract_pdf_chunks(uploaded_file):
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    reader = PdfReader(uploaded_file)

    pages = []

    for page_number, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""

        if text.strip():
            pages.append({
                "text": text,
                "page": page_number
            })

    if not pages:
        raise ValueError(
            "No readable text found in this PDF. "
            "Scanned PDFs may require OCR."
        )

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150
    )

    chunks = []

    for page in pages:
        for chunk_text in splitter.split_text(page["text"]):
            chunks.append({
                "text": chunk_text,
                "page": page["page"],
                "filename": uploaded_file.name
            })

    return chunks


# --------------------------------------------------
# 5. ADD PDF CHUNKS TO CHROMA
# --------------------------------------------------

def add_pdf_to_collection(
    uploaded_file,
    embedding_model,
    collection
):
    chunks = extract_pdf_chunks(uploaded_file)

    documents = [chunk["text"] for chunk in chunks]

    embeddings = embedding_model.encode(
        documents,
        normalize_embeddings=True
    ).tolist()

    ids = [str(uuid.uuid4()) for _ in chunks]

    metadatas = [
        {
            "source": chunk["filename"],
            "page": chunk["page"]
        }
        for chunk in chunks
    ]

    collection.add(
        ids=ids,
        documents=documents,
        embeddings=embeddings,
        metadatas=metadatas
    )

    return len(chunks)


# --------------------------------------------------
# 6. RETRIEVE RELEVANT DOCUMENT CHUNKS
# --------------------------------------------------

def retrieve_documents(
    query,
    embedding_model,
    collection,
    top_k=TOP_K
):
    if collection.count() == 0:
        return []

    query_embedding = embedding_model.encode(
        query,
        normalize_embeddings=True
    ).tolist()

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=min(top_k, collection.count()),
        include=[
            "documents",
            "metadatas",
            "distances"
        ]
    )

    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0]

    relevant_documents = []

    for document, metadata, distance in zip(
        documents,
        metadatas,
        distances
    ):
        if document and distance <= MAX_DISTANCE:
            relevant_documents.append({
                "document": document,
                "metadata": metadata or {},
                "distance": distance
            })

    return relevant_documents


# --------------------------------------------------
# 7. GENERATE ANSWER USING GROQ
# --------------------------------------------------

def generate_answer(query, context, groq_client):
    if context:
        system_prompt = """
You are a helpful document question-answering assistant.

Answer the user's question using the provided document context.
Do not invent facts or claim that the documents say something
they do not say.

If the context does not contain enough information to answer,
clearly state that the documents do not provide enough information.
"""

        user_prompt = f"""
DOCUMENT CONTEXT:
{context}

USER QUESTION:
{query}

Answer clearly and accurately using the document context.
"""

    else:
        system_prompt = """
You are a helpful AI assistant.
The user's documents did not provide sufficiently relevant context.
Answer using your general knowledge, and clearly mention uncertainty
when appropriate. Do not pretend that your answer came from a document.
"""

        user_prompt = query

    response = groq_client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=[
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": user_prompt
            }
        ],
        temperature=0.1,
        max_tokens=1024
    )

    return response.choices[0].message.content


# --------------------------------------------------
# 8. LOAD COMPONENTS
# --------------------------------------------------

try:
    (
        embedding_model,
        client,
        collection,
        groq_client
    ) = load_components()

except Exception as error:
    st.error(f"Could not initialize the application: {error}")
    st.stop()


# --------------------------------------------------
# 9. DATABASE DIAGNOSTICS
# --------------------------------------------------

with st.expander("🔧 Database diagnostics"):
    st.write("**Database path:**", VECTOR_STORE_PATH)

    available_collections = [
        item.name for item in client.list_collections()
    ]

    st.write("**Available collections:**", available_collections)
    st.write("**Active collection:**", COLLECTION_NAME)
    st.write("**Stored document chunks:**", collection.count())

    if collection.count() == 0:
        st.warning(
            "The active collection is empty. Upload a PDF below "
            "to add documents before asking document-specific questions."
        )


# --------------------------------------------------
# 10. PDF UPLOAD
# --------------------------------------------------

st.subheader("📄 Add documents")

uploaded_files = st.file_uploader(
    "Upload PDF files",
    type=["pdf"],
    accept_multiple_files=True
)

if st.button("Process uploaded PDFs", type="primary"):
    if not uploaded_files:
        st.warning("Please upload at least one PDF first.")
    else:
        total_chunks = 0

        progress = st.progress(0)

        for index, uploaded_file in enumerate(uploaded_files):
            try:
                with st.spinner(
                    f"Processing {uploaded_file.name}..."
                ):
                    chunk_count = add_pdf_to_collection(
                        uploaded_file,
                        embedding_model,
                        collection
                    )

                    total_chunks += chunk_count

                st.success(
                    f"{uploaded_file.name}: "
                    f"{chunk_count} chunks added."
                )

            except Exception as error:
                st.error(
                    f"Could not process {uploaded_file.name}: {error}"
                )

            progress.progress(
                (index + 1) / len(uploaded_files)
            )

        st.success(f"Finished processing. Added {total_chunks} chunks.")
        st.rerun()


# --------------------------------------------------
# 11. ASK QUESTIONS
# --------------------------------------------------

st.divider()
st.subheader("💬 Ask a question")

query = st.text_area(
    "Enter your question",
    placeholder="Example: What is Python?",
    height=100
)

if st.button("Get Answer"):
    if not query.strip():
        st.warning("Please enter a question.")

    else:
        try:
            with st.spinner("Searching documents and generating answer..."):
                results = retrieve_documents(
                    query,
                    embedding_model,
                    collection,
                    top_k=TOP_K
                )

                context = "\n\n".join(
                    item["document"] for item in results
                )

                answer = generate_answer(
                    query,
                    context,
                    groq_client
                )

            st.subheader("🤖 Answer")
            st.write(answer)

            if results:
                st.subheader("📚 Retrieved sources")

                for index, item in enumerate(results, start=1):
                    metadata = item["metadata"]

                    source = metadata.get("source", "Unknown source")
                    page = metadata.get("page", "Unknown")

                    with st.expander(
                        f"Source {index}: {source} — Page {page}"
                    ):
                        st.write(item["document"])
                        st.caption(
                            f"Chroma distance: {item['distance']:.4f}"
                        )

            else:
                st.info(
                    "No relevant document chunks passed the "
                    "similarity threshold. The answer uses the "
                    "LLM's general knowledge, not your documents."
                )

        except Exception as error:
            st.error(f"Something went wrong: {error}")
