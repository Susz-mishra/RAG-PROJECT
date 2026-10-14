import os
from pathlib import Path

import streamlit as st
import chromadb
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings


# --------------------------------------------------
# 1. LOAD ENVIRONMENT VARIABLES
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent

load_dotenv(
    dotenv_path=BASE_DIR / ".env",
    override=True
)


# --------------------------------------------------
# 2. STREAMLIT PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="My RAG Assistant",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 RAG Document Assistant")
st.caption(
    "Ask questions about your documents, "
    "or use the LLM when relevant information is unavailable."
)


# --------------------------------------------------
# 3. CONFIGURATION
# --------------------------------------------------

VECTOR_STORE_PATH = str(BASE_DIR / "data" / "vector_store")
COLLECTION_NAME = "pdf_documents"

# IMPORTANT:
# This MUST be the same embedding model used to create
# your existing Chroma vector store.
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Starting threshold only. Tune using your actual distances.
MAX_DISTANCE = 0.9
TOP_K = 3


# --------------------------------------------------
# 4. LOAD COMPONENTS
# --------------------------------------------------

@st.cache_resource
def load_components():

    api_key = os.getenv("GROQ_API_KEY")

    if not api_key:
        raise ValueError(
            "GROQ_API_KEY is missing. "
            "Check the variable name in your .env file."
        )

    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL
    )

    client = chromadb.PersistentClient(
        path=VECTOR_STORE_PATH
    )

    collection = client.get_collection(
        name=COLLECTION_NAME
    )

    llm = ChatGroq(
        model="openai/gpt-oss-120b",
        groq_api_key=api_key,
        temperature=0.1,
        max_tokens=1024
    )

    return embeddings, collection, llm


# --------------------------------------------------
# 5. RETRIEVE DOCUMENTS AND GENERATE ANSWER
# --------------------------------------------------

def answer_question(question, embeddings, collection, llm):

    # Generate query embedding
    query_embedding = embeddings.embed_query(question)

    # Retrieve nearest chunks from Chroma
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=TOP_K,
        include=[
            "documents",
            "metadatas",
            "distances"
        ]
    )

    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0]

    # Keep only chunks within the configured distance threshold
    relevant_chunks = [
        (doc, meta, distance)
        for doc, meta, distance in zip(
            documents,
            metadatas,
            distances
        )
        if (
            doc
            and doc.strip()
            and distance is not None
            and distance <= MAX_DISTANCE
        )
    ]

    # Build context only from accepted chunks
    context = "\n\n".join(
        doc for doc, _, _ in relevant_chunks
    )

    # --------------------------------------------------
    # PATH A: RELEVANT DOCUMENT CONTEXT FOUND
    # --------------------------------------------------

    if context:

        prompt = f"""
You are a helpful document question-answering assistant.

Answer the user's question using the supplied document context.

Rules:
- Use the context as the source for document-specific claims.
- If the context does not contain the answer, clearly say
  that the documents do not provide enough information.
- Do not invent document contents.
- Give a clear, direct answer.

DOCUMENT CONTEXT:
{context}

QUESTION:
{question}

ANSWER:
"""

        answer_source = "Document context (RAG)"

    # --------------------------------------------------
    # PATH B: NO CHUNKS PASSED THE DISTANCE THRESHOLD
    # --------------------------------------------------

    else:

        prompt = f"""
You are a helpful AI assistant.

The document search did not find sufficiently close
context for this question.

Answer the question using your general knowledge.
Do not claim that your answer comes from the user's documents.
If you are uncertain, say so clearly.

QUESTION:
{question}

ANSWER:
"""

        answer_source = "LLM general knowledge (fallback)"

    # Call the LLM
    response = llm.invoke(prompt)

    return response.content, relevant_chunks, answer_source, distances


# --------------------------------------------------
# 6. STREAMLIT USER INTERFACE
# --------------------------------------------------

question = st.text_input(
    "Ask a question",
    placeholder="For example: What is Python?"
)

if st.button("Get Answer", type="primary"):

    if not question.strip():

        st.warning("Please enter a question.")

    else:

        try:

            with st.spinner(
                "Searching documents and generating an answer..."
            ):

                embeddings, collection, llm = load_components()

                answer, retrieved_chunks, answer_source, distances = (
                    answer_question(
                        question,
                        embeddings,
                        collection,
                        llm
                    )
                )

            st.subheader("💬 Answer")
            st.write(answer)

            st.caption(f"Answer source: {answer_source}")

            # Show accepted chunks
            if retrieved_chunks:

                st.subheader("📄 Relevant Document Chunks")

                for i, (document, metadata, distance) in enumerate(
                    retrieved_chunks,
                    start=1
                ):

                    with st.expander(f"Source chunk {i}"):

                        st.write(document)

                        st.caption(
                            f"Chroma distance: {distance:.4f}"
                        )

                        st.json(metadata or {})

            else:

                st.info(
                    "No chunks passed the relevance threshold. "
                    "The answer was generated using the LLM fallback."
                )

            # Optional diagnostics to help tune retrieval
            with st.expander("🔍 Retrieval diagnostics"):

                if distances:
                    st.write("Retrieved distances:", distances)
                    st.write("Configured maximum distance:", MAX_DISTANCE)
                else:
                    st.write("No document distances were returned.")

        except Exception as error:

            st.error(f"Something went wrong: {error}")

            st.info(
                "Check the API key, vector-store path, collection name, "
                "embedding model, and installed packages."
            )