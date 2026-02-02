import os
import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from huggingface_hub import InferenceClient
from langchain_huggingface import (
    HuggingFaceEndpoint,
    HuggingFaceEmbeddings
)

# -----------------------------
# CONFIG
# -----------------------------
load_dotenv()

DATA_FOLDER = "data"
FALLBACK_RESPONSE = "Sorry can not find the answer"
# HF_MODEL = "mistralai/Mistral-7B-Instruct-v0.2"
# HF_MODEL = "HuggingFaceH4/zephyr-7b-beta"
HF_MODEL = "google/flan-t5-large"


# -----------------------------
# LOAD FILES
# -----------------------------
def load_files():
    documents = []

    for file in os.listdir(DATA_FOLDER):
        path = os.path.join(DATA_FOLDER, file)

        if file.endswith(".csv"):
            df = pd.read_csv(path)
            documents.append(
                Document(
                    page_content=df.to_string(index=False),
                    metadata={"source": file}
                )
            )
        elif file.endswith(".xlsx"):
            df = pd.read_excel(path)
            documents.append(
                Document(
                    page_content=df.to_string(index=False),
                    metadata={"source": file}
                )
            )

    return documents

# -----------------------------
# VECTOR STORE
# -----------------------------
def generate_answer(prompt: str) -> str:
    client = InferenceClient(
        model="google/flan-t5-large",
        token=os.getenv("HUGGINGFACEHUB_API_TOKEN"),
        provider="hf-inference"   # 🔥 CRITICAL FIX
    )

    response = client.text_generation(
        prompt=prompt,
        max_new_tokens=256,
        temperature=0
    )

    return response.strip()


def build_vector_store(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150
    )

    chunks = splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    return FAISS.from_documents(chunks, embeddings)

# -----------------------------
# CHAT LOGIC (STRICT)
# -----------------------------
def ask_question(vectorstore, question):
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    docs = retriever.invoke(question)

    if not docs:
        return FALLBACK_RESPONSE

    context = "\n\n".join(doc.page_content for doc in docs)

    prompt = f"""
            You are a financial data assistant.

            RULES:
            - Use ONLY the data provided below
            - Do NOT guess or infer
            - If the answer is not explicitly found, respond exactly:
            "{FALLBACK_RESPONSE}"

            DATA:
            {context}

            QUESTION:
            {question}

            ANSWER:
            """

    response = generate_answer(prompt)

    if FALLBACK_RESPONSE.lower() in response.lower():
        return FALLBACK_RESPONSE

    return response

# def ask_question(vectorstore, question):
#     retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
#     docs = retriever.invoke(question)

#     if not docs:
#         return FALLBACK_RESPONSE

#     context = "\n\n".join(doc.page_content for doc in docs)

#     llm = HuggingFaceEndpoint(
#         repo_id=HF_MODEL,
#         temperature=0,
#         max_new_tokens=300,
#         huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN")
#     )

#     prompt = f"""
# You are a financial data assistant.

# RULES:
# - Use ONLY the data provided below
# - Do NOT guess or infer
# - If the answer is not explicitly found, respond exactly:
# "{FALLBACK_RESPONSE}"

# DATA:
# {context}

# QUESTION:
# {question}

# ANSWER:
# """

#     response = llm.invoke(prompt).strip()

#     if FALLBACK_RESPONSE.lower() in response.lower():
#         return FALLBACK_RESPONSE

#     return response

# -----------------------------
# STREAMLIT UI
# -----------------------------
st.set_page_config(page_title="Fund Data Chatbot (HF)")
st.title(" Fund & Trade Chatbot (Hugging Face)")

if "vectorstore" not in st.session_state:
    with st.spinner("Indexing data..."):
        docs = load_files()
        st.session_state.vectorstore = build_vector_store(docs)

question = st.text_input("Ask a question based on the uploaded files:")

if st.button("Ask"):
    if question.strip():
        answer = ask_question(st.session_state.vectorstore, question)
        st.write("### Answer")
        st.write(answer)
