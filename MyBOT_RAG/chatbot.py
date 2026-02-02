import os
import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI

# -----------------------------
# CONFIG
# -----------------------------
load_dotenv()

FALLBACK_RESPONSE = "Sorry can not find the answer"

HOLDINGS_FILE = "./Data/holdings.csv"
TRADES_FILE = "./Data/trades.csv"

EMBED_MODEL = "text-embedding-3-small"
CHAT_MODEL = "gpt-4o-mini"

# -----------------------------
# LOAD FILES
# -----------------------------
def load_files():
    documents = []

    holdings_df = pd.read_csv(HOLDINGS_FILE)
    trades_df = pd.read_csv(TRADES_FILE)

    documents.append(
        Document(
            page_content=holdings_df.to_string(index=False),
            metadata={"source": "holdings.csv"}
        )
    )

    documents.append(
        Document(
            page_content=trades_df.to_string(index=False),
            metadata={"source": "trades.csv"}
        )
    )

    return documents

# -----------------------------
# VECTOR STORE
# -----------------------------
def build_vector_store(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150
    )

    chunks = splitter.split_documents(documents)

    embeddings = OpenAIEmbeddings(model=EMBED_MODEL)

    return FAISS.from_documents(chunks, embeddings)

# -----------------------------
# OPENAI ANSWER GENERATION
# -----------------------------
def generate_answer(prompt: str) -> str:
    llm = ChatOpenAI(
        model=CHAT_MODEL,
        temperature=0
    )

    response = llm.invoke(prompt).content.strip()
    return response

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

        RULES (MANDATORY):
        - Answer ONLY using the DATA below
        - Do NOT use outside knowledge
        - Do NOT guess or infer
        - If the answer is NOT explicitly present, respond EXACTLY with:
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

# -----------------------------
# STREAMLIT UI
# -----------------------------
st.set_page_config(page_title="Holdings & Trades Chatbot")
st.title("📊 Holdings & Trades Data Chatbot")

if "vectorstore" not in st.session_state:
    with st.spinner("Loading and indexing data..."):
        docs = load_files()
        st.session_state.vectorstore = build_vector_store(docs)

question = st.text_input("Ask a question based on holdings & trades data:")

if st.button("Ask"):
    if question.strip():
        answer = ask_question(st.session_state.vectorstore, question)
        st.write("### Answer")
        st.write(answer)
