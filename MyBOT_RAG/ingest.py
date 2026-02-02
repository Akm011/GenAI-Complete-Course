from dotenv import load_dotenv
load_dotenv()

import pandas as pd
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

import os

load_dotenv()

DATA_PATH = "Data"
VECTOR_DB_PATH = "Vectorstore"

def csv_to_text(file_path, source):
    df = pd.read_csv(file_path)
    docs = []
    for _, row in df.iterrows():
        text = f"Source: {source}\n"
        for col in df.columns:
            text += f"{col}: {row[col]}\n"
        docs.append(text)
    return docs

def main():
    documents = []

    documents.extend(csv_to_text(f"{DATA_PATH}/holdings.csv", "holdings.csv"))
    documents.extend(csv_to_text(f"{DATA_PATH}/trades.csv", "trades.csv"))

    embeddings = OpenAIEmbeddings()

    vectorstore = FAISS.from_texts(
        texts=documents,
        embedding=embeddings
    )

    vectorstore.save_local(VECTOR_DB_PATH)
    print("✅ Vector store created successfully")

if __name__ == "__main__":
    main()
