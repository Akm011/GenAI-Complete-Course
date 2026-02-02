from dotenv import load_dotenv
load_dotenv()

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain.chains import RetrievalQA


VECTOR_DB_PATH = "Vectorstore"

def load_rag_chain():
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.load_local(
        VECTOR_DB_PATH,
        embeddings,
        allow_dangerous_deserialization=True
    )

    llm = ChatOpenAI(
        model="gpt-4",
        temperature=0
    )

    prompt = PromptTemplate(
        template="""
            You are a financial data assistant.

            Rules:
            - Use ONLY the provided context.
            - DO NOT use external knowledge.
            - DO NOT guess.
            - If the answer is not explicitly available in the context, respond EXACTLY:
            "Sorry can not find the answer"

            Context:
            {context}

            Question:
            {question}

            Answer:
            """,
        input_variables=["context", "question"]
    )

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=vectorstore.as_retriever(search_kwargs={"k": 5}),
        chain_type="stuff",
        chain_type_kwargs={"prompt": prompt},
        return_source_documents=False
    )

    return qa_chain
