from rag_chain import load_rag_chain

def main():
    qa_chain = load_rag_chain()
    print("📊 CSV RAG Chatbot Ready (type 'exit' to quit)\n")

    while True:
        query = input("User: ")
        if query.lower() == "exit":
            break

        response = qa_chain.run(query)
        print(f"Bot: {response}\n")

if __name__ == "__main__":
    main()
