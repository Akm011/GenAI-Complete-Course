import streamlit as st
from rag_chain import load_rag_chain

# Page config
st.set_page_config(
    page_title="CSV RAG Chatbot",
    page_icon="📊",
    layout="centered"
)

st.title("📊 CSV RAG Chatbot")
st.caption("Answers only from holdings.csv & trades.csv")

# Load RAG chain once
@st.cache_resource
def load_chain():
    return load_rag_chain()

qa_chain = load_chain()

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# User input
user_input = st.chat_input("Ask a question about your CSV data...")

if user_input:
    # Show user message
    st.session_state.messages.append(
        {"role": "user", "content": user_input}
    )
    with st.chat_message("user"):
        st.markdown(user_input)

    # Get response
    with st.spinner("Thinking..."):
        answer = qa_chain.run(user_input)

    # Show assistant response
    st.session_state.messages.append(
        {"role": "assistant", "content": answer}
    )
    with st.chat_message("assistant"):
        st.markdown(answer)
