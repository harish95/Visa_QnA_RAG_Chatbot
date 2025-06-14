# app.py

import os
import streamlit as st
st.set_page_config(page_title="Israel Entry QnA Chatbot", page_icon="🛂")

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_groq import ChatGroq



# Load .env file
load_dotenv()
groq_key = os.getenv("GROQ_API_KEY")

# Step 1: Load vectorstore
@st.cache_resource
def load_rag_chain():
    persist_dir = "vectorstore"
    embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-large-en-v1.5")
    vectorstore = Chroma(
        embedding_function=embeddings,
        persist_directory=persist_dir
    )
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

    # Prompt and LLM
    system_prompt = (
        "You are an assistant for answering questions based on Israeli entry regulations. "
        "Use the retrieved context below to answer the question concisely. "
        "If the answer is not in the context, say 'I don’t know.'\n\n"
        "{context}"
    )
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}")
    ])

    llm = ChatGroq(groq_api_key=groq_key, model_name="Llama3-8b-8192")
    qa_chain = create_stuff_documents_chain(llm, prompt)
    return create_retrieval_chain(retriever, qa_chain)

rag_chain = load_rag_chain()

# Step 2: Streamlit UI
st.title("🛂 Israel Entry QnA Chatbot")
st.caption("Ask any question about entering Israel – rules, documents, exemptions, or contact info.")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

query = st.chat_input("Ask your question about Israel's entry rules...")

if query:
    with st.spinner("Searching..."):
        result = rag_chain.invoke({"input": "Represent this question for retrieval: " + query})

        # Prepare sources
        sources = "\n\n**Sources:**"
        for i, doc in enumerate(result.get("context", [])):
            url = doc.metadata.get("source", "Unknown")
            snippet = doc.page_content[:180].strip().replace("\n", " ")
            sources += f"\n{i+1}. [{url}] - \"{snippet}...\""

        full_response = result["answer"] + sources
        st.session_state.chat_history.append((query, full_response))

# Display conversation history
for user, bot in st.session_state.chat_history:
    with st.chat_message("user"):
        st.markdown(user)
    with st.chat_message("assistant"):
        st.markdown(bot)
