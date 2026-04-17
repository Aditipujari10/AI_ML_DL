


import streamlit as st
import os

from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

from langchain_groq import ChatGroq


# ---------------- UI ----------------
st.title("🌐 Website Chatbot (RAG)")
st.write("Ask questions from any website")

# Inputs
groq_api = st.text_input("Enter Groq API Key", type="password")
url = st.text_input("Enter Website URL")

# ---------------- FUNCTIONS ----------------
def load_website(url):
    loader = WebBaseLoader(url)
    return loader.load()

def split_docs(docs):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=200
    )
    return splitter.split_documents(docs)

def create_db(chunks):
    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2"
    )
    return FAISS.from_documents(chunks, embeddings)

def get_retriever(db):
    return db.as_retriever(search_kwargs={"k": 5})

def load_llm(api_key):
    return ChatGroq(
        model="llama-3.1-8b-instant",
        temperature=0,
        groq_api_key=api_key
    )

def build_chain(retriever, llm):
    prompt = ChatPromptTemplate.from_template(
        """Answer ONLY from the context.
If not found, say "Not available on website".

Context:
{context}

Question:
{question}

Answer:"""
    )

    return (
        {
            "context": retriever,
            "question": RunnablePassthrough()
        }
        | prompt
        | llm
    )


# ---------------- LOAD WEBSITE ----------------
if st.button("Load Website"):
    if not url or not groq_api:
        st.warning("Please enter URL and API key")
    else:
        with st.spinner("Processing website..."):
            try:
                docs = load_website(url)
                chunks = split_docs(docs)
                db = create_db(chunks)
                retriever = get_retriever(db)
                llm = load_llm(groq_api)

                st.session_state.chain = build_chain(retriever, llm)

                st.success("Website loaded successfully!")

            except Exception as e:
                st.error(f"Error: {e}")


# ---------------- CHAT ----------------
if "chain" in st.session_state:
    query = st.text_input("Ask your question")

    if query:
        response = st.session_state.chain.invoke(query)

        st.write("### Answer:")
        st.write(response.content)
