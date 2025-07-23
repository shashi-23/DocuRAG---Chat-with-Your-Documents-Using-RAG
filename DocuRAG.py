import streamlit as st
import os
from dotenv import load_dotenv, find_dotenv
from langchain.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain.schema import Document


# --- HELPER FUNCTIONS ---

def load_document(file_path):
    _, extension = os.path.splitext(file_path)
    try:
        if extension == '.pdf':
            from langchain.document_loaders import PyPDFLoader
            loader = PyPDFLoader(file_path)
        elif extension == '.docx':
            from langchain.document_loaders import Docx2txtLoader
            loader = Docx2txtLoader(file_path)
        elif extension == '.txt':
            from langchain.document_loaders import TextLoader
            loader = TextLoader(file_path)
        else:
            st.error('Unsupported file format!')
            return None
        return loader.load()
    except Exception as e:
        st.error(f"Failed to load document: {e}")
        return None


def chunk_data(data, chunk_size=512):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=0)
    return text_splitter.split_documents(data)


def create_embeddings_chroma(chunks):
    embeddings = OpenAIEmbeddings(model='text-embedding-3-small')
    vector_store = Chroma.from_documents(chunks, embeddings)
    return vector_store


def get_relevant_context(question, retriever, max_docs=4):
    """Retrieve and strictly filter relevant documents for the user's query."""
    docs = retriever.get_relevant_documents(question)
    if not docs:
        return []

    question_words = set(question.lower().split())
    filtered_docs = [doc for doc in docs if any(word in doc.page_content.lower() for word in question_words)]

    return filtered_docs[:max_docs] if filtered_docs else docs[:1]


def generate_answer(context_docs, question):
    """Generate an answer strictly from document context."""
    context_text = "\n\n".join(doc.page_content for doc in context_docs)

    prompt = f"""
You are an expert assistant. Use ONLY the context below to answer the question.
If the answer is not in the context, respond: "I don't know based on the document."

---
CONTEXT:
{context_text}
---
QUESTION: {question}
---
ANSWER:
""".strip()

    llm = ChatOpenAI(model_name='gpt-4o-mini', temperature=0)
    return llm.invoke(prompt).content


def generate_summary(raw_data):
    """Summarize the document based on the first few pages."""
    initial_text = "\n\n".join([doc.page_content for doc in raw_data[:4]])
    prompt = f"""
You are given the beginning of a document. Write a concise summary capturing the main purpose,
topics, and key findings or sections.

---
DOCUMENT START:
{initial_text}
---
SUMMARY:
""".strip()

    llm = ChatOpenAI(model_name='gpt-4o-mini', temperature=0)
    return llm.invoke(prompt).content


# --- STREAMLIT APP ---

def main():
    load_dotenv(find_dotenv(), override=True)

    st.set_page_config(page_title="Chat With Your Documents", page_icon="📄")
    st.header("📄 DocuRAG - Chat with Your Documents Using RAG")

    if 'vector_store' not in st.session_state:
        st.session_state.vector_store = None
    if 'raw_data' not in st.session_state:
        st.session_state.raw_data = None
    if 'retriever' not in st.session_state:
        st.session_state.retriever = None
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []

    with st.sidebar:
        st.subheader("Upload a Document")
        uploaded_file = st.file_uploader("Upload a PDF, DOCX, or TXT file", type=['pdf', 'docx', 'txt'])

        if st.button("Process Document"):
            if uploaded_file:
                with st.spinner("Processing..."):
                    os.makedirs("temp", exist_ok=True)
                    file_path = os.path.join("temp", uploaded_file.name)

                    with open(file_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())

                    docs = load_document(file_path)
                    if docs:
                        chunks = chunk_data(docs)
                        vector_store = create_embeddings_chroma(chunks)
                        retriever = vector_store.as_retriever(search_type="mmr", search_kwargs={"k": 12})

                        st.session_state.raw_data = docs
                        st.session_state.vector_store = vector_store
                        st.session_state.retriever = retriever
                        st.success("Document processed successfully!")
                    os.remove(file_path)
            else:
                st.warning("Please upload a document.")

        if st.button("Reset Chat"):
            st.session_state.chat_history = []
            st.session_state.raw_data = None
            st.session_state.vector_store = None
            st.session_state.retriever = None
            st.success("Chat reset.")

    # Display previous chat
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Chat input
    if prompt := st.chat_input("Ask a question or request a summary..."):
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        if not st.session_state.retriever:
            st.warning("Please upload and process a document first.")
        else:
            with st.spinner("Thinking..."):
                is_summary = any(word in prompt.lower() for word in ["summary", "summarize", "overview", "about", "contents"])
                if is_summary:
                    answer = generate_summary(st.session_state.raw_data)
                else:
                    context = get_relevant_context(prompt, st.session_state.retriever)
                    if context:
                        answer = generate_answer(context, prompt)
                    else:
                        answer = "I don't know based on the document."

                st.session_state.chat_history.append({"role": "assistant", "content": answer})
                with st.chat_message("assistant"):
                    st.markdown(answer)


if __name__ == '__main__':
    main()
