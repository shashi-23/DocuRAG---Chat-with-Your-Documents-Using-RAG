# 🧠 DocuRAG - Chat with Your Documents Using RAG

**DocuRAG** is an intelligent document chat assistant powered by **RAG (Retrieval-Augmented Generation)**. Upload any PDF, DOCX, or TXT file, and start asking questions — the answers come **strictly from your document**, not hallucinations.

Built with **LangChain**, **Chroma**, and **OpenAI GPT**, all wrapped in a clean, easy-to-use **Streamlit** app.

---

## 🚀 Features

- 🔍 **RAG-Based Answering**: Combines document retrieval + GPT-based generation.
- 📄 **Multi-format Uploads**: Supports PDF, DOCX, and TXT files.
- 🧠 **Contextual Summaries**: Generate quick document overviews.
- 💬 **Conversational UI**: Chat with your document like talking to a colleague.
- 📚 **No Hallucinations**: GPT is grounded by your actual document content.
- 🧠 **Conversational Memory**: (Optional) Store and recall previous turns.

---

## 🛠 Tech Stack

| Component         | Description                                 |
|------------------|---------------------------------------------|
| **Streamlit**     | Frontend interface                          |
| **LangChain**     | Document loaders, RAG chain, memory         |
| **Chroma DB**     | Local vector database for document chunks   |
| **OpenAI GPT-4o** | LLM for response generation                 |
| **OpenAI Embeddings** | Embeds chunks for semantic search    |

---

## Demo

<img width="862" height="436" alt="image" src="https://github.com/user-attachments/assets/1f4d0b84-8e41-4ff8-acf9-babdd4302048" />
