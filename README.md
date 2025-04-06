# 💡 RAG System Powered by DeepSeek-R1 LLM Model

## 📝 Overview

This project is a **Retrieval-Augmented Generation (RAG)** System that enables users to upload PDF files, process their contents, and interact with them through conversational queries. It leverages **Groq's Mixtral-8x7b LLM model (DeepSeek-R1 variant)** for advanced natural language understanding and delivers highly accurate responses.

The backend utilizes **FAISS** for vector similarity search and **OpenAI embeddings** for effective retrieval, while the frontend is powered by **Streamlit** for an intuitive user experience.

---

## 🚀 Key Features

- 📄 **PDF Upload and Processing**  
  Upload multiple PDF files, extract their text, and store them in a searchable **FAISS vector database**.

- 💬 **Conversational Querying**  
  Ask natural language questions related to the uploaded documents and get contextually relevant answers.

- 🎯 **Accurate Answers**  
  Powered by **DeepSeek-R1**, the app provides detailed responses or politely informs when context is unavailable.

- 🔄 **RAG Workflow**  
  Seamlessly combines information retrieval with generative AI for reliable and informative answers.

- 🖥️ **Streamlit Interface**  
  A user-friendly web app interface to upload files, visualize processing steps, and interact with the chatbot.

---

## 🌐 Live Demo

👉 Try it out here: [https://deepseek-pdf-chat.streamlit.app/](https://deepseek-pdf-chat.streamlit.app/)

---

## 🛠️ Technology Stack

### 🔧 Backend
- **LangChain**: Framework for building LLM-powered pipelines.
- **FAISS**: Scalable vector store for similarity search.
- **Hugging Face**: Used for generating embeddings from document text.
- **Groq LLM (DeepSeek-R1)**: High-performance model for natural language understanding and response generation.

### 🎨 Frontend
- **Streamlit**: Lightweight and powerful tool to build the interactive web app.

---
