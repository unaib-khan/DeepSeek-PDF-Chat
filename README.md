RAG System Powered by DeepSeek-R1 LLM Model
Overview
This project is a Retrieval-Augmented Generation (RAG) System that allows users to upload PDF files, process their contents, and interact with them through conversational queries. The app leverages Groq's Mixtral-8x7b LLM model (DeepSeek-R1 variant) for advanced natural language understanding and answering questions with high accuracy. The backend features FAISS for vector search and OpenAI embeddings for efficient similarity-based retrieval.

Key Features
PDF Upload and Processing: Upload multiple PDF files, extract their text, and store it in a searchable FAISS vector database.
Conversational Querying: Ask natural language questions about the content of the uploaded PDFs.
Accurate Answers: The app uses the DeepSeek-R1 model from Groq, which provides detailed answers, or informs when the information is not available in the context.
RAG Workflow: Combines information retrieval with generative capabilities for seamless and reliable responses.
Streamlit Interface: A user-friendly interface for uploading files, processing text, and interacting with the system.
Technology Stack
Backend
LangChain: Framework for building LLM-powered workflows.
FAISS: Scalable vector database for similarity search.
Hugging face: Used for creating embeddings from the extracted text.
Groq LLM: High-performing language model for answering questions.
Frontend
Streamlit: Simplified app creation for interactive dashboards and interfaces.