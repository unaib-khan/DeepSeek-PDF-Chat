import streamlit as st
import subprocess
import sys

# Ensure PyPDF2 is installed
subprocess.check_call([sys.executable, "-m", "pip", "install", "PyPDF2"])

from PyPDF2 import PdfReader

from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
st.secrets("GROQ_API_KEY")

# Load Sentence Transformer model
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

def get_pdf_text(pdf_docs):
    """Extracts text from uploaded PDF files."""
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(text):
    """Splits extracted text into manageable chunks."""
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks):
    """Creates and saves a FAISS vector store from text chunks."""
    vector_store = FAISS.from_texts(text_chunks, embedding=embedding_model)  # ✅ Pass embedding model instance
    vector_store.save_local("faiss_index")

def get_conversational_chain():
    """Sets up a conversational chain using Groq LLM."""
    prompt_template = """
    Answer the question as detailed as possible from the provided context. If the answer is not in
    the provided context, just say, "answer is not available in the context." Do not provide incorrect answers.

    Context:
    {context}?

    Question:
    {question}

    Answer:
    """
    
    model = ChatGroq(
        temperature=0.3,
        model_name="deepseek-r1-distill-llama-70b",  # Using Mixtral model through Groq
        groq_api_key=os.getenv("GROQ_API_KEY")
    )
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

def user_input(user_question):
    """Handles user queries by retrieving answers from the vector store."""
    vector_store = FAISS.load_local("faiss_index", embeddings=embedding_model, allow_dangerous_deserialization=True)
    docs = vector_store.similarity_search(user_question, k=3)  # Use similarity_search()
    
    chain = get_conversational_chain()
    response = chain(
        {"input_documents": docs, "question": user_question},
        return_only_outputs=True
    )
    
    st.markdown(f"### Reply:\n{response['output_text']}")

def main():
    """Main function to run the Streamlit app."""
    st.set_page_config(page_title="Chat PDF", page_icon=":books:", layout="wide")
    st.title("Chat with PDF using Deepseek ")
    
    st.sidebar.header("Upload & Process PDF Files")
    
    with st.sidebar:
        pdf_docs = st.file_uploader("Upload your PDF files:", accept_multiple_files=True, type=["pdf"])
        if st.button("Submit & Process"):
            with st.spinner("Processing your files..."):
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                get_vector_store(text_chunks)
                st.success("PDFs processed and indexed successfully!")

    st.markdown(
        "### Ask Questions from Your PDF Files :mag:\n"
        "Once you upload and process your PDFs, type your questions below."
    )

    user_question = st.text_input("Enter your question:", placeholder="What do you want to know?")

    if user_question:
        with st.spinner("Fetching your answer..."):
            user_input(user_question)


if __name__ == "__main__":
    main()
