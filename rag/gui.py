import os
import streamlit as st
from pdfplumber import open as open_pdf
from transformers import BertTokenizer, BertModel
import torch
import numpy as np
import faiss
from langchain.vectorstores import FAISS
from chatbot import query_general_model, query_rag  # Ensure these functions are implemented

# Define PDF processing functions
def get_pdf_text(pdf_file):
    st.write("Extracting text from PDF...")
    with open_pdf(pdf_file) as pdf:
        text = "".join(page.extract_text() for page in pdf.pages)
    st.write("Text extraction complete.")
    return text

def get_text_chunks(text, chunk_size=1000, chunk_overlap=200):
    st.write("Chunking text...")
    text_chunks = []
    position = 0
    while position < len(text):
        start_index = max(0, position - chunk_overlap)
        end_index = position + chunk_size
        chunk = text[start_index:end_index]
        text_chunks.append(chunk)
        position = end_index - chunk_overlap
    st.write("Text chunking complete.")
    return text_chunks

def get_embeddings(texts, model, tokenizer):
    st.write("Generating embeddings...")
    embeddings = []
    for text in texts:
        inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=512)
        with torch.no_grad():
            outputs = model(**inputs)
        embedding = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
        embeddings.append(embedding)
    st.write("Embedding generation complete.")
    return np.array(embeddings)

def get_vectorstore(text_chunks):
    st.write("Initializing BERT Tokenizer and Model...")
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased')
    
    st.write("Generating vector store...")
    embeddings = get_embeddings(text_chunks, model, tokenizer)
    
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    
    def embedding_function(text):
        inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=512)
        with torch.no_grad():
            outputs = model(**inputs)
        return outputs.last_hidden_state.mean(dim=1).squeeze().numpy()

    docstore = {i: chunk for i, chunk in enumerate(text_chunks)}
    index_to_docstore_id = {i: i for i in range(len(text_chunks))}

    vector_store = FAISS(
        embedding_function=embedding_function,
        docstore=docstore,
        index_to_docstore_id=index_to_docstore_id,
        index=index
    )
    st.write("Vector store initialization complete.")
    return vector_store

# Define the handle_query function
def handle_query():
    try:
        model = selected_model if selected_model else "llama3.1"
        
        if st.session_state.vector_store:
            # Use the vector store for context-aware querying
            response = query_rag(st.session_state.query_input, st.session_state.vector_store, model)
        else:
            response = query_general_model(st.session_state.query_input, model)
        
        st.session_state.messages.append({"role": "user", "content": st.session_state.query_input})
        st.session_state.messages.append({"role": "assistant", "content": response})
        st.session_state.query_input = ""  # Clear input field

    except Exception as e:
        st.error(f"Error: {e}")

# Streamlit app layout
def main():
    st.set_page_config(page_title="CHAT DAT", page_icon=":books:")
    
    # Add custom CSS to the Streamlit app
    st.markdown('<link rel="stylesheet" href="style.css">', unsafe_allow_html=True)
    
    # Add custom HTML to display title with gradient and logo
    st.markdown("""
        <div class="title-container">
            <img src="logo.png" class="logo" alt="Logo">
            <div class="title-text">CHAT DAT</div>
        </div>
    """, unsafe_allow_html=True)

    # Initialize session state
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "text_chunks" not in st.session_state:
        st.session_state.text_chunks = []
    if "query_input" not in st.session_state:
        st.session_state.query_input = ""
    if "vector_store" not in st.session_state:
        st.session_state.vector_store = None

    # Display chat messages from history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    st.sidebar.subheader("Settings")
    model_options = ["llama3.1", "meta-llama/Meta-Llama-3.1-8B", "another_model_2"]
    global selected_model
    selected_model = st.sidebar.selectbox("Select Model", model_options, index=model_options.index("llama3.1"))
    temperature = st.sidebar.slider('Temperature', min_value=0.01, max_value=1.0, value=0.1, step=0.01)
    top_p = st.sidebar.slider('Top P', min_value=0.01, max_value=1.0, value=0.9, step=0.01)
    max_length = st.sidebar.slider('Max Length', min_value=32, max_value=128, value=120, step=8)

    st.sidebar.header("Upload PDF")
    uploaded_file = st.sidebar.file_uploader("", type="pdf")

    if uploaded_file is not None:
        temp_pdf_path = "temp_uploaded_pdf.pdf"
        with open(temp_pdf_path, "wb") as f:
            f.write(uploaded_file.read())

        try:
            # Extract text and generate chunks
            raw_text = get_pdf_text(uploaded_file)
            st.session_state.text_chunks = get_text_chunks(raw_text)
            
            # Generate vector store
            st.session_state.vector_store = get_vectorstore(st.session_state.text_chunks)
            
            # Add PDF processing info to chat history
            st.session_state.messages.append({"role": "system", "content": "PDF content has been processed and vector store created."})
        finally:
            os.remove(temp_pdf_path)

    # React to user input
    if prompt := st.chat_input("Type your message here..."):
        st.session_state.query_input = prompt
        handle_query()

if __name__ == '__main__':
    main()
