from sklearn import pipeline
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import numpy as np
import torch
import os

# Set the Hugging Face token
os.environ['HF_HOME'] = 'path/to/your/hf_cache'  # Optional: Define cache directory if needed
os.environ['HF_TOKEN'] = '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'

PROMPT_TEMPLATE = """
Answer the question based only on the following context:

{context}

---

Answer the question based on the above context: {question}
"""

# Function to load and return the model and tokenizer based on the model_name
def load_model(model_name: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=os.getenv('HF_TOKEN'))
    model = AutoModelForCausalLM.from_pretrained(model_name, use_auth_token=os.getenv('HF_TOKEN'))
    return tokenizer, model

# Function to handle the prompt and get a response from the model
def handle_prompt(query_text: str, context_text: str, tokenizer, model):
    # Create the prompt
    prompt = PROMPT_TEMPLATE.format(context=context_text, question=query_text)
    
    # Tokenize the prompt
    inputs = tokenizer(prompt, return_tensors='pt', truncation=True, max_length=512)
    
    # Use the tokenizer and model to get a response
    pipe = pipeline(
        'text-generation',
        model=model,
        tokenizer=tokenizer,
        device=0 if torch.cuda.is_available() else -1
    )
    
    # Set max_new_tokens to control the length of the output
    response = pipe(prompt, max_new_tokens=150, num_return_sequences=1)[0]['generated_text']
    
    return response

def query_rag(query_text: str, vector_store, model_name: str):
    # Load the model and tokenizer
    tokenizer, model = load_model(model_name)
    
    # Retrieve the query embedding
    query_embedding = vector_store.embedding_function(query_text)
    
    # Perform similarity search
    distances, indices = vector_store.index.search(np.array([query_embedding]), k=5)  # Retrieve top 5 most similar chunks
    
    # Get the relevant chunks from the docstore
    relevant_chunks = [vector_store.docstore[i] for i in indices[0]]
    
    # Join relevant chunks to form the context
    context_text = ' '.join(relevant_chunks)
    
    # Handle the prompt with the loaded model
    response = handle_prompt(query_text, context_text, tokenizer, model)
    
    return response

def query_general_model(query_text: str, model_name: str):
    # General model context is empty
    context_text = ""
    
    # Load the model and tokenizer
    tokenizer, model = load_model(model_name)
    
    # Handle the prompt with the loaded model
    response = handle_prompt(query_text, context_text, tokenizer, model)
    
    return response
