import os
import csv
from pdfplumber import open as open_pdf
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
import torch

def get_pdf_text(pdf_file):
    """Extract text from a PDF file."""
    with open_pdf(pdf_file) as pdf:
        text = "".join(page.extract_text() for page in pdf.pages)
    if not text:
        print(f"Warning: No text extracted from {pdf_file}.")
    return text

def get_text_chunks(text, chunk_size=1000, chunk_overlap=200):
    """Split text into chunks."""
    text_chunks = []
    position = 0
    while position < len(text):
        start_index = max(0, position - chunk_overlap)
        end_index = position + chunk_size
        chunk = text[start_index:end_index]
        text_chunks.append(chunk)
        position = end_index - chunk_overlap
    return text_chunks

def derive_difficulty_from_question(question):
    """Determine difficulty level based on question length."""
    question_length = len(question.split())
    if question_length <= 10:
        return 'easy'
    elif question_length <= 20:
        return 'medium'
    else:
        return 'hard'

def derive_difficulty_from_answer(answer):
    """Determine difficulty level based on answer length."""
    answer_length = len(answer.split())
    if answer_length <= 10:
        return 'easy'
    elif answer_length <= 30:
        return 'medium'
    else:
        return 'hard'

def generate_qa_pairs(text_chunks, qa_model, qa_tokenizer, max_length=512):
    """Generate QA pairs from text chunks."""
    qa_pairs = []
    questions = [
        "What is the main idea of this text?",
        "What are the key details mentioned?",
        "What is the purpose of this section?",
        "What significant events or points are discussed?",
        "Are there any notable names or terms mentioned?",
        "What conclusions or insights can be drawn from this text?",
        "What questions would you have after reading this section?",
        "What context or background information is provided?",
        "How does this text relate to the overall topic or theme?",
        "What are the implications or impacts discussed in this text?"
    ]
    
    for idx, chunk in enumerate(text_chunks):
        print(f"Processing chunk {idx + 1}/{len(text_chunks)}")
        for question in questions:
            inputs = qa_tokenizer.encode_plus(question, chunk, return_tensors='pt', max_length=max_length, truncation=True)
            with torch.no_grad():
                outputs = qa_model(**inputs)
            answer_start = torch.argmax(outputs.start_logits)
            answer_end = torch.argmax(outputs.end_logits) + 1
            answer = qa_tokenizer.convert_tokens_to_string(
                qa_tokenizer.convert_ids_to_tokens(inputs.input_ids[0, answer_start:answer_end])
            )
            if answer.strip():
                difficulty_question = derive_difficulty_from_question(question)
                difficulty_answer = derive_difficulty_from_answer(answer)
                qa_pairs.append({
                    "chunk_index": idx,
                    "question": question,
                    "answer": answer,
                    "difficulty_from_questioner": difficulty_question,
                    "difficulty_from_answerer": difficulty_answer
                })
            else:
                print(f"No answer found for question: {question}")
    print(f"Generated {len(qa_pairs)} QA pairs.")
    return qa_pairs

def save_qa_pairs_to_file(qa_pairs, file_path):
    """Save all QA pairs to a CSV file."""
    if not qa_pairs:
        print("No QA pairs to save.")
    else:
        try:
            # Save QA pairs in a CSV file
            with open(file_path, 'w', newline='') as csvfile:
                fieldnames = ['ArticleTitle', 'Question', 'Answer', 'DifficultyFromQuestioner', 'DifficultyFromAnswerer', 'ArticleFile']
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                for pair in qa_pairs:
                    writer.writerow({
                        'ArticleTitle': pair['article_title'],
                        'Question': pair['question'],
                        'Answer': pair['answer'],
                        'DifficultyFromQuestioner': pair['difficulty_from_questioner'],
                        'DifficultyFromAnswerer': pair['difficulty_from_answerer'],
                        'ArticleFile': pair['article_file']
                    })
            print(f"QA pairs saved to {file_path}")
        except Exception as e:
            print(f"Error saving QA pairs to file: {e}")

def main(folder_path, output_file_path):
    """Process all PDF files in a folder and save QA pairs to a CSV file."""
    # Initialize QA model and tokenizer
    qa_tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased-distilled-squad")
    qa_model = AutoModelForQuestionAnswering.from_pretrained("distilbert-base-uncased-distilled-squad")
    
    # List to hold all QA pairs from all files
    all_qa_pairs = []

    for file_name in os.listdir(folder_path):
        if file_name.lower().endswith('.pdf'):
            pdf_file_path = os.path.join(folder_path, file_name)
            article_title = os.path.splitext(file_name)[0]  # Use the file name (without extension) as the title
            print(f"Processing file: {pdf_file_path}")
            
            # Extract and process text
            raw_text = get_pdf_text(pdf_file_path)
            if not raw_text:
                print(f"No text extracted from {pdf_file_path}. Skipping.")
                continue

            text_chunks = get_text_chunks(raw_text)
            print(f"Extracted {len(text_chunks)} text chunks from {pdf_file_path}.")

            # Generate question-answer pairs
            qa_pairs = generate_qa_pairs(text_chunks, qa_model, qa_tokenizer)
            
            # Append QA pairs with additional information
            for pair in qa_pairs:
                all_qa_pairs.append({
                    'article_title': article_title,
                    'question': pair['question'],
                    'answer': pair['answer'],
                    'difficulty_from_questioner': pair['difficulty_from_questioner'],
                    'difficulty_from_answerer': pair['difficulty_from_answerer'],
                    'article_file': article_title
                })

    # Save all QA pairs to a CSV file
    save_qa_pairs_to_file(all_qa_pairs, output_file_path)

if __name__ == '__main__':
    folder_path = 'pdfs'  # Replace with the path to your folder containing PDF files
    output_file_path = 'qa_dataset.csv'  # Replace with your desired output file path
    main(folder_path, output_file_path)
