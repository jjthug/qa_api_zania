import json
from typing import List
from PyPDF2 import PdfReader
from fastapi import UploadFile, HTTPException
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OpenAIEmbeddings
import concurrent.futures
from dotenv import load_dotenv
import traceback

load_dotenv('.env')

embeddings = OpenAIEmbeddings()

# Function to extract text from PDF
def extract_text_from_pdf(file: UploadFile) -> str:
    try:
        pdf_reader = PdfReader(file.file)
        text = ""
        for page in pdf_reader.pages:
            extracted_text = page.extract_text()
            if extracted_text:
                text += extracted_text
        if not text:
            raise ValueError("No text extracted from PDF.")
        return text
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=400, detail=f"Error extracting text from PDF: {str(e)}")

# Function to extract text from JSON
def extract_text_from_json(file: UploadFile) -> str:
    try:
        json_content = json.loads(file.file.read().decode("utf-8"))
        return json.dumps(json_content, indent=2)
    except json.JSONDecodeError as e:
        traceback.print_exc()
        raise HTTPException(status_code=400, detail=f"Invalid JSON format: {str(e)}")
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=400, detail=f"Error reading JSON: {str(e)}")

# Function to extract questions from JSON file
def extract_questions_from_json(file: UploadFile) -> List[str]:
    try:
        questions_content = json.loads(file.file.read().decode("utf-8"))
        if not isinstance(questions_content, list):
            raise ValueError("Expected a list of questions in the JSON file.")
        return questions_content
    except json.JSONDecodeError as e:
        traceback.print_exc()
        raise HTTPException(status_code=400, detail=f"Invalid JSON format: {str(e)}")
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=400, detail=f"Error reading questions from JSON: {str(e)}")

# Function to chunk the document into smaller sections using Langchain's text splitter
def chunk_document(document: str, chunk_size: int = 500) -> List[Document]:
    try:
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=50)
        chunks = text_splitter.create_documents([document])
        if not chunks:
            raise ValueError("No chunks created from the document.")
        return chunks
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error splitting document into chunks: {str(e)}")

# Perform similarity search to find the top relevant chunks
def get_top_chunks(chunks: List[Document], query: str, top_k: int = 10) -> List[Document]:
    try:
        vector_store = FAISS.from_documents(chunks, embeddings)
        top_chunks = vector_store.similarity_search(query, top_k=top_k)
        if not top_chunks:
            raise ValueError(f"No relevant chunks found for query: {query}")
        return top_chunks
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error performing similarity search: {str(e)}")

def process_question(question: str, chunks: List[Document], qa_chain):
    try:
        # Perform similarity search to get the top 10 relevant chunks
        top_chunks = get_top_chunks(chunks, question, top_k=10)

        # Pass the relevant chunks as context to the LLM for answering the question
        response = qa_chain.run(input_documents=top_chunks, question=question)
        return question, response
    except Exception as e:
        traceback.print_exc()
        raise Exception(f"Error processing question '{question}': {str(e)}")

def get_answers_parallel(questions: List[str], chunks: List[Document], qa_chain):
    answers = {}

    # Use ThreadPoolExecutor to parallelize the process
    try:
        with concurrent.futures.ThreadPoolExecutor() as executor:
            # Submit tasks to the thread pool
            futures = {executor.submit(process_question, question, chunks, qa_chain): question for question in questions}

            # Process completed tasks as they finish
            for future in concurrent.futures.as_completed(futures):
                question, response = future.result()
                answers[question] = response.strip()

        return answers
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error during parallel processing of questions: {str(e)}")
