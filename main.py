from fastapi import FastAPI, UploadFile, File, HTTPException
from typing import List, Dict
from langchain.chains.question_answering import load_qa_chain
from langchain_community.llms import OpenAI
from utils import (
    chunk_document,
    extract_questions_from_json,
    extract_text_from_json,
    extract_text_from_pdf,
    get_answers_parallel,
)
from dotenv import load_dotenv

load_dotenv('.env')

app = FastAPI()

# Load the QA chain with a Langchain-compatible LLM
llm = OpenAI(temperature=0)
qa_chain = load_qa_chain(llm, chain_type="stuff")

# API endpoint
@app.post("/answer-questions")
async def answer_questions(
    questions_file: UploadFile = File(...), 
    input_file: UploadFile = File(...)
) -> Dict[str, str]:
    try:
        # Extract text from the uploaded input file
        if input_file.content_type == "application/pdf":
            file_text = extract_text_from_pdf(input_file)
        elif input_file.content_type == "application/json":
            file_text = extract_text_from_json(input_file)
        else:
            raise HTTPException(status_code=400, detail="Unsupported file format. Only PDF and JSON are accepted.")

        # Extract questions from the JSON file
        questions = extract_questions_from_json(questions_file)

        # Check if questions are empty
        if not questions:
            raise HTTPException(status_code=400, detail="No questions found in the provided JSON file.")

        # Chunk the document
        chunks = chunk_document(file_text)

        # Answer each question based on the top 10 chunks retrieved by similarity search
        answers = get_answers_parallel(questions, chunks, qa_chain)

        return answers

    except HTTPException as http_ex:
        raise http_ex
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")
