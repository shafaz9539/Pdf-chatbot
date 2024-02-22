from fastapi import FastAPI, Request, UploadFile, File, Form, HTTPException
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from PyPDF2 import PdfReader
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from dotenv import load_dotenv
from langchain.llms import OpenAI
import os
from io import BytesIO

# Load environment variables from .env file
load_dotenv()

# Create a FastAPI instance
app = FastAPI()

# Mount static files directory
app.mount("/static", StaticFiles(directory="static"), name="static")

# Initialize Jinja2Templates for HTML templates
templates = Jinja2Templates(directory="templates")

# Global variables
VectorStore = None
file_name = None

# Endpoint for uploading PDF files
@app.post("/upload/", response_class=HTMLResponse)
async def upload_pdf(request: Request, file: UploadFile = File(...)):
    try:
        global VectorStore, file_name
        file_name = file.filename
        # Extract contents from the file
        contents = await file.read()
        pdf = PdfReader(BytesIO(contents))
        text = ''
        for page in pdf.pages:   
            text += page.extract_text()
        # Split the raw text into chunks
        text_splitter = CharacterTextSplitter(
            separator="\n",
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
        )
        chunks = text_splitter.split_text(text)
        
        # Create embeddings using OpenAIEmbeddings
        embeddings = OpenAIEmbeddings()
        VectorStore = FAISS.from_texts(chunks, embedding=embeddings)
        
        return templates.TemplateResponse("success.html", {"request": request, "file_name": file_name, "extracted_text": text})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Endpoint for querying the uploaded PDF file
@app.post("/query/", response_class=HTMLResponse)
async def response(request: Request, query: str = Form(...)):
    try:
        if query: 
            # Perform similarity search based on the query
            docs = VectorStore.similarity_search(query=query)
            llm = OpenAI()
            chain = load_qa_chain(llm=llm, chain_type="stuff")
            response = chain.run(input_documents=docs, question=query)
        
        return templates.TemplateResponse("success.html", {"request": request, "file_name": file_name, "response": response})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Exception handler for handling HTTPExceptions
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    return templates.TemplateResponse("error.html", {"request": request, "error": exc.detail})

# Main endpoint returning the main HTML template
@app.get("/", response_class=HTMLResponse)
async def main(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, "file_name": None})
