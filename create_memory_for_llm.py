import os
import chardet
from langchain.schema import Document
from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())

DATA_DIRECTORY = "data/"

def get_file_encoding(file_path):
    with open(file_path, "rb") as file:
        detected = chardet.detect(file.read())
        return detected.get("encoding", "utf-8")

def load_text_documents(directory):
    documents = []
    for filename in os.listdir(directory):
        if filename.endswith(".txt"):
            file_path = os.path.join(directory, filename)
            encoding = get_file_encoding(file_path)
            with open(file_path, "r", encoding=encoding, errors="ignore") as file:
                content = file.read()
                documents.append(Document(page_content=content, metadata={"source": file_path}))
    return documents

documents = load_text_documents(directory=DATA_DIRECTORY)

def split_documents_into_chunks(docs):
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    return splitter.split_documents(docs)

document_chunks = split_documents_into_chunks(docs=documents)
print(f"Number of document chunks: {len(document_chunks)}")

def load_embedding_model():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

embedding_model = load_embedding_model()

FAISS_DB_PATH = "vectorstore/db_faiss"

faiss_db = FAISS.from_documents(document_chunks, embedding_model)
faiss_db.save_local(FAISS_DB_PATH)
