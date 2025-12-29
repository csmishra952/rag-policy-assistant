import os
from dotenv import load_dotenv
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings 
from langchain_community.vectorstores import Chroma
load_dotenv()
DATA_PATH = "./data"
DB_PATH = "./chroma_db"
def ingest_data():
    print("Starting data ingestion...")
    loader = DirectoryLoader(DATA_PATH, glob="*.txt", loader_cls=TextLoader)
    documents = loader.load()
    print(f"Loaded {len(documents)} documents.")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Split documents into {len(chunks)} chunks.")
    print("Creating vector database using Local Embeddings...")
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    db = Chroma.from_documents(
        documents=chunks, 
        embedding=embeddings, 
        persist_directory=DB_PATH
    ) 
    print(f"Success! Vector DB saved to {DB_PATH}")
if __name__ == "__main__":
    ingest_data()