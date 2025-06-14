# build_vectorstore.py

import os
import bs4
from dotenv import load_dotenv
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# Load environment variables
load_dotenv()

# Step 1: Load and scrape content from URLs
urls = [
    "https://israel-entry.piba.gov.il/",
    "https://israel-entry.piba.gov.il/en/required-documents/",
    "https://israel-entry.piba.gov.il/en/exemptions-and-restrictions/",
    "https://israel-entry.piba.gov.il/en/entry-permits/",
    "https://israel-entry.piba.gov.il/en/special-cases/",
    "https://israel-entry.piba.gov.il/en/contact/",
]

loader = WebBaseLoader(
    web_paths=urls,
    bs_kwargs=dict(
        parse_only=bs4.SoupStrainer(
            class_=(
                "container", "entry-content", "page-content", "main-content", "article"
            )
        )
    )
)

docs = loader.load()

# Step 2: Split into chunks
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = splitter.split_documents(docs)

# Step 3: Embed and save to Chroma vectorstore
embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-large-en-v1.5")
persist_dir = "vectorstore"

vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory=persist_dir
)

print("✅ Vectorstore created and saved to:", persist_dir)
