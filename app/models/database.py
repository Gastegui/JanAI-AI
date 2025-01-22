"""
This script demonstrates a pipeline for processing PDF documents, splitting them into smaller chunks,
calculating embeddings for those chunks, and adding them to a Chroma database for persistent storage.
The main functionalities include:
- Loading PDF documents from a directory.
- Splitting documents into manageable text chunks.
- Computing embeddings for text chunks using a specified embedding model.
- Adding new chunks to the Chroma vector database, while avoiding duplicates.
- Clearing the database when needed.
"""

from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema.document import Document
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
import os
import shutil

CHROMA_PATH = "chroma"  # Path where the Chroma database is stored
DATA_PATH = "data"      # Directory where PDF files are located

def main():
    """Main entry point of the script."""
    documents = load_documents()           # Load all documents from the data directory
    chunks = split_documents(documents)    # Split the documents into smaller chunks
    print(chunks)                          # Print the chunks (for debugging purposes)
    add_to_chroma(chunks)                  # Add the chunks to the Chroma database

def load_documents():
    """
    Load all PDF documents from the specified data directory.

    Returns:
        list[Document]: A list of loaded documents.
    """
    document_loader = PyPDFDirectoryLoader(DATA_PATH)  # Initialize the PDF loader with the data path
    return document_loader.load()                      # Load and return the document

def split_documents(documents: list[Document]):
    """
    Split documents into smaller chunks for better processing and embedding.

    Args:
        documents (list[Document]): List of documents to split.

    Returns:
        list[Document]: List of document chunks.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,          # Maximum size of each chunk
        chunk_overlap=80,        # Overlap between chunks to maintain context
        length_function=len,     # Function to calculate chunk length
        is_separator_regex=False # Indicates whether the separator is a regex
    )
    return text_splitter.split_documents(documents)  # Split and return the document chunks

def get_embedding_function():
    """
    Initialize the embedding function for generating vector embeddings.

    Returns:
        OllamaEmbeddings: The embedding function object.
    """
    embeddings = OllamaEmbeddings(model="nomic-embed-text")  # Use the "nomic-embed-text" model
    return embeddings

def add_to_chroma(chunks: list[Document]):
    """
    Add document chunks to the Chroma database, avoiding duplicates.

    Args:
        chunks (list[Document]): List of document chunks to add to the database.
    """
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=get_embedding_function())  # Initialize the database

    chunks_with_ids = calculate_chunk_ids(chunks)  # Calculate unique IDs for each chunk

    existing_items = db.get(include=[])           # Retrieve existing items in the database
    existing_ids = set(existing_items["ids"])     # Extract existing IDs
    print(f"Number of existing documents in DB: {len(existing_ids)}")

    new_chunks = []  # Prepare a list for new chunks to add
    for chunk in chunks_with_ids:
        if chunk.metadata["id"] not in existing_ids:  # Check for duplicates
            new_chunks.append(chunk)

    if len(new_chunks):  # If there are new chunks, add them to the database
        print(f"👉 Adding new documents: {len(new_chunks)}")
        new_chunk_ids = [chunk.metadata["id"] for chunk in new_chunks]
        db.add_documents(new_chunks, ids=new_chunk_ids)
        db.persist()  # Persist the changes
    else:
        print("✅ No new documents to add")  # Inform that no new documents were added

def calculate_chunk_ids(chunks):
    """
    Generate unique IDs for each document chunk based on its source and page number.

    Args:
        chunks (list[Document]): List of document chunks.

    Returns:
        list[Document]: List of document chunks with assigned IDs.
    """
    last_page_id = None  # Track the last page ID
    current_chunk_index = 0  # Track the chunk index on the same page

    for chunk in chunks:
        source = chunk.metadata.get("source")  # Extract the document source
        page = chunk.metadata.get("page")      # Extract the page number
        current_page_id = f"{source}:{page}"   # Create a unique page identifier

        if current_page_id == last_page_id:    # If still on the same page, increment the index
            current_chunk_index += 1
        else:
            current_chunk_index = 0           # Reset the index for a new page

        chunk_id = f"{current_page_id}:{current_chunk_index}"  # Create a unique chunk ID
        last_page_id = current_page_id                        # Update the last page ID

        chunk.metadata["id"] = chunk_id  # Assign the ID to the chunk's metadata

    return chunks  # Return the updated chunks

def clear_database():
    """
    Clear the Chroma database by deleting the directory where it's stored.
    """
    if os.path.exists(CHROMA_PATH):  # Check if the database directory exists
        shutil.rmtree(CHROMA_PATH)  # Remove the directory and its contents

if __name__ == "__main__":
    main()  # Execute the main function