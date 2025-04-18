import os
import glob
from typing import List, Dict, Any
import pandas as pd
from langchain.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class BookEncoder:
    def __init__(self, 
                 embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
                 chunk_size: int = 1000,
                 chunk_overlap: int = 200,
                 persist_directory: str = "./chroma_db"):
        """
        Initialize the BookEncoder.
        
        Args:
            embedding_model_name: HuggingFace model to use for embeddings
            chunk_size: Size of text chunks to split documents into
            chunk_overlap: Overlap between chunks
            persist_directory: Directory to persist the vector database
        """
        self.embedding_model_name = embedding_model_name
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.persist_directory = persist_directory
        
        # Initialize embeddings
        logger.info(f"Loading embedding model: {embedding_model_name}")
        self.embeddings = HuggingFaceEmbeddings(model_name=embedding_model_name)
        
        # Initialize text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len
        )
        
        # Initialize or load vector store
        self._load_or_create_vectorstore()
        
    def _load_or_create_vectorstore(self):
        """Load existing vector store or create a new one"""
        if os.path.exists(self.persist_directory) and os.listdir(self.persist_directory):
            logger.info(f"Loading existing vector store from {self.persist_directory}")
            self.vectorstore = Chroma(
                persist_directory=self.persist_directory,
                embedding_function=self.embeddings
            )
        else:
            logger.info("Creating new vector store")
            self.vectorstore = Chroma(
                persist_directory=self.persist_directory,
                embedding_function=self.embeddings
            )
    
    def load_and_process_document(self, file_path: str, metadata: Dict[str, Any] = None) -> List:
        """
        Load and process a single document file.
        
        Args:
            file_path: Path to the document file
            metadata: Additional metadata to store with the document
            
        Returns:
            List of processed document chunks
        """
        logger.info(f"Processing document: {file_path}")
        
        # Determine file type and use appropriate loader
        if file_path.lower().endswith('.pdf'):
            loader = PyPDFLoader(file_path)
        elif file_path.lower().endswith(('.txt', '.md')):
            loader = TextLoader(file_path)
        else:
            raise ValueError(f"Unsupported file format: {file_path}")
        
        # Load and split documents
        documents = loader.load()
        
        # Add metadata if provided
        if metadata:
            for doc in documents:
                doc.metadata.update(metadata)
        
        # Split into chunks
        chunks = self.text_splitter.split_documents(documents)
        logger.info(f"Split document into {len(chunks)} chunks")
        
        return chunks
    
    def encode_documents(self, file_paths: List[str], metadatas: List[Dict[str, Any]] = None) -> None:
        """
        Encode multiple documents and store them in the vector database.
        
        Args:
            file_paths: List of paths to document files
            metadatas: List of metadata dictionaries corresponding to each file
        """
        all_chunks = []
        
        for i, file_path in enumerate(file_paths):
            metadata = metadatas[i] if metadatas and i < len(metadatas) else None
            try:
                chunks = self.load_and_process_document(file_path, metadata)
                all_chunks.extend(chunks)
            except Exception as e:
                logger.error(f"Error processing {file_path}: {e}")
        
        if all_chunks:
            logger.info(f"Adding {len(all_chunks)} chunks to vector store")
            self.vectorstore.add_documents(all_chunks)
            self.vectorstore.persist()
            logger.info("Documents successfully encoded and stored")
    
    def encode_directory(self, directory_path: str, file_pattern: str = "*.*") -> None:
        """
        Encode all matching documents in a directory.
        
        Args:
            directory_path: Path to directory containing documents
            file_pattern: Pattern to match files (e.g., "*.pdf" for PDFs only)
        """
        search_pattern = os.path.join(directory_path, file_pattern)
        file_paths = glob.glob(search_pattern)
        
        logger.info(f"Found {len(file_paths)} files matching pattern {file_pattern} in {directory_path}")
        
        # Extract book titles from filenames as default metadata
        metadatas = []
        for file_path in file_paths:
            filename = os.path.basename(file_path)
            book_title = os.path.splitext(filename)[0]
            metadatas.append({"source": file_path, "title": book_title})
        
        self.encode_documents(file_paths, metadatas)

# Example usage
if __name__ == "__main__":
    # Initialize encoder
    encoder = BookEncoder(
        embedding_model_name="sentence-transformers/all-MiniLM-L6-v2",
        chunk_size=1000,
        chunk_overlap=200,
        persist_directory="./book_knowledge_base"
    )
    
    # Example: Encode a single PDF book
    # encoder.encode_documents(["path/to/book.pdf"], [{"title": "Book Title", "author": "Author Name"}])
    
    # Example: Encode all PDFs in a directory
    # encoder.encode_directory("path/to/books", "*.pdf")
    
    print("Run this script with your specific book files or directories to encode them.")
    print("Example: encoder.encode_directory('/path/to/books', '*.pdf')")
