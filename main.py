import os
from typing import List, Dict, Any, Optional
import logging
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import Ollama
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class RagQuerySystem:
    def __init__(self, 
                 embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
                 llm_model_name: str = "hermes3:3b",
                 persist_directory: str = "./chroma_db",
                 temperature: float = 0.1,
                 top_k: int = 4):
        """
        Initialize the RAG Query System.
        
        Args:
            embedding_model_name: HuggingFace model to use for embeddings
            llm_model_name: Ollama model to use for generation
            persist_directory: Directory where the vector database is stored
            temperature: Temperature for the LLM
            top_k: Number of relevant chunks to retrieve
        """
        self.embedding_model_name = embedding_model_name
        self.llm_model_name = llm_model_name
        self.persist_directory = persist_directory
        self.temperature = temperature
        self.top_k = top_k
        
        # Initialize embeddings
        logger.info(f"Loading embedding model: {embedding_model_name}")
        self.embeddings = HuggingFaceEmbeddings(model_name=embedding_model_name)
        
        # Load vector store
        if os.path.exists(persist_directory) and os.listdir(persist_directory):
            logger.info(f"Loading vector store from {persist_directory}")
            self.vectorstore = Chroma(
                persist_directory=persist_directory,
                embedding_function=self.embeddings
            )
        else:
            raise FileNotFoundError(f"Vector store not found at {persist_directory}. Encode books first.")
        
        # Initialize LLM with streaming
        callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
        logger.info(f"Initializing Ollama LLM: {llm_model_name}")
        self.llm = Ollama(
            model=llm_model_name,
            temperature=temperature,
            callback_manager=callback_manager
        )
        
        # Create RAG prompt template
        self.prompt_template = PromptTemplate(
            input_variables=["context", "question"],
            template="""You are a helpful AI assistant that answers questions based on the provided context.
            
Context information is below:
--------------------------
{context}
--------------------------

Given the context information and not prior knowledge, answer the following question. 
If the answer cannot be determined from the context, say "I don't have enough information to answer this question."

Question: {question}

Answer: """
        )
        
        # Create LLM chain
        self.chain = LLMChain(llm=self.llm, prompt=self.prompt_template)
    
    def query(self, question: str, filter_metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Query the RAG system with a question.
        
        Args:
            question: Question to ask
            filter_metadata: Optional metadata filter for retrieval
            
        Returns:
            Answer from the RAG system
        """
        logger.info(f"Querying: {question}")
        
        # Retrieve relevant documents
        if filter_metadata:
            docs = self.vectorstore.similarity_search(question, k=self.top_k, filter=filter_metadata)
        else:
            docs = self.vectorstore.similarity_search(question, k=self.top_k)
        
        # Log retrieved sources
        sources = set()
        for doc in docs:
            if "source" in doc.metadata:
                sources.add(doc.metadata["source"])
            if "title" in doc.metadata:
                sources.add(doc.metadata["title"])
        
        logger.info(f"Retrieved {len(docs)} chunks from sources: {', '.join(sources)}")
        
        # Prepare context from retrieved documents
        context = "\n\n".join([doc.page_content for doc in docs])
        
        # Generate answer
        logger.info("Generating answer...")
        response = self.chain.run(context=context, question=question)
        
        return response
    
    def list_available_books(self) -> List[str]:
        """
        List all book titles available in the vector store.
        
        Returns:
            List of book titles
        """
        # This is a simplified approach - in a real implementation,
        # you would query the Chroma collection's metadata
        all_docs = self.vectorstore.get()
        titles = set()
        
        if all_docs and "metadatas" in all_docs:
            for metadata in all_docs["metadatas"]:
                if "title" in metadata:
                    titles.add(metadata["title"])
        
        return list(titles)

# Example usage
if __name__ == "__main__":
    try:
        # Initialize the RAG system
        rag_system = RagQuerySystem(
            embedding_model_name="sentence-transformers/all-MiniLM-L6-v2",
            llm_model_name="hermes3:3b",
            persist_directory="./chroma_db",  # Should match the directory used in encoder
            temperature=0.1,
            top_k=4
        )
        
        # Demo interface
        print("RAG Query System initialized.")
        print("Available books:", rag_system.list_available_books())
        
        while True:
            question = input("\nEnter your question (or 'exit' to quit): ")
            if question.lower() == 'exit':
                break
                
            # Optional: Filter by specific book
            book_filter = input("Filter by book title (leave empty for all books): ")
            filter_metadata = {"title": book_filter} if book_filter else None
            
            print("\nGenerating answer...\n")
            response = rag_system.query(question, filter_metadata)
            print("\n" + "="*50)
            
    except Exception as e:
        logger.error(f"Error: {e}")
        print(f"Error: {e}")
        print("Make sure you've encoded books using the encoder script first.")
