import os
from typing import List, Dict, Any, Optional

from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import Ollama
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

import gradio as gr

class RagQuerySystem:
    def __init__(
        self,
        embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        llm_model_name: str = "hermes3:3b",
        persist_directory: str = "./chroma_db",
        temperature: float = 0.1,
        top_k: int = 4,
    ):
        """
        Initialize the RAG Query System without any CLI logging.
        """
        self.embeddings = HuggingFaceEmbeddings(model_name=embedding_model_name)

        if os.path.exists(persist_directory) and os.listdir(persist_directory):
            self.vectorstore = Chroma(
                persist_directory=persist_directory,
                embedding_function=self.embeddings
            )
        else:
            raise FileNotFoundError(
                f"Vector store not found at {persist_directory}. Encode books first."
            )

        self.llm = Ollama(
            model=llm_model_name,
            temperature=temperature,
        )

        template = """You are a helpful AI assistant that answers questions based on the provided context.

Context information is below:
--------------------------
{context}
--------------------------

Given the context information and not prior knowledge, answer the following question.
If the answer cannot be determined from the context, say "I don't have enough information to answer this question."

Question: {question}

Answer: """
        self.chain = LLMChain(
            llm=self.llm,
            prompt=PromptTemplate(
                input_variables=["context", "question"],
                template=template
            )
        )
        self.top_k = top_k

    def query(self, question: str, filter_metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Return the answer string for a question, optionally filtered by metadata.
        """
        # retrieve
        docs = (
            self.vectorstore.similarity_search(question, k=self.top_k, filter=filter_metadata)
            if filter_metadata
            else self.vectorstore.similarity_search(question, k=self.top_k)
        )

        # prepare context
        context = "\n\n".join(doc.page_content for doc in docs)
        return self.chain.run(context=context, question=question)

    def list_available_books(self) -> List[str]:
        """
        Return unique titles from the vector store metadata.
        """
        all_docs = self.vectorstore.get()
        titles = {
            md.get("title")
            for md in all_docs.get("metadatas", [])
            if md.get("title")
        }
        return sorted(titles)


def build_ui():
    rag = RagQuerySystem(
        embedding_model_name="sentence-transformers/all-MiniLM-L6-v2",
        llm_model_name="hermes3:3b",
        persist_directory="./chroma_db",
        temperature=0.1,
        top_k=4,
    )

    # fetch titles for dropdown
    books = rag.list_available_books()
    books.insert(0, "")  # option for "no filter"

    def answer(question: str, book: str):
        filt = {"title": book} if book else None
        return rag.query(question, filter_metadata=filt)

    with gr.Blocks() as demo:
        gr.Markdown("## 📚 RAG Query Playground")
        with gr.Row():
            question_input = gr.Textbox(
                label="Enter your question", placeholder="What do you want to know?"
            )
            book_dropdown = gr.Dropdown(
                choices=books,
                label="Filter by book (optional)"
            )
        answer_output = gr.Textbox(label="Answer", interactive=False)

        question_input.submit(answer, [question_input, book_dropdown], answer_output)
        book_dropdown.change(answer, [question_input, book_dropdown], answer_output)

    return demo

if __name__ == "__main__":
    ui = build_ui()
    ui.launch()

