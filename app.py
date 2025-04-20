import os
import re
from typing import List, Dict, Any, Optional, Tuple

from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import Ollama
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

import gradio as gr
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
import base64

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

        # Enhanced prompt template for graph generation
        template = """You are a helpful AI assistant that answers questions based on the provided context.

Context information is below:
--------------------------
{context}
--------------------------

Given the context information and not prior knowledge, answer the following question.
If the answer cannot be determined from the context, say "I don't have enough information to answer this question."

If the user is asking for a graph, chart, or visual representation of data:
1. Structure your response to include a table of data in a format that can be parsed:
   - Start the data table with "DATA_TABLE_START" and end with "DATA_TABLE_END"
   - Format the table as CSV with column headers in the first row
   - Make sure the data is numerical where appropriate
2. Specify which kind of chart would be best (bar, line, pie, scatter, etc.) by including "CHART_TYPE: [type]"
3. Provide a title for the chart with "CHART_TITLE: [title]"
4. Then continue with your normal response explaining the data

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

    def query(self, question: str, filter_metadata: Optional[Dict[str, Any]] = None) -> Tuple[str, Optional[Dict]]:
        """
        Return the answer string for a question and chart data if applicable.
        """
        # retrieve
        docs = (
            self.vectorstore.similarity_search(question, k=self.top_k, filter=filter_metadata)
            if filter_metadata
            else self.vectorstore.similarity_search(question, k=self.top_k)
        )

        # prepare context
        context = "\n\n".join(doc.page_content for doc in docs)
        response = self.chain.run(context=context, question=question)

        # Check if the response contains chart data
        chart_data = self._extract_chart_data(response)

        # Clean up the response by removing the chart metadata if present
        if chart_data:
            clean_response = self._clean_chart_metadata(response)
            return clean_response, chart_data

        return response, None

    def _extract_chart_data(self, text: str) -> Optional[Dict]:
        """Extract chart data and metadata from the response."""
        # Check for data table
        data_table_match = re.search(r'DATA_TABLE_START\s*(.*?)\s*DATA_TABLE_END', text, re.DOTALL)
        if not data_table_match:
            return None

        data_table_text = data_table_match.group(1).strip()

        # Extract chart type
        chart_type_match = re.search(r'CHART_TYPE:\s*(\w+)', text)
        chart_type = chart_type_match.group(1).lower() if chart_type_match else 'bar'

        # Extract chart title
        chart_title_match = re.search(r'CHART_TITLE:\s*(.*?)(?:\n|$)', text)
        chart_title = chart_title_match.group(1).strip() if chart_title_match else 'Chart'

        # Convert to DataFrame
        try:
            # Use StringIO to parse CSV
            df = pd.read_csv(pd.StringIO(data_table_text))

            return {
                'data': df,
                'chart_type': chart_type,
                'chart_title': chart_title
            }
        except Exception:
            return None

    def _clean_chart_metadata(self, text: str) -> str:
        """Remove chart metadata from the response."""
        # Remove data table
        text = re.sub(r'DATA_TABLE_START\s*.*?\s*DATA_TABLE_END', '', text, flags=re.DOTALL)
        # Remove chart type
        text = re.sub(r'CHART_TYPE:\s*\w+\s*\n?', '', text)
        # Remove chart title
        text = re.sub(r'CHART_TITLE:\s*.*?(?:\n|$)', '', text)
        # Clean up any double newlines
        text = re.sub(r'\n\s*\n\s*\n', '\n\n', text)
        return text.strip()

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


def generate_chart(chart_data: Dict) -> str:
    """Generate chart from data and return as base64 encoded image."""
    if chart_data is None:
        return None

    df = chart_data['data']
    chart_type = chart_data['chart_type']
    chart_title = chart_data['chart_title']

    plt.figure(figsize=(10, 6))

    # Set style
    sns.set_style("whitegrid")

    # Create appropriate chart based on the type
    if chart_type == 'bar':
        if len(df.columns) >= 2:
            sns.barplot(x=df.columns[0], y=df.columns[1], data=df)
    elif chart_type == 'line':
        if len(df.columns) >= 2:
            sns.lineplot(x=df.columns[0], y=df.columns[1], data=df)
    elif chart_type == 'pie':
        if len(df.columns) >= 2:
            plt.pie(df[df.columns[1]], labels=df[df.columns[0]], autopct='%1.1f%%')
    elif chart_type == 'scatter':
        if len(df.columns) >= 3:
            sns.scatterplot(x=df.columns[0], y=df.columns[1], hue=df.columns[2] if len(df.columns) > 2 else None, data=df)
        else:
            sns.scatterplot(x=df.columns[0], y=df.columns[1], data=df)
    elif chart_type == 'histogram':
        if len(df.columns) >= 1:
            sns.histplot(df[df.columns[0]])

    plt.title(chart_title)
    plt.tight_layout()

    # Save plot to BytesIO object
    buf = BytesIO()
    plt.savefig(buf, format='png', dpi=100)
    buf.seek(0)
    plt.close()

    # Convert to base64 for embedding in HTML
    img_str = base64.b64encode(buf.read()).decode('utf-8')

    return f"data:image/png;base64,{img_str}"


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
    books.insert(0, "All Books")  # Changed from empty string for better UX

    def answer(question: str, book: str, k_value: int, temperature: float):
        if not question.strip():
            return "Please enter a question", None, None

        # Update system parameters
        rag.top_k = k_value
        rag.llm.temperature = temperature

        # Apply filter if a specific book is selected
        filt = {"title": book} if book and book != "All Books" else None

        # Get response and possibly chart data
        response, chart_data = rag.query(question, filter_metadata=filt)

        # Generate chart if data is available
        chart_image = None
        if chart_data:
            chart_image = generate_chart(chart_data)

        # Get context sources for transparency
        sources = ""
        if filt:
            sources = f"Sources: Filtered to '{book}'"
        else:
            sources = "Sources: All available books"

        return response, chart_image, sources

    with gr.Blocks(css="""
        .container { max-width: 900px; margin: auto; }
        .header { text-align: center; margin-bottom: 20px; }
        .response-container { min-height: 200px; }
        .chart-display { display: flex; justify-content: center; margin-top: 10px; }
        .footer { margin-top: 20px; text-align: center; font-size: 0.8em; color: #666; }
    """) as demo:
        with gr.Row(elem_classes="header"):
            gr.Markdown("# 📚 Enhanced RAG Query System")
            gr.Markdown("Ask questions about your documents and get visualized insights")

        with gr.Row():
            with gr.Column(scale=3):
                question_input = gr.Textbox(
                    label="Enter your question",
                    placeholder="What do you want to know? You can ask for charts by including 'show me a graph of...' in your question.",
                    lines=3
                )
            with gr.Column(scale=1):
                book_dropdown = gr.Dropdown(
                    choices=books,
                    label="Filter by book",
                    value="All Books"
                )

        with gr.Row():
            with gr.Column(scale=1):
                k_slider = gr.Slider(
                    minimum=1,
                    maximum=10,
                    value=4,
                    step=1,
                    label="Number of context chunks (k)",
                    info="Higher values provide more context but may be slower"
                )
            with gr.Column(scale=1):
                temp_slider = gr.Slider(
                    minimum=0.0,
                    maximum=1.0,
                    value=0.1,
                    step=0.05,
                    label="Temperature",
                    info="Higher values make responses more creative but potentially less accurate"
                )

        submit_btn = gr.Button("Submit Question", variant="primary")

        with gr.Row():
            sources_text = gr.Markdown(label="Sources")

        with gr.Tabs():
            with gr.TabItem("Answer"):
                answer_output = gr.Markdown(elem_classes="response-container")
            with gr.TabItem("Visualization"):
                chart_output = gr.Image(label="Chart", elem_classes="chart-display")

        with gr.Row(elem_classes="footer"):
            gr.Markdown("Powered by LangChain and Ollama | Created with Gradio")

        submit_btn.click(
            answer,
            inputs=[question_input, book_dropdown, k_slider, temp_slider],
            outputs=[answer_output, chart_output, sources_text]
        )
        question_input.submit(
            answer,
            inputs=[question_input, book_dropdown, k_slider, temp_slider],
            outputs=[answer_output, chart_output, sources_text]
        )

    return demo

if __name__ == "__main__":
    ui = build_ui()
    ui.launch(share=True)  # Added share=True for easier sharing
