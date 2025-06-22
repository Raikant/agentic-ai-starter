import glob
import os

from langchain.agents import initialize_agent, Tool
from langchain.agents.agent_types import AgentType
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import PyPDFLoader
from langchain.memory import ConversationBufferMemory
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

# Create an embedding model
embedding = OpenAIEmbeddings()
llm = ChatOpenAI(temperature=0, model_name="gpt-4.1-nano")


class MultiPDFManager:
    def __init__(self, pdf_directory):
        self.pdf_directory = pdf_directory
        self.documents = {}
        self.combined_index = None
        self.load_all_pdfs()

    def load_all_pdfs(self):
        """Load all PDFs from directory and create a combined index"""
        all_documents = []
        pdf_files = glob.glob(os.path.join(self.pdf_directory, "*.pdf"))

        print(f"Loading {len(pdf_files)} PDF files...")

        for pdf_file in pdf_files:
            try:
                loader = PyPDFLoader(pdf_file)
                docs = loader.load()

                # Add metadata to track which document each chunk came from
                for doc in docs:
                    doc.metadata['source_file'] = os.path.basename(pdf_file)

                all_documents.extend(docs)
                print(f"Loaded: {os.path.basename(pdf_file)}")
            except Exception as e:
                print(f"Error loading {pdf_file}: {e}")

        # Create a single combined index
        if all_documents:
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200
            )
            texts = text_splitter.split_documents(all_documents)

            from langchain.vectorstores import FAISS
            self.combined_index = FAISS.from_documents(texts, embedding)
            print(f"Created combined index with {len(texts)} chunks")

    def query_documents(self, query, k=5):
        """Query all documents and return results with source information"""
        if not self.combined_index:
            return "No documents loaded"

        # Get relevant documents
        docs = self.combined_index.similarity_search_with_score(query, k=k)

        # Format response with source information
        context = []
        sources = set()

        for doc, score in docs:
            context.append(doc.page_content)
            sources.add(doc.metadata.get('source_file', 'Unknown'))

        # Use LLM to generate an answer based on context
        context_text = "\n\n".join(context)
        prompt = f"""Based on the following context from multiple PDF documents, answer the question: {query}

Context:
{context_text}

Answer:"""

        response = llm.predict(prompt)

        # Add source information
        sources_list = list(sources)
        if sources_list:
            response += f"\n\nSources: {', '.join(sources_list)}"

        return response


# Initialize the multi-PDF manager
pdf_manager = MultiPDFManager("../resources")  # Directory containing all your PDFs


def create_multi_pdf_tool():
    def multi_pdf_qa(query):
        return pdf_manager.query_documents(query)

    return Tool(
        name="Multi-PDF QA Tool",
        func=multi_pdf_qa,
        description="useful for answering questions about any of the loaded PDF documents. Can search across all documents and provide source information."
    )


# Create the single tool that handles all PDFs
multi_pdf_tool = create_multi_pdf_tool()

# Step 2: Define the agent
memory = ConversationBufferMemory(memory_key="chat_history")
print("model name: ", llm.model_name)
agent = initialize_agent(
    tools=[multi_pdf_tool],
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
    memory=memory
)

# Step 3: Run the agent
if __name__ == '__main__':
    print("Multi-PDF Agentic AI: Ask me anything based on the loaded PDFs")
    while True:
        user_input = input("\nYou: ")
        if user_input.lower() in ["exit", "quit"]:
            break
        response = agent.run(user_input)
        print(f"\nAgent: {response}")
