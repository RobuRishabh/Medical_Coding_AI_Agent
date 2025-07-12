import torch
from pathlib import Path
import logging
from smolagents import Tool
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.docstore.document import Document
from duckduckgo_search import DDGS

# ───────────────────────────────────────────────
# LOGGING SETUP
# ───────────────────────────────────────────────
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('smolagent_tools.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

logger.info("Starting smolagent_tools initialization")

# ───────────────────────────────────────────────
# INITIALIZE PATHS
# ───────────────────────────────────────────────
MARKDOWN_DIR = Path("Outputs/markdown")
VECTOR_DB_DIR = Path("Outputs/vector_db")

# ───────────────────────────────────────────────
# KNOWLEDGE BASE RETRIEVER TOOL
# ───────────────────────────────────────────────
class KnowledgeBaseRetriever(Tool):
    name = "knowledge_base_retriever"
    description = "Uses semantic search to retrieve relevant medical coding information from the embedded PDF knowledge base. Use this to find specific coding guidelines, procedures, and medical coding rules."
    inputs = {
        "query": {
            "type": "string",
            "description": "The medical coding question or search term. Use descriptive terms related to medical procedures, diagnoses, or coding guidelines rather than questions.",
        }
    }
    output_type = "string"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        logger.info("Initializing KnowledgeBaseRetriever...")
        
        try:
            # Initialize embedding model
            logger.info("Setting up embedding model...")
            device = "cuda" if torch.cuda.is_available() else "cpu"
            logger.info(f"Using device: {device}")
            
            self.embedding_model = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2",
                model_kwargs={"device": device}
            )
            
            # Initialize vector database
            logger.info("Setting up vector database...")
            if VECTOR_DB_DIR.exists():
                self.vector_db = Chroma(
                    embedding_function=self.embedding_model,
                    persist_directory=str(VECTOR_DB_DIR)
                )
                logger.info("Vector database loaded successfully")
            else:
                logger.warning("Vector database not found, will return empty results")
                self.vector_db = None
                
        except Exception as e:
            logger.error(f"Failed to initialize KnowledgeBaseRetriever: {e}")
            self.vector_db = None

    def forward(self, query: str) -> str:
        """Execute the retrieval based on the provided query."""
        logger.info(f"Performing knowledge base search for: '{query}'")
        
        if not isinstance(query, str):
            error_msg = "Search query must be a string"
            logger.error(error_msg)
            return error_msg
            
        if not self.vector_db:
            error_msg = "Vector database not available"
            logger.error(error_msg)
            return error_msg

        try:
            # Retrieve relevant documents
            logger.debug("Performing similarity search...")
            results = self.vector_db.similarity_search(query, k=4)
            logger.info(f"Found {len(results)} relevant documents")
            
            if not results:
                logger.warning("No relevant documents found")
                return "No relevant medical coding information found in the knowledge base."
            
            # Format the retrieved documents
            formatted_results = "\n🔍 Retrieved Medical Coding Information:\n"
            for i, doc in enumerate(results, 1):
                formatted_results += f"\n===== Document {i} =====\n"
                formatted_results += doc.page_content
                formatted_results += "\n" + "="*50 + "\n"
            
            logger.info("Knowledge base search completed successfully")
            return formatted_results
            
        except Exception as e:
            error_msg = f"Error during knowledge base search: {str(e)}"
            logger.error(error_msg)
            return error_msg

# ───────────────────────────────────────────────
# WEB SEARCH TOOL
# ───────────────────────────────────────────────
class WebSearchTool(Tool):
    name = "web_search"
    description = "Searches the web for current medical coding information when the knowledge base doesn't have sufficient information. Use this for recent updates, clarifications, or additional context."
    inputs = {
        "query": {
            "type": "string",
            "description": "The medical coding question or search term to find current information on the web. Include terms like 'medical coding', 'CPT', 'ICD-10', etc.",
        }
    }
    output_type = "string"

    def forward(self, query: str) -> str:
        """Execute web search based on the provided query."""
        logger.info(f"Performing web search for: '{query}'")
        
        if not isinstance(query, str):
            error_msg = "Search query must be a string"
            logger.error(error_msg)
            return error_msg

        try:
            logger.debug("Initializing DuckDuckGo search...")
            with DDGS() as ddgs:
                logger.debug("Performing web search...")
                results = list(ddgs.text(query, max_results=3))
                logger.info(f"Web search returned {len(results)} results")
                
                if not results:
                    logger.warning("No web search results found")
                    return "No relevant web results found for your medical coding query."
                
                # Format the web search results
                formatted_results = "\n🌐 Web Search Results:\n"
                for i, result in enumerate(results, 1):
                    formatted_results += f"\n===== Result {i} =====\n"
                    formatted_results += f"Title: {result.get('title', 'N/A')}\n"
                    formatted_results += f"URL: {result.get('href', 'N/A')}\n"
                    formatted_results += f"Content: {result.get('body', 'N/A')}\n"
                    formatted_results += "="*50 + "\n"
                
                logger.info("Web search completed successfully")
                return formatted_results
                
        except Exception as e:
            error_msg = f"Error during web search: {str(e)}"
            logger.error(error_msg)
            return error_msg

# ───────────────────────────────────────────────
# INITIALIZE TOOLS
# ───────────────────────────────────────────────
logger.info("Initializing tools...")
knowledge_base_retriever = KnowledgeBaseRetriever()
web_search_tool = WebSearchTool()

logger.info("smolagent_tools initialization completed")