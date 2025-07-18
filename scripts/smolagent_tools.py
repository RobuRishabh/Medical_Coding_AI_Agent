import torch
from pathlib import Path
import logging
from smolagents import Tool
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.docstore.document import Document
from duckduckgo_search import DDGS
from typing import List, Optional, Dict, Any

# ───────────────────────────────────────────────
# LOGGING SETUP
# ───────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
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
    description = "Uses semantic search to retrieve relevant medical coding information from the embedded " \
    "PDF knowledge base. Use this to find specific coding guidelines, procedures, and medical coding rules."
    inputs = {
        "query": {
            "type": "string",
            "description": "The medical coding question or search term. Use descriptive terms related "
            "to medical procedures, diagnoses, or coding guidelines rather than questions.",
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
        """Execute the retrieval without caching"""
        logger.info(f"Performing knowledge base search for: '{query[:50]}...'")
        
        if not isinstance(query, str):
            error_msg = "Search query must be a string"
            logger.error(error_msg)
            return error_msg
            
        if not self.vector_db:
            error_msg = "Vector database not available"
            logger.error(error_msg)
            return error_msg

        try:
            # Perform similarity search
            results = self.vector_db.similarity_search(query, k=2)
            
            if not results:
                logger.warning("No relevant documents found")
                return "No relevant medical coding information found in the knowledge base."
            
            # Format results
            formatted_results = "\n🔍 Knowledge Base Results:\n"
            for i, doc in enumerate(results, 1):
                # Truncate long content for readability
                content = doc.page_content
                if len(content) > 800:
                    content = content[:800] + "...\n[Content truncated for brevity]"
                
                formatted_results += f"\n--- Result {i} ---\n{content}\n"
            
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
    description = "Searches the web for current medical coding information when the knowledge base doesn't " \
    "have sufficient information. Use this for recent updates, clarifications, or additional context."
    inputs = {
        "query": {
            "type": "string",
            "description": "The medical coding question or search term to find current information on the web. "
            "Include terms like 'medical coding', 'CPT', 'ICD-10', etc.",
        }
    }
    output_type = "string"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def forward(self, query: str) -> str:
        """Execute web search without caching"""
        logger.info(f"Performing web search for: '{query[:50]}...'")
        
        if not isinstance(query, str):
            error_msg = "Search query must be a string"
            logger.error(error_msg)
            return error_msg

        try:
            logger.debug("Initializing DuckDuckGo search...")
            with DDGS() as ddgs:
                logger.debug("Performing web search...")
                # Get 2 results for faster processing
                results = list(ddgs.text(query, max_results=2))
                
                if not results:
                    return "No relevant web results found for your medical coding query."
                
                # Format results
                formatted_results = "\n🌐 Web Search Results:\n"
                for i, result in enumerate(results, 1):
                    title = result.get('title', 'N/A')
                    url = result.get('href', 'N/A')
                    body = result.get('body', 'N/A')
                    
                    # Truncate long content
                    if len(body) > 400:
                        body = body[:400] + "..."
                    
                    formatted_results += f"\n--- Result {i} ---\n"
                    formatted_results += f"Title: {title}\n"
                    formatted_results += f"URL: {url}\n"
                    formatted_results += f"Content: {body}\n"
                
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

# Export tool names for the prompt template
TOOL_NAMES = [knowledge_base_retriever.name, web_search_tool.name]

logger.info("Smolagent_tools initialization completed")