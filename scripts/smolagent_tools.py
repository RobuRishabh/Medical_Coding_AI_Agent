import torch
from pathlib import Path
import logging
from smolagents import Tool
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.docstore.document import Document
from duckduckgo_search import DDGS
import functools
import hashlib
import time
from typing import List, Optional, Dict, Any
from collections import OrderedDict
import threading

# ───────────────────────────────────────────────
# LOGGING SETUP - Remove file handler
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
# THREAD-SAFE CACHE WITH TTL
# ───────────────────────────────────────────────
class TTLCache:
    """Thread-safe cache with TTL (Time To Live)"""
    def __init__(self, max_size: int = 100, ttl_seconds: int = 300):  # 5 minutes TTL
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self.cache = OrderedDict()
        self.timestamps = {}
        self.lock = threading.RLock()
    
    def _is_expired(self, key: str) -> bool:
        """Check if cache entry is expired"""
        if key not in self.timestamps:
            return True
        return time.time() - self.timestamps[key] > self.ttl_seconds
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache"""
        with self.lock:
            if key in self.cache and not self._is_expired(key):
                # Move to end (most recently used)
                self.cache.move_to_end(key)
                return self.cache[key]
            else:
                # Remove expired entry
                self.remove(key)
                return None
    
    def set(self, key: str, value: Any) -> None:
        """Set value in cache"""
        with self.lock:
            # Remove if exists
            if key in self.cache:
                del self.cache[key]
                del self.timestamps[key]
            
            # Add new entry
            self.cache[key] = value
            self.timestamps[key] = time.time()
            
            # Remove oldest if over max size
            while len(self.cache) > self.max_size:
                oldest_key = next(iter(self.cache))
                self.remove(oldest_key)
    
    def remove(self, key: str) -> None:
        """Remove entry from cache"""
        if key in self.cache:
            del self.cache[key]
        if key in self.timestamps:
            del self.timestamps[key]
    
    def clear(self) -> None:
        """Clear all cache entries"""
        with self.lock:
            self.cache.clear()
            self.timestamps.clear()

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
        
        # Initialize caching
        self.cache = TTLCache(max_size=200, ttl_seconds=600)  # 10 minutes TTL
        self.query_stats = {"hits": 0, "misses": 0, "total": 0}
        
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
    
    def _normalize_query(self, query: str) -> str:
        """Normalize query for consistent caching"""
        # Convert to lowercase and strip whitespace
        normalized = query.lower().strip()
        
        # Remove common question words that don't affect search
        stop_words = ['what', 'is', 'the', 'how', 'do', 'i', 'code', 'should', 'use', 'which']
        words = normalized.split()
        filtered_words = [word for word in words if word not in stop_words]
        
        return ' '.join(filtered_words)
    
    def _get_cache_key(self, query: str, k: int) -> str:
        """Generate cache key for query"""
        normalized_query = self._normalize_query(query)
        cache_input = f"{normalized_query}:{k}"
        return hashlib.md5(cache_input.encode()).hexdigest()
    
    def _similarity_search_cached(self, query: str, k: int = 2) -> List:
        """Cached similarity search"""
        if not self.vector_db:
            return []
        
        # Check cache first
        cache_key = self._get_cache_key(query, k)
        cached_result = self.cache.get(cache_key)
        
        self.query_stats["total"] += 1
        
        if cached_result is not None:
            self.query_stats["hits"] += 1
            logger.debug(f"Cache hit for query: '{query[:50]}...'")
            return cached_result
        
        # Cache miss - perform actual search
        self.query_stats["misses"] += 1
        logger.debug(f"Cache miss for query: '{query[:50]}...'")
        
        try:
            # Perform search with reduced k for speed
            results = self.vector_db.similarity_search(query, k=k)
            
            # Cache the results
            self.cache.set(cache_key, results)
            
            return results
            
        except Exception as e:
            logger.error(f"Similarity search failed: {e}")
            return []
    
    def forward(self, query: str) -> str:
        """Execute the retrieval with caching and optimization"""
        logger.info(f"Performing optimized knowledge base search for: '{query[:50]}...'")
        
        if not isinstance(query, str):
            error_msg = "Search query must be a string"
            logger.error(error_msg)
            return error_msg
            
        if not self.vector_db:
            error_msg = "Vector database not available"
            logger.error(error_msg)
            return error_msg

        try:
            # Use cached search with reduced k value for speed
            results = self._similarity_search_cached(query, k=2)  # Reduced from 4 to 2
            
            if not results:
                logger.warning("No relevant documents found")
                return "No relevant medical coding information found in the knowledge base."
            
            # Format results more efficiently
            formatted_results = "\n🔍 Knowledge Base Results:\n"
            for i, doc in enumerate(results, 1):
                # Truncate long content to reduce processing time and improve readability
                content = doc.page_content
                if len(content) > 800:  # Increased from 500 for better context
                    content = content[:800] + "...\n[Content truncated for brevity]"
                
                formatted_results += f"\n--- Result {i} ---\n{content}\n"
            
            # Add cache statistics in debug mode
            hit_rate = (self.query_stats["hits"] / self.query_stats["total"]) * 100 if self.query_stats["total"] > 0 else 0
            logger.debug(f"Cache hit rate: {hit_rate:.1f}% ({self.query_stats['hits']}/{self.query_stats['total']})")
            
            logger.info("Optimized knowledge base search completed successfully")
            return formatted_results
            
        except Exception as e:
            error_msg = f"Error during knowledge base search: {str(e)}"
            logger.error(error_msg)
            return error_msg
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache performance statistics"""
        return {
            "hit_rate": (self.query_stats["hits"] / self.query_stats["total"]) * 100 if self.query_stats["total"] > 0 else 0,
            "total_queries": self.query_stats["total"],
            "cache_hits": self.query_stats["hits"],
            "cache_misses": self.query_stats["misses"],
            "cache_size": len(self.cache.cache)
        }

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
        self.cache = TTLCache(max_size=150, ttl_seconds=900)  # 15 minutes TTL for web results
        self.query_stats = {"hits": 0, "misses": 0, "total": 0}

    def _get_cache_key(self, query: str) -> str:
        """Generate cache key for web search query"""
        normalized_query = query.lower().strip()
        return hashlib.md5(normalized_query.encode()).hexdigest()

    def forward(self, query: str) -> str:
        """Execute web search with caching"""
        logger.info(f"Performing optimized web search for: '{query[:50]}...'")
        
        if not isinstance(query, str):
            error_msg = "Search query must be a string"
            logger.error(error_msg)
            return error_msg

        # Check cache first
        cache_key = self._get_cache_key(query)
        cached_result = self.cache.get(cache_key)
        
        self.query_stats["total"] += 1
        
        if cached_result is not None:
            self.query_stats["hits"] += 1
            logger.debug(f"Web search cache hit for query: '{query[:50]}...'")
            return cached_result

        # Cache miss - perform actual search
        self.query_stats["misses"] += 1
        logger.debug(f"Web search cache miss for query: '{query[:50]}...'")

        try:
            logger.debug("Initializing DuckDuckGo search...")
            with DDGS() as ddgs:
                logger.debug("Performing web search...")
                # Reduced max_results from 3 to 2 for faster processing
                results = list(ddgs.text(query, max_results=2))
                
                if not results:
                    result = "No relevant web results found for your medical coding query."
                    self.cache.set(cache_key, result)
                    return result
                
                # Format results more efficiently
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
                
                # Cache the results
                self.cache.set(cache_key, formatted_results)
                
                # Log cache statistics
                hit_rate = (self.query_stats["hits"] / self.query_stats["total"]) * 100 if self.query_stats["total"] > 0 else 0
                logger.debug(f"Web search cache hit rate: {hit_rate:.1f}% ({self.query_stats['hits']}/{self.query_stats['total']})")
                
                logger.info("Optimized web search completed successfully")
                return formatted_results
                
        except Exception as e:
            error_msg = f"Error during web search: {str(e)}"
            logger.error(error_msg)
            return error_msg
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get web search cache performance statistics"""
        return {
            "hit_rate": (self.query_stats["hits"] / self.query_stats["total"]) * 100 if self.query_stats["total"] > 0 else 0,
            "total_queries": self.query_stats["total"],
            "cache_hits": self.query_stats["hits"],
            "cache_misses": self.query_stats["misses"],
            "cache_size": len(self.cache.cache)
        }

# ───────────────────────────────────────────────
# INITIALIZE OPTIMIZED TOOLS
# ───────────────────────────────────────────────
logger.info("Initializing optimized tools...")
knowledge_base_retriever = KnowledgeBaseRetriever()
web_search_tool = WebSearchTool()

# Export tool names for the prompt template
TOOL_NAMES = [knowledge_base_retriever.name, web_search_tool.name]

# ───────────────────────────────────────────────
# UTILITY FUNCTIONS FOR PERFORMANCE MONITORING
# ───────────────────────────────────────────────
def get_tools_performance_stats() -> Dict[str, Any]:
    """Get performance statistics for all tools"""
    return {
        "knowledge_base": knowledge_base_retriever.get_cache_stats(),
        "web_search": web_search_tool.get_cache_stats()
    }

def clear_all_caches():
    """Clear all tool caches"""
    knowledge_base_retriever.cache.clear()
    web_search_tool.cache.clear()
    logger.info("All tool caches cleared")

logger.info("Optimized smolagent_tools initialization completed")