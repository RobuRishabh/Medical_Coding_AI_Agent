import streamlit as st
import os
import json
import logging
import re
from dotenv import load_dotenv
from smolagents import CodeAgent
from scripts.smolagent_tools import knowledge_base_retriever, web_search_tool, TOOL_NAMES
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from datetime import datetime
import litellm
from pathlib import Path

# ───────────────────────────────────────────────
# LOGGING SETUP - Single Output.log file (overwrite mode)
# ───────────────────────────────────────────────
# Ensure Outputs directory exists
outputs_dir = Path("Outputs")
outputs_dir.mkdir(exist_ok=True)

# Clear any existing handlers to avoid conflicts
for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)

# Configure logging to write to Output.log (overwrite each time)
log_file_path = outputs_dir / "Output.log"
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(str(log_file_path), mode='w', encoding='utf-8'),  # 'w' mode overwrites the file
        logging.StreamHandler()
    ],
    force=True  # Force reconfiguration
)

logger = logging.getLogger(__name__)

# Test logging immediately
logger.info("=== Medical Coding AI Agent Application Started ===")
logger.info(f"Log file location: {log_file_path.absolute()}")
logger.info(f"Current working directory: {os.getcwd()}")

# ───────────────────────────────────────────────
# STREAMLIT PAGE CONFIG WITH TIMEOUT SETTINGS
# ───────────────────────────────────────────────
st.set_page_config(
    page_title="CPC Medical Coding Assistant",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

logger.info("Streamlit page config set successfully")

# Configure Streamlit for long-running processes
import streamlit.web.cli as stcli
import sys

# Set unlimited timeout for long processes
os.environ["STREAMLIT_SERVER_MAX_UPLOAD_SIZE"] = "1000"
os.environ["STREAMLIT_SERVER_MAX_MESSAGE_SIZE"] = "1000"
os.environ["STREAMLIT_BROWSER_GATHER_USAGE_STATS"] = "false"

logger.info("Streamlit environment variables configured")

# ───────────────────────────────────────────────
# LOAD ENV VARIABLES
# ───────────────────────────────────────────────
load_dotenv()
logger.info("Environment variables loaded")

# Check for required environment variables
required_vars = ["OPENAI_API_KEY", "AGENT_MODEL"]
for var in required_vars:
    if os.getenv(var):
        logger.info(f"Environment variable {var} is set")
    else:
        logger.warning(f"Environment variable {var} is not set")

# ───────────────────────────────────────────────
# Load system prompt
# ───────────────────────────────────────────────
@st.cache_data
def load_system_prompt(path="prompts.json") -> str:
    logger.info(f"Loading system prompt from {path}")
    try:
        with open(path, "r", encoding="utf-8") as file:
            prompts = json.load(file)
        prompt = prompts.get("CPC_AGENT_PROMPT")
        logger.info("System prompt loaded successfully")
        return prompt
    except Exception as e:
        logger.error(f"Failed to load system prompt: {e}")
        st.error(f"Failed to load system prompt: {e}")
        return "You are a medical coding assistant."

# ───────────────────────────────────────────────
# OPTIMIZED AGENT CREATION WITH CACHING
# ───────────────────────────────────────────────
@st.cache_resource
def create_fast_agent(use_knowledge_base: bool = True, use_web_search: bool = True):
    """Create a faster, more efficient agent with caching"""
    logger.info("Creating optimized fast agent...")
    
    try:
        # Use LiteLLM wrapper for faster responses
        llm_wrapper = LiteLLMWrapper(
            model_name=os.getenv("AGENT_MODEL", "gpt-3.5-turbo"),
            api_key=os.getenv("OPENAI_API_KEY")
        )
        
        # Optimize tool selection - use cached tool instances
        selected_tools = []
        if use_web_search:
            selected_tools.append(web_search_tool)  # Use cached instance
        if use_knowledge_base:
            selected_tools.append(knowledge_base_retriever)  # Use cached instance
        
        logger.info(f"Creating agent with {len(selected_tools)} tools")
        
        # Create agent with performance optimizations
        agent = CodeAgent(
            tools=selected_tools,
            model=llm_wrapper,
            max_steps=1,  # Keep this low for speed
            additional_authorized_imports=["re", "json", "os"]
            # Removed verbose parameter as it's not supported
        )
        
        # Load and set optimized system prompt
        system_prompt = load_system_prompt()
        tool_names = [tool.name for tool in selected_tools]
        enhanced_system_prompt = system_prompt.format(tool_names=tool_names)
        
        # Set the system prompt
        agent.prompt_templates["system_prompt"] = enhanced_system_prompt
        
        logger.info("Fast agent created successfully")
        return agent
        
    except Exception as e:
        logger.error(f"Failed to create fast agent: {e}")
        st.error(f"Failed to create fast agent: {e}")
        return None

@st.cache_resource
def create_test_optimized_agent():
    """Create agent specifically optimized for test taking"""
    logger.info("Creating test-optimized agent...")
    
    try:
        # Use faster model settings for test taking
        llm_wrapper = LiteLLMWrapper(
            model_name=os.getenv("AGENT_MODEL", "gpt-3.5-turbo"),
            api_key=os.getenv("OPENAI_API_KEY")
        )
        
        # For tests, we want both tools for maximum accuracy
        selected_tools = [knowledge_base_retriever, web_search_tool]
        
        # Create agent with test-specific optimizations
        agent = CodeAgent(
            tools=selected_tools,
            model=llm_wrapper,
            max_steps=1,  # Single step for faster responses
            additional_authorized_imports=["re", "json", "os"]
            # Removed verbose parameter as it's not supported
        )
        
        # Load test-specific system prompt
        try:
            with open("prompts.json", "r", encoding="utf-8") as file:
                prompts = json.load(file)
            test_prompt = prompts.get("PRACTICE_TEST_PROMPT", "")
            if test_prompt:
                agent.prompt_templates["system_prompt"] = test_prompt
        except Exception as e:
            logger.warning(f"Could not load test prompt: {e}")
        
        logger.info("Test-optimized agent created successfully")
        return agent
        
    except Exception as e:
        logger.error(f"Failed to create test-optimized agent: {e}")
        return None

# ───────────────────────────────────────────────
# AGENT POOL FOR PARALLEL PROCESSING
# ───────────────────────────────────────────────
@st.cache_resource
def create_agent_pool(pool_size: int = 3):
    """Create a pool of agents for parallel processing"""
    logger.info(f"Creating agent pool with {pool_size} agents...")
    
    agent_pool = []
    for i in range(pool_size):
        try:
            agent = create_test_optimized_agent()
            if agent:
                agent_pool.append(agent)
                logger.info(f"Agent {i+1}/{pool_size} created successfully")
        except Exception as e:
            logger.error(f"Failed to create agent {i+1}: {e}")
    
    logger.info(f"Agent pool created with {len(agent_pool)} agents")
    return agent_pool

# ───────────────────────────────────────────────
# OPTIMIZED LITELLM WRAPPER
# ───────────────────────────────────────────────
class LiteLLMWrapper:
    def __init__(self, model_name, api_key=None):
        self.model_name = model_name
        if api_key:
            litellm.api_key = api_key
        
        # Aggressive optimizations for speed
        litellm.request_timeout = 30  # Reduced from 60 seconds
        litellm.caching = True
        litellm.max_retries = 1  # Reduced retries for speed
        
        # Set model-specific optimizations
        if "gpt-3.5" in model_name:
            self.default_max_tokens = 800
        elif "gpt-4" in model_name:
            self.default_max_tokens = 1000
        elif "nano" in model_name.lower():  # For your gpt-4.1-nano model
            self.default_max_tokens = 600  # Smaller for nano model
        else:
            self.default_max_tokens = 600
        
        logger.info(f"LiteLLM optimized for speed with model: {model_name}")

    def convert_messages(self, messages):
        """Convert messages to LiteLLM format (optimized version)"""
        if not isinstance(messages, list):
            messages = [messages]
            
        converted = []
        for msg in messages:
            role = getattr(msg, "role", None)
            content = getattr(msg, "content", None)

            if role == "user":
                converted.append({"role": "user", "content": content})
            elif role == "assistant":
                converted.append({"role": "assistant", "content": content})
            elif role == "system":
                converted.append({"role": "system", "content": content})
            elif hasattr(role, 'value') and role.value == 'tool-response':
                # Handle tool responses more efficiently
                if isinstance(content, list) and len(content) > 0:
                    text_content = content[0].get('text', str(content))
                else:
                    text_content = str(content)
                converted.append({"role": "user", "content": text_content})
            else:
                # Fallback conversion
                if isinstance(content, list) and len(content) > 0:
                    text_content = content[0].get('text', str(content))
                else:
                    text_content = str(content)
                converted.append({"role": "user", "content": text_content})
        
        return converted

    def generate(self, messages, **kwargs):
        try:
            # Convert messages to LiteLLM format
            litellm_messages = self.convert_messages(messages)
            
            # Optimized parameters for speed (especially for nano model)
            litellm_kwargs = {
                'model': self.model_name,
                'messages': litellm_messages,
                'temperature': kwargs.get('temperature', 0.1),  # Lower for consistency
                'max_tokens': kwargs.get('max_tokens', self.default_max_tokens),
                'timeout': 30,  # Reduced timeout
                'stream': False,  # Disable streaming for speed
            }
            
            # Add frequency penalty to reduce repetition (if supported by model)
            if "gpt" in self.model_name:
                litellm_kwargs['frequency_penalty'] = 0.1
            
            # Make the API call
            response = litellm.completion(**litellm_kwargs)
            
            # Extract content from response
            content = response.choices[0].message.content
            
            # Minimal post-processing for speed
            content = re.sub(r'```(?:python|tool_code)?\s*\n(.*?)\n```', r'<code>\1</code>', content, flags=re.DOTALL)
            
            # Create optimized result object
            class OptimizedLiteLLMResult:
                def __init__(self, content, response_obj):
                    self.content = content
                    self.tool_calls = None
                    
                    # Minimal token usage tracking
                    usage = getattr(response_obj, 'usage', None)
                    if usage:
                        class TokenUsage:
                            def __init__(self, usage_data):
                                self.input_tokens = getattr(usage_data, 'prompt_tokens', 0)
                                self.output_tokens = getattr(usage_data, 'completion_tokens', 0)
                                self.total_tokens = getattr(usage_data, 'total_tokens', 0)
                        self.token_usage = TokenUsage(usage)
                    else:
                        class TokenUsage:
                            def __init__(self):
                                self.input_tokens = 0
                                self.output_tokens = 0
                                self.total_tokens = 0
                        self.token_usage = TokenUsage()
            
            return OptimizedLiteLLMResult(content=content, response_obj=response)
            
        except Exception as e:
            logger.error(f"Optimized LiteLLM.generate failed: {e}")
            raise

    def invoke(self, messages, **kwargs):
        """Compatibility method for LangChain-style invoke"""
        result = self.generate(messages, **kwargs)
        
        class SimpleResponse:
            def __init__(self, content):
                self.content = content
        
        return SimpleResponse(result.content)

# ───────────────────────────────────────────────
# BACKWARD COMPATIBILITY
# ───────────────────────────────────────────────
def create_dynamic_agent(use_knowledge_base=True, use_web_search=True):
    """Backward compatibility wrapper - uses fast agent"""
    logger.info(f"Creating dynamic agent - KB: {use_knowledge_base}, Web: {use_web_search}")
    return create_fast_agent(use_knowledge_base, use_web_search)

# ───────────────────────────────────────────────
# Initialize Model with LiteLLM
# ───────────────────────────────────────────────
@st.cache_resource
def initialize_model():
    logger.info("Initializing LiteLLM language model...")
    try:
        model_name = os.getenv("AGENT_MODEL", "gpt-3.5-turbo")
        api_key = os.getenv("OPENAI_API_KEY")
        
        # Create LiteLLM wrapper
        llm = LiteLLMWrapper(model_name, api_key)
        
        logger.info(f"LiteLLM model initialized successfully: {model_name}")
        return llm
    except Exception as e:
        logger.error(f"Failed to initialize LiteLLM model: {e}")
        st.error(f"Failed to initialize model: {e}")
        return None

# ───────────────────────────────────────────────
# Initialize Agent with dynamic tools
# ───────────────────────────────────────────────
@st.cache_resource
def initialize_base_agent():
    logger.info("Initializing base agent...")
    llm = initialize_model()
    if not llm:
        logger.error("Failed to initialize model for base agent")
        return None, None
    
    try:
        logger.info("Creating base agent components...")
        system_prompt = load_system_prompt()
        
        # Debug: Check available tools
        logger.info(f"knowledge_base_retriever: {knowledge_base_retriever}")
        logger.info(f"web_search_tool: {web_search_tool}")
        
        # Log tool names if they have a name attribute
        if hasattr(knowledge_base_retriever, 'name'):
            logger.info(f"Knowledge base tool name: {knowledge_base_retriever.name}")
        if hasattr(web_search_tool, 'name'):
            logger.info(f"Web search tool name: {web_search_tool.name}")
        
        logger.info("Base agent components created successfully")
        return llm, system_prompt
    except Exception as e:
        logger.error(f"Failed to create base agent: {e}")
        st.error(f"Failed to create base agent: {e}")
        return None, None

def extract_citations_from_response(response_text, agent):
    """Extract and format citations from agent response"""
    citations = {
        'web_sources': [],
        'knowledge_base_sources': []
    }
    
    # Try to extract citations from the response text
    if "**Sources:**" in response_text:
        # Parse existing citations
        sources_section = response_text.split("**Sources:**")[1] if "**Sources:**" in response_text else ""
        
        # Extract web sources
        web_pattern = r'- Web Search: \[([^\]]+)\]\(([^)]+)\)(.*)'
        web_matches = re.findall(web_pattern, sources_section)
        for match in web_matches:
            citations['web_sources'].append({
                'name': match[0],
                'url': match[1],
                'description': match[2].strip(' -')
            })
        
        # Extract knowledge base sources
        kb_pattern = r'- Knowledge Base: ([^,]+), Section: "([^"]+)"(.*)'
        kb_matches = re.findall(kb_pattern, sources_section)
        for match in kb_matches:
            citations['knowledge_base_sources'].append({
                'document': match[0],
                'section': match[1],
                'description': match[2].strip(' -')
            })
    
    return citations

def display_enhanced_sources(response_text, use_knowledge_base, use_web_search):
    """Display which tools were used"""
    st.markdown("---")
    st.subheader("🔧 Tools Used")
    
    # Tool usage summary
    col_ref1, col_ref2 = st.columns(2)
    
    with col_ref1:
        if use_knowledge_base:
            st.success("✅ Knowledge Base Used")
        else:
            st.info("➖ Knowledge Base Not Used")
    
    with col_ref2:
        if use_web_search:
            st.success("✅ Web Search Used")
        else:
            st.info("➖ Web Search Not Used")

def chat_interface():
    """Chat interface page"""
    logger.info("Rendering chat interface")
    st.title("🩺 CPC Medical Coding Assistant - Chat Interface")
    st.markdown("---")
    
    # Main interface
    col1, col2 = st.columns([2, 1])

    with col1:
        st.header("💬 Ask Your Question")
        
        # Initialize session state for question if not exists
        if 'current_question' not in st.session_state:
            st.session_state.current_question = ""
        
        # Input area
        question = st.text_area(
            "Enter your medical coding question:",
            placeholder="e.g., What is the CPT code for a routine colonoscopy?",
            height=100,
            value=st.session_state.current_question,
            key="chat_question_input"
        )
        
        # Update session state when text area changes
        if question != st.session_state.current_question:
            st.session_state.current_question = question
        
        # Example questions
        st.subheader("💡 Example Questions")
        examples = [
            "How do I code a chest X-ray with interpretation?",
            "What ICD-10 code should I use for Type 2 diabetes?",
            "What's the difference between CPT 99213 and 99214?",
        ]
        
        for i, example in enumerate(examples):
            if st.button(f"📝 {example}", key=f"chat_example_{i}"):
                st.session_state.current_question = example
                st.rerun()

    with col2:
        st.header("⚙️ Options")
        # Web search is always enabled (can't be turned off)
        st.markdown("🌐 **Web Search**: Always Enabled")
        st.info("Web search is always active to provide the most current information.")
        use_web_search = True  # Always True
        
        # Search options
        use_knowledge_base = st.checkbox("🔍 Search Knowledge Base", value=True, key="chat_use_kb")
        st.info("Knowledge base search is optional. Uncheck to skip searching the embedded documents.")
        # Display options
        show_sources = st.checkbox("📖 Show Sources", value=True, key="chat_show_sources")

    # Submit button
    if st.button("🚀 Get Answer", type="primary", use_container_width=True, key="chat_get_answer"):
        if not st.session_state.current_question.strip():
            st.warning("Please enter a question!")
            logger.warning("User tried to submit empty question")
            return
        
        logger.info(f"User submitted question: {st.session_state.current_question[:100]}...")
        
        # Check if base components are ready
        llm, system_prompt = initialize_base_agent()
        if not llm or not system_prompt:
            st.error("❌ Agent Failed to Initialize")
            logger.error("Failed to initialize base agent for chat")
            return
        
        # Show processing
        with st.spinner("🔄 Processing your question..."):
            try:
                logger.info(f"Processing query: {st.session_state.current_question}")
                logger.info(f"Using tools - Knowledge Base: {use_knowledge_base}, Web Search: {use_web_search}")
                
                # Create agent with selected tools
                agent = create_dynamic_agent(use_knowledge_base, use_web_search)
                if not agent:
                    st.error("Failed to create agent with selected tools")
                    logger.error("Failed to create dynamic agent")
                    return
                
                # Get response from agent
                response = agent.run(st.session_state.current_question)
                
                # Display results
                st.markdown("---")
                st.header("🤖 Answer")
                
                # Format and display response
                if isinstance(response, str):
                    st.markdown(response)
                    response_text = response
                else:
                    st.write(response)
                    response_text = str(response)
                
                # Show enhanced sources and references if requested
                if show_sources:
                    display_enhanced_sources(response_text, use_knowledge_base, use_web_search)
                
                logger.info("Query processed successfully")
                
            except Exception as e:
                logger.error(f"Error processing query: {e}")
                st.error(f"⚠️ Error: {str(e)}")
                st.info("Please try rephrasing your question or check the system logs.")

def practice_test_interface():
    """Interface for automated practice test"""
    logger.info("Rendering practice test interface")
    st.title("🩺 CPC Medical Coding Assistant - Practice Test")
    st.markdown("---")
    
    st.header("🎯 Automated Practice Test Runner")
    
    # Initialize session state for test workflow
    if 'test_workflow_state' not in st.session_state:
        st.session_state.test_workflow_state = {
            'step': 1,
            'questions_extracted': False,
            'answers_extracted': False,
            'extraction_results': None,
            'test_completed': False
        }
        logger.info("Initialized test workflow state")
    
    # Step 1: File Upload and Extraction
    st.subheader("📁 Step 1: Upload Files and Extract Questions/Answers")
    
    # File upload
    col1, col2 = st.columns(2)
    
    with col1:
        test_file = st.file_uploader(
            "Upload Practice Test (PDF)", 
            type=['pdf'],
            help="Upload the practice test without answers",
            key="test_file_upload"
        )
    
    with col2:
        answers_file = st.file_uploader(
            "Upload Answer Key (PDF)", 
            type=['pdf'],
            help="Upload the practice test with answers",
            key="answers_file_upload"
        )
    
    # Add checkbox to skip extraction for development
    skip_extraction = st.checkbox("Skip PDF extraction (use cached data)", value=False, key="skip_extraction_cb")
    
    # Step 1 Button: Extract Questions and Answers
    extract_button_disabled = not skip_extraction and (not test_file or not answers_file)
    
    if st.button("🔍 Extract Questions and Answers", 
                 type="primary", 
                 disabled=extract_button_disabled,
                 key="extract_questions_answers"):
        
        if skip_extraction:
            logger.info("User chose to skip extraction, using cached data")
            st.info("Using cached test data...")
            st.session_state.test_workflow_state['questions_extracted'] = True
            st.session_state.test_workflow_state['answers_extracted'] = True
            st.session_state.test_workflow_state['step'] = 2
            st.success("✅ Using cached data - Ready for automated test!")
        else:
            logger.info("Starting PDF extraction process")
            # Save uploaded files
            test_path = f"temp_test_{datetime.now().timestamp()}.pdf"
            answers_path = f"temp_answers_{datetime.now().timestamp()}.pdf"
            
            with open(test_path, "wb") as f:
                f.write(test_file.read())
            with open(answers_path, "wb") as f:
                f.write(answers_file.read())
            
            logger.info(f"Saved uploaded files: {test_path}, {answers_path}")
            
            # Extract questions and answers
            with st.spinner("🔄 Extracting questions and answers from PDFs..."):
                try:
                    # Import and initialize test processor
                    from scripts.test_processor import TestProcessor
                    
                    processor = TestProcessor()

                    # Extract questions and answers using the processor methods
                    questions_data = processor.extract_questions_from_pdf(test_path)
                    answers_data = processor.extract_answers_from_pdf(answers_path)

                    # Store extraction results
                    st.session_state.test_workflow_state['extraction_results'] = {
                        'questions': questions_data,
                        'answers': answers_data,
                        'test_path': test_path,
                        'answers_path': answers_path
                    }
                    
                    st.session_state.test_workflow_state['questions_extracted'] = True
                    st.session_state.test_workflow_state['answers_extracted'] = True
                    st.session_state.test_workflow_state['step'] = 2
                    
                    # Display extraction summary
                    st.success("✅ Extraction completed successfully!")
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Questions Extracted", len(questions_data) if questions_data else 0)
                    with col2:
                        st.metric("Answers Extracted", len(answers_data) if answers_data else 0)
                    
                    # Show a preview of extracted data
                    if questions_data and len(questions_data) > 0:
                        with st.expander("📋 Preview Extracted Questions (First 3)"):
                            for i, question in enumerate(questions_data[:3]):
                                st.write(f"**Q{i+1}:** {question['question'][:200]}...")
                                if question.get('options'):
                                    for option in question['options']:
                                        st.write(f"  - {option}")
                    
                    if answers_data and len(answers_data) > 0:
                        with st.expander("📝 Preview Extracted Answers (First 3)"):
                            for i, answer in enumerate(answers_data[:3]):
                                st.write(f"**A{i+1}:** {answer}")
                    
                    # Show validation summary
                    if questions_data and answers_data:
                        st.subheader("🔍 Extraction Validation")
                        
                        # Check if counts match
                        if len(questions_data) == len(answers_data):
                            st.success(f"✅ Perfect match: {len(questions_data)} questions and {len(answers_data)} answers")
                        else:
                            st.error(f"❌ Mismatch: {len(questions_data)} questions vs {len(answers_data)} answers")
                            st.warning("This mismatch will cause accuracy issues during testing!")
                        
                        # Show answer distribution
                        if answers_data:
                            from collections import Counter
                            answer_counts = Counter(answers_data)
                            
                            col1, col2, col3, col4 = st.columns(4)
                            with col1:
                                st.metric("A answers", answer_counts.get('A', 0))
                            with col2:
                                st.metric("B answers", answer_counts.get('B', 0))
                            with col3:
                                st.metric("C answers", answer_counts.get('C', 0))
                            with col4:
                                st.metric("D answers", answer_counts.get('D', 0))
                            
                            # Check for distribution issues
                            total_answers = len(answers_data)
                            max_count = max(answer_counts.values()) if answer_counts else 0
                            max_percentage = (max_count / total_answers * 100) if total_answers > 0 else 0
                            
                            if max_percentage > 60:
                                st.warning(f"⚠️ One answer dominates ({max_percentage:.1f}%) - may indicate extraction error")
                            else:
                                st.success("✅ Answer distribution appears reasonable")
                    
                    # Save extracted data for debugging
                    processor.save_extracted_data(questions_data, answers_data)
                    logger.info("Extraction process completed successfully")
                    
                except ImportError as e:
                    logger.error(f"Missing module during extraction: {e}")
                    st.error(f"❌ Missing module: {str(e)}")
                    st.info("Please ensure test_processor.py is available in the scripts directory.")
                except Exception as e:
                    logger.error(f"Error during extraction: {e}")
                    st.error(f"❌ Error during extraction: {str(e)}")
                    st.info("Please check the PDF files and try again.")
                    # Clean up temporary files on error
                    try:
                        if 'test_path' in locals() and os.path.exists(test_path):
                            os.remove(test_path)
                        if 'answers_path' in locals() and os.path.exists(answers_path):
                            os.remove(answers_path)
                    except:
                        pass
    
    # Show extraction status
    if st.session_state.test_workflow_state['questions_extracted']:
        st.success("✅ Questions extracted successfully")
    if st.session_state.test_workflow_state['answers_extracted']:
        st.success("✅ Answers extracted successfully")
    
    # Step 2: Agent Configuration and Test Execution
    if st.session_state.test_workflow_state['step'] >= 2:
        st.markdown("---")
        st.subheader("⚙️ Step 2: Configure Agent and Run Test")
        
        # Agent configuration
        col1, col2, col3 = st.columns(3)
        
        with col1:
            use_knowledge_base = st.checkbox("Use Knowledge Base", value=True, key="test_use_kb")
        with col2:
            use_web_search = st.checkbox("Use Web Search", value=True, key="test_use_web")
        with col3:
            temperature = st.slider("Temperature", 0.0, 1.0, 0.2, key="test_temperature")
        
        # Display agent configuration summary
        st.info(f"**Agent Configuration:**\n"
                f"- Model: {os.getenv('AGENT_MODEL', 'gpt-3.5-turbo')}\n"
                f"- Knowledge Base: {'✅' if use_knowledge_base else '❌'}\n"
                f"- Web Search: {'✅' if use_web_search else '❌'}\n"
                f"- Temperature: {temperature}")
        
        # Step 2 Button: Run Automated Test
        test_button_disabled = not (st.session_state.test_workflow_state['questions_extracted'] and 
                                   st.session_state.test_workflow_state['answers_extracted'])
        
        if st.button("🚀 Run Automated Test", 
                     type="primary", 
                     use_container_width=True,
                     disabled=test_button_disabled,
                     key="run_automated_test"):
            
            logger.info("Starting automated test execution")
            
            # Check if base components are ready
            llm, system_prompt = initialize_base_agent()
            if not llm or not system_prompt:
                st.error("❌ Agent Failed to Initialize")
                logger.error("Agent failed to initialize for test execution")
                return
            
            # Create a single progress bar and status text outside the test execution
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Run automated test
            with st.spinner("🔄 Running automated test..."):
                try:
                    from scripts.test_runner import AutomatedTestRunner
                    
                    agent_config = {
                        'model': os.getenv("AGENT_MODEL", "gpt-3.5-turbo"),
                        'tools': ['knowledge_base_retriever', 'web_search'] if use_knowledge_base and use_web_search else ['web_search'],
                        'temperature': temperature,
                        'timeout': 3600
                    }
                    
                    logger.info(f"Created agent config: {agent_config}")
                    
                    # Create a callback for progress updates using the single progress bar
                    def progress_callback(current, total, message):
                        progress = current / total if total > 0 else 0
                        progress_bar.progress(progress)
                        status_text.text(f"Progress: {current}/{total} - {message}")
                        logger.info(f"Test progress: {current}/{total} - {message}")
                    
                    runner = AutomatedTestRunner(agent_config)
                    
                    # Use cached data if available, otherwise run full test
                    if skip_extraction:
                        logger.info("Running test with cached data")
                        results = runner.run_test_with_cached_data(progress_callback=progress_callback)
                    else:
                        logger.info("Running test with extracted data")
                        # Use the extracted data from step 1
                        extraction_results = st.session_state.test_workflow_state['extraction_results']
                        results = runner.run_test_with_extracted_data(
                            extraction_results['questions'],
                            extraction_results['answers'],
                            progress_callback=progress_callback
                        )
                    
                    # Clean up temporary files after test completion
                    if not skip_extraction and st.session_state.test_workflow_state['extraction_results']:
                        try:
                            test_path = st.session_state.test_workflow_state['extraction_results']['test_path']
                            answers_path = st.session_state.test_workflow_state['extraction_results']['answers_path']
                            if os.path.exists(test_path):
                                os.remove(test_path)
                                logger.info(f"Cleaned up {test_path}")
                            if os.path.exists(answers_path):
                                os.remove(answers_path)
                                logger.info(f"Cleaned up {answers_path}")
                        except Exception as e:
                            logger.warning(f"Failed to clean up temporary files: {e}")
                    
                    # Check if results is None
                    if results is None:
                        st.error("❌ Test runner returned no results")
                        logger.error("Test runner returned None results")
                        return
                    
                    # Clear the progress bar and status text after completion
                    progress_bar.empty()
                    status_text.empty()
                    
                    # Mark test as completed
                    st.session_state.test_workflow_state['test_completed'] = True
                    
                    # Display results
                    st.success(f"🎉 Test completed! Score: {results['score_percentage']:.1f}%")
                    logger.info(f"Test completed with score: {results['score_percentage']:.1f}%")
                    
                    # Display detailed results
                    st.subheader("📊 Test Results Summary")
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Total Questions", results['questions_answered'])
                    with col2:
                        st.metric("Correct Answers", results['correct_answers'])
                    with col3:
                        st.metric("Score", f"{results['score_percentage']:.1f}%")
                    
                    # Generate and display report
                    from scripts.results_generator import ResultsGenerator
                    generator = ResultsGenerator(results)
                    report = generator.generate_comprehensive_report()
                    
                    with st.expander("📋 View Detailed Report"):
                        st.markdown(report)
                    
                    # Download links
                    st.download_button(
                        "📥 Download Full Report",
                        data=report.encode('utf-8'),
                        file_name=f"practice_test_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
                        mime="text/markdown",
                        key="download_report"
                    )
                    
                    # Show JSON results as well
                    json_data = json.dumps(results, indent=2, default=str, ensure_ascii=False)
                    st.download_button(
                        "📥 Download Raw Results (JSON)",
                        data=json_data.encode('utf-8'),
                        file_name=f"practice_test_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                        mime="application/json",
                        key="download_json"
                    )
                    
                except FileNotFoundError as e:
                    logger.error(f"Cached data not found: {e}")
                    st.error(f"❌ Cached data not found: {str(e)}")
                    st.info("Please run the test with PDFs first to generate cached data.")
                except Exception as e:
                    logger.error(f"Error running test: {e}")
                    st.error(f"❌ Error running test: {str(e)}")
                    st.info("Please check the logs for more details.")
    
    # Step 3: Results and New Test Option
    if st.session_state.test_workflow_state.get('test_completed', False):
        st.markdown("---")
        st.subheader("🔄 Step 3: Start New Test")
        
        if st.button("🆕 Start New Test", key="start_new_test"):
            logger.info("User started new test, resetting workflow state")
            st.session_state.test_workflow_state = {
                'step': 1,
                'questions_extracted': False,
                'answers_extracted': False,
                'extraction_results': None,
                'test_completed': False
            }
            st.rerun()
    
    # Progress indicator in sidebar
    st.sidebar.markdown("### 📊 Progress")
    
    # Step 1 indicator
    if st.session_state.test_workflow_state['step'] >= 1:
        if st.session_state.test_workflow_state['questions_extracted'] and st.session_state.test_workflow_state['answers_extracted']:
            st.sidebar.success("✅ Step 1: Files Uploaded & Extracted")
        else:
            st.sidebar.info("⏳ Step 1: Upload Files & Extract")
    else:
        st.sidebar.info("⏳ Step 1: Upload Files & Extract")
    
    # Step 2 indicator
    if st.session_state.test_workflow_state['step'] >= 2:
        if st.session_state.test_workflow_state.get('test_completed', False):
            st.sidebar.success("✅ Step 2: Test Completed")
        else:
            st.sidebar.info("⏳ Step 2: Configure & Run Test")
    else:
        st.sidebar.info("⏳ Step 2: Configure & Run Test")
    
    # Step 3 indicator
    if st.session_state.test_workflow_state.get('test_completed', False):
        st.sidebar.success("✅ Step 3: Ready for New Test")
    else:
        st.sidebar.info("⏳ Step 3: New Test Option")

# Main function
def main():
    logger.info("Application main function started")
    
    # Initialize session state
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
        logger.info("Initialized chat history in session state")
    
    # Navigation
    selected_page = st.sidebar.selectbox(
        "Select Page",
        ["Chat Interface", "Practice Test"],
        key="page_selector"
    )
    
    logger.info(f"User selected page: {selected_page}")
    
    # Route to appropriate interface
    if selected_page == "Chat Interface":
        chat_interface()
    elif selected_page == "Practice Test":
        practice_test_interface()

if __name__ == "__main__":
    logger.info("=== Starting Medical Coding AI Agent Application ===")
    main()
    logger.info("=== Application execution completed ===")